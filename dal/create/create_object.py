from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from bson import ObjectId  # type: ignore
except Exception:
    class ObjectId(str):
        @staticmethod
        def is_valid(value: Any) -> bool:
            return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{24}", value) is not None

        def __new__(cls, value: Optional[str] = None):
            if value is None:
                import secrets
                value = secrets.token_hex(12)
            if not cls.is_valid(value):
                raise ValueError(f"Invalid ObjectId value: {value}")
            return str.__new__(cls, value)

try:
    from pymongo import MongoClient  # type: ignore
    from pymongo.database import Database  # type: ignore
except Exception:
    MongoClient = None  # type: ignore
    Database = Any  # type: ignore

from mapping.mongodb_blueprint import (
    MongoCollectionBlueprint,
    MongoDBBlueprint,
    build_mongodb_blueprint,
    load_ontology_blueprint,
)


@dataclass
class ObjectValidationIssue:
    level: str  # error | warning | info
    code: str
    message: str
    path: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectValidationResult:
    ok: bool
    requested_class: str
    final_class: Optional[str]
    collection_name: Optional[str]
    normalized_object_fields: Optional[Dict[str, Any]]
    errors: List[ObjectValidationIssue] = field(default_factory=list)
    warnings: List[ObjectValidationIssue] = field(default_factory=list)
    infos: List[ObjectValidationIssue] = field(default_factory=list)


class CreateObjectValidationError(ValueError):
    pass


class CreateObjectValidator:
    """
    Validate only the *object-property / reference-field* part of a create payload.

    Intended usage:
    - create_class.py resolves the final concrete class
    - create_data.py validates scalar/data-property fields
    - create_object.py validates reference/object-property fields

    Current scope:
    - allowed reference fields
    - single-value vs array checks
    - objectId shape checks
    - min/max cardinality checks that are directly available in blueprint refs
    - target collection existence checks (when MongoDB connection is provided)
    - basic target-collection matching using blueprint target_collection/target_classes

    Out of scope for this v1:
    - full inverse maintenance
    - full only/value-restriction semantic reasoning across subtype graphs
    - cross-document inverse cardinality enforcement
    - automatic back-reference updates
    """

    def __init__(
            self,
            ontology_blueprint_path: str | Path,
            *,
            mongo_uri: Optional[str] = None,
            db_name: Optional[str] = None,
            strict_unknown_fields: bool = True,
            require_reference_targets_to_exist: bool = True,
            coerce_string_object_ids: bool = True,
    ) -> None:
        self.ontology_blueprint_path = str(ontology_blueprint_path)
        self.ontology_bp: Dict[str, Any] = load_ontology_blueprint(self.ontology_blueprint_path)
        self.mongo_bp: MongoDBBlueprint = build_mongodb_blueprint(
            self.ontology_bp,
            source_blueprint_path=self.ontology_blueprint_path,
        )
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.strict_unknown_fields = strict_unknown_fields
        self.require_reference_targets_to_exist = require_reference_targets_to_exist
        self.coerce_string_object_ids = coerce_string_object_ids

        self.class_to_collection: Dict[str, str] = {
            col.source_class: col.name for col in self.mongo_bp.collections.values()
        }
        self.abstract_classes = set(self.mongo_bp.abstract_classes)
        self.union_axioms = self._build_union_axioms()
        self.parent_map, self.child_map = self._build_parent_child_maps()

    # --------------------------------------------------
    # blueprint helpers
    # --------------------------------------------------

    def _build_union_axioms(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        axioms = self.ontology_bp.get("axioms", {}).get("equivalent_class_axioms", []) or []
        for ax in axioms:
            if ax.get("expression_type") == "object_union_of":
                class_name = ax.get("class_name")
                operands = ax.get("operands", []) or []
                if class_name:
                    out[class_name] = list(operands)
        return out

    def _build_parent_child_maps(self) -> Tuple[Dict[str, set[str]], Dict[str, set[str]]]:
        parents: Dict[str, set[str]] = {}
        children: Dict[str, set[str]] = {}
        axioms = self.ontology_bp.get("axioms", {}).get("subclass_axioms", []) or []
        for ax in axioms:
            sub = ax["sub_class"]
            sup = ax["super_class"]
            parents.setdefault(sub, set()).add(sup)
            parents.setdefault(sup, set())
            children.setdefault(sup, set()).add(sub)
            children.setdefault(sub, set())
        return parents, children

    def _all_ancestors(self, class_name: str) -> set[str]:
        seen: set[str] = set()
        stack = list(self.parent_map.get(class_name, set()))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(self.parent_map.get(cur, set()))
        return seen

    def _is_descendant_of(self, child: str, parent: str) -> bool:
        return parent in self._all_ancestors(child)

    def _get_collection(self, class_name: str) -> Optional[MongoCollectionBlueprint]:
        col_name = self.class_to_collection.get(class_name)
        return self.mongo_bp.collections.get(col_name) if col_name else None

    # --------------------------------------------------
    # generic helpers
    # --------------------------------------------------

    @staticmethod
    def _get_dotted(doc: Dict[str, Any], dotted: str) -> Any:
        cur: Any = doc
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        return cur

    @staticmethod
    def _has_dotted(doc: Dict[str, Any], dotted: str) -> bool:
        cur: Any = doc
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return False
            cur = cur[part]
        return True

    @staticmethod
    def _set_dotted(doc: Dict[str, Any], dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        cur = doc
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value

    @staticmethod
    def _flatten_keys(doc: Dict[str, Any], prefix: str = "") -> set[str]:
        out: set[str] = set()
        for key, value in doc.items():
            path = f"{prefix}.{key}" if prefix else key
            out.add(path)
            if isinstance(value, dict):
                out.update(CreateObjectValidator._flatten_keys(value, path))
        return out

    @staticmethod
    def _coerce_object_id(value: Any) -> Tuple[Optional[ObjectId], bool]:
        if isinstance(value, ObjectId):
            return value, True
        if isinstance(value, str) and ObjectId.is_valid(value):
            return ObjectId(value), True
        return None, False

    def _normalize_single_object_id(self, value: Any) -> Tuple[Optional[ObjectId], bool]:
        if isinstance(value, ObjectId):
            return value, True
        if self.coerce_string_object_ids and isinstance(value, str) and ObjectId.is_valid(value):
            return ObjectId(value), True
        return None, False

    def _normalize_object_id_array(self, value: Any) -> Tuple[Optional[List[ObjectId]], bool]:
        if not isinstance(value, list):
            return None, False
        out: List[ObjectId] = []
        for item in value:
            oid, ok = self._normalize_single_object_id(item)
            if not ok or oid is None:
                return None, False
            out.append(oid)
        return out, True

    # --------------------------------------------------
    # class resolution
    # --------------------------------------------------

    def _resolve_final_class(
            self,
            requested_class: str,
            subtype: Optional[str],
            errors: List[ObjectValidationIssue],
            infos: List[ObjectValidationIssue],
    ) -> Tuple[Optional[str], Optional[MongoCollectionBlueprint]]:
        classes = self.ontology_bp.get("classes", {}) or {}

        if requested_class not in classes:
            errors.append(ObjectValidationIssue(
                level="error",
                code="unknown_class",
                message=f"Requested class '{requested_class}' does not exist.",
            ))
            return None, None

        if requested_class in self.union_axioms:
            allowed_subtypes = self.union_axioms[requested_class]
            if not subtype:
                errors.append(ObjectValidationIssue(
                    level="error",
                    code="concrete_subtype_required",
                    message=(
                        f"Class '{requested_class}' is a union-equivalent class. "
                        f"Choose one concrete subtype: {', '.join(allowed_subtypes)}."
                    ),
                    details={"allowed_subtypes": allowed_subtypes},
                ))
                return None, None
            if subtype not in allowed_subtypes:
                errors.append(ObjectValidationIssue(
                    level="error",
                    code="invalid_union_subtype",
                    message=(
                        f"Subtype '{subtype}' is not valid for '{requested_class}'. "
                        f"Allowed: {', '.join(allowed_subtypes)}."
                    ),
                    details={"allowed_subtypes": allowed_subtypes},
                ))
                return None, None
            requested_class = subtype
            infos.append(ObjectValidationIssue(
                level="info",
                code="parent_routed_to_subtype",
                message=(
                    f"Requested union class routed to concrete subtype '{requested_class}'."
                ),
            ))
        elif subtype:
            if subtype not in classes:
                errors.append(ObjectValidationIssue(
                    level="error",
                    code="unknown_subtype",
                    message=f"Subtype '{subtype}' does not exist.",
                ))
                return None, None
            if not self._is_descendant_of(subtype, requested_class) and subtype != requested_class:
                errors.append(ObjectValidationIssue(
                    level="error",
                    code="invalid_subtype",
                    message=(
                        f"Subtype '{subtype}' is not a subclass of '{requested_class}'."
                    ),
                ))
                return None, None
            requested_class = subtype
            infos.append(ObjectValidationIssue(
                level="info",
                code="requested_subtype_selected",
                message=f"Concrete subtype '{requested_class}' selected.",
            ))

        if requested_class in self.abstract_classes:
            errors.append(ObjectValidationIssue(
                level="error",
                code="abstract_class_not_instantiable",
                message=f"Class '{requested_class}' is abstract and cannot be instantiated directly.",
            ))
            return None, None

        collection = self._get_collection(requested_class)
        if not collection:
            errors.append(ObjectValidationIssue(
                level="error",
                code="no_collection_for_class",
                message=f"No MongoDB collection mapping found for class '{requested_class}'.",
            ))
            return None, None

        return requested_class, collection

    # --------------------------------------------------
    # db helpers
    # --------------------------------------------------

    def _open_db(self) -> Optional[Database]:
        if not self.mongo_uri or not self.db_name or MongoClient is None:
            return None
        client = MongoClient(self.mongo_uri)
        return client[self.db_name]

    def _target_document_exists(self, db: Optional[Database], collection_name: str, oid: ObjectId) -> bool:
        if db is None:
            return True
        return db[collection_name].find_one({"_id": oid}, {"_id": 1}) is not None

    def _infer_allowed_target_collections(self, ref: MongoReferenceBlueprint) -> List[str]:
        out: List[str] = []
        if ref.target_collection:
            out.append(ref.target_collection)
        for cls in ref.target_classes:
            col = self._get_collection(cls)
            if col and col.name not in out:
                out.append(col.name)
        return out

    # --------------------------------------------------
    # core validation
    # --------------------------------------------------

    def validate_object_fields(
            self,
            requested_class: str,
            payload: Dict[str, Any],
            *,
            subtype: Optional[str] = None,
            include_non_reference_field_warnings: bool = True,
    ) -> ObjectValidationResult:
        errors: List[ObjectValidationIssue] = []
        warnings: List[ObjectValidationIssue] = []
        infos: List[ObjectValidationIssue] = []

        final_class, collection = self._resolve_final_class(requested_class, subtype, errors, infos)
        if not final_class or not collection:
            return ObjectValidationResult(
                ok=False,
                requested_class=requested_class,
                final_class=final_class,
                collection_name=None,
                normalized_object_fields=None,
                errors=errors,
                warnings=warnings,
                infos=infos,
            )

        allowed_refs: Dict[str, MongoReferenceBlueprint] = {r.name: r for r in collection.references}
        normalized: Dict[str, Any] = {}

        flat_keys = self._flatten_keys(payload)

        # 1) unknown reference-ish fields / non-reference note
        for key in sorted(flat_keys):
            if key in allowed_refs:
                continue
            if any(key == fld.name or key.startswith(fld.name + ".") for fld in collection.fields):
                if include_non_reference_field_warnings:
                    infos.append(ObjectValidationIssue(
                        level="info",
                        code="scalar_field_ignored_by_object_validator",
                        message=f"Field '{key}' is a scalar/data field and is ignored by create_object.py.",
                        path=key,
                    ))
                continue
            if self.strict_unknown_fields and (key.endswith("_id") or key.endswith("_ids") or key.startswith("has_")):
                errors.append(ObjectValidationIssue(
                    level="error",
                    code="unknown_reference_field",
                    message=f"Field '{key}' is not an allowed object/reference field for class '{final_class}'.",
                    path=key,
                ))

        db = self._open_db()

        # 2) validate each allowed reference actually present
        for ref_name, ref in allowed_refs.items():
            if not self._has_dotted(payload, ref_name):
                if ref.required:
                    errors.append(ObjectValidationIssue(
                        level="error",
                        code="missing_required_reference",
                        message=f"Required reference field '{ref_name}' is missing.",
                        path=ref_name,
                    ))
                continue

            raw_value = self._get_dotted(payload, ref_name)
            allowed_target_collections = self._infer_allowed_target_collections(ref)

            if ref.is_array:
                normalized_arr, ok = self._normalize_object_id_array(raw_value)
                if not ok or normalized_arr is None:
                    errors.append(ObjectValidationIssue(
                        level="error",
                        code="invalid_reference_array",
                        message=f"Reference field '{ref_name}' must be a list of ObjectId values.",
                        path=ref_name,
                    ))
                    continue

                if ref.min_items is not None and len(normalized_arr) < ref.min_items:
                    errors.append(ObjectValidationIssue(
                        level="error",
                        code="reference_min_items_violation",
                        message=(
                            f"Reference field '{ref_name}' must contain at least {ref.min_items} item(s); "
                            f"got {len(normalized_arr)}."
                        ),
                        path=ref_name,
                        details={"min_items": ref.min_items, "actual": len(normalized_arr)},
                    ))
                if ref.max_items is not None and len(normalized_arr) > ref.max_items:
                    errors.append(ObjectValidationIssue(
                        level="error",
                        code="reference_max_items_violation",
                        message=(
                            f"Reference field '{ref_name}' must contain at most {ref.max_items} item(s); "
                            f"got {len(normalized_arr)}."
                        ),
                        path=ref_name,
                        details={"max_items": ref.max_items, "actual": len(normalized_arr)},
                    ))

                if self.require_reference_targets_to_exist and allowed_target_collections:
                    missing: List[str] = []
                    for oid in normalized_arr:
                        found = False
                        for target_col in allowed_target_collections:
                            if self._target_document_exists(db, target_col, oid):
                                found = True
                                break
                        if not found:
                            missing.append(str(oid))
                    if missing:
                        errors.append(ObjectValidationIssue(
                            level="error",
                            code="missing_reference_targets",
                            message=(
                                f"Reference field '{ref_name}' contains ObjectId(s) that were not found in allowed "
                                f"target collections: {', '.join(missing)}."
                            ),
                            path=ref_name,
                            details={"allowed_target_collections": allowed_target_collections, "missing_ids": missing},
                        ))

                self._set_dotted(normalized, ref_name, normalized_arr)

            else:
                if isinstance(raw_value, list):
                    errors.append(ObjectValidationIssue(
                        level="error",
                        code="single_reference_received_array",
                        message=f"Reference field '{ref_name}' is single-valued and cannot be a list.",
                        path=ref_name,
                    ))
                    continue

                normalized_oid, ok = self._normalize_single_object_id(raw_value)
                if not ok or normalized_oid is None:
                    errors.append(ObjectValidationIssue(
                        level="error",
                        code="invalid_reference_objectid",
                        message=f"Reference field '{ref_name}' must be an ObjectId value.",
                        path=ref_name,
                    ))
                    continue

                if self.require_reference_targets_to_exist and allowed_target_collections:
                    found = False
                    for target_col in allowed_target_collections:
                        if self._target_document_exists(db, target_col, normalized_oid):
                            found = True
                            break
                    if not found:
                        errors.append(ObjectValidationIssue(
                            level="error",
                            code="missing_reference_target",
                            message=(
                                f"ObjectId '{normalized_oid}' in field '{ref_name}' was not found in any allowed target collection."
                            ),
                            path=ref_name,
                            details={"allowed_target_collections": allowed_target_collections,
                                     "object_id": str(normalized_oid)},
                        ))

                self._set_dotted(normalized, ref_name, normalized_oid)

            # 3) surface service-layer notes as warnings/info
            if ref.validation_layer == "service" and ref.service_rules:
                warnings.append(ObjectValidationIssue(
                    level="warning",
                    code="service_layer_reference_rule",
                    message=(
                        f"Reference field '{ref_name}' has additional service-layer semantics that are not fully enforced here."
                    ),
                    path=ref_name,
                    details={"service_rules": list(ref.service_rules)},
                ))

        return ObjectValidationResult(
            ok=len(errors) == 0,
            requested_class=requested_class,
            final_class=final_class,
            collection_name=collection.name,
            normalized_object_fields=normalized,
            errors=errors,
            warnings=warnings,
            infos=infos,
        )

    def assert_object_valid(
            self,
            requested_class: str,
            payload: Dict[str, Any],
            *,
            subtype: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = self.validate_object_fields(requested_class, payload, subtype=subtype)
        if not result.ok or result.normalized_object_fields is None:
            lines = [f"[{e.code}] {e.message}" for e in result.errors]
            raise CreateObjectValidationError("Object/reference validation failed:\n" + "\n".join(lines))
        return result.normalized_object_fields


# --------------------------------------------------
# convenience factory
# --------------------------------------------------


def build_create_object_validator(
        ontology_blueprint_path: str | Path,
        *,
        mongo_uri: Optional[str] = None,
        db_name: Optional[str] = None,
        strict_unknown_fields: bool = True,
        require_reference_targets_to_exist: bool = True,
        coerce_string_object_ids: bool = True,
) -> CreateObjectValidator:
    return CreateObjectValidator(
        ontology_blueprint_path,
        mongo_uri=mongo_uri,
        db_name=db_name,
        strict_unknown_fields=strict_unknown_fields,
        require_reference_targets_to_exist=require_reference_targets_to_exist,
        coerce_string_object_ids=coerce_string_object_ids,
    )


if __name__ == "__main__":
    # Minimal manual smoke example.
    validator = build_create_object_validator(
        "/mnt/data/ontology_blueprint.json",
        mongo_uri=None,
        db_name=None,
    )
    res = validator.validate_object_fields(
        "Participant",
        {
            "has_sex_id": "507f1f77bcf86cd799439011",
        },
    )
    print("ok:", res.ok)
    print("errors:", [f"{e.code}: {e.message}" for e in res.errors])
    print("normalized:", res.normalized_object_fields)
