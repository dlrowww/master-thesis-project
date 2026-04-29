from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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

# Reuse the existing project builder so this validator always follows
# the same blueprint logic as the generator.
from mapping.mongodb_blueprint import (
    MongoCollectionBlueprint,
    MongoDBBlueprint,
    build_mongodb_blueprint,
    load_ontology_blueprint,
)


# ============================================================
# Public result model
# ============================================================


@dataclass
class ValidationIssue:
    level: str  # "error" | "warning" | "info"
    code: str
    message: str
    path: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CreateValidationResult:
    ok: bool
    requested_class: str
    final_class: Optional[str]
    collection_name: Optional[str]
    prepared_document: Optional[Dict[str, Any]]
    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    infos: List[ValidationIssue] = field(default_factory=list)


class CreateValidationError(ValueError):
    """Raised when strict validation fails."""


# ============================================================
# Helpers
# ============================================================


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _set_dotted(doc: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = doc
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def _get_dotted(doc: Dict[str, Any], dotted: str) -> Any:
    cur: Any = doc
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _has_dotted(doc: Dict[str, Any], dotted: str) -> bool:
    cur: Any = doc
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


def _flatten_keys(doc: Dict[str, Any], prefix: str = "") -> Set[str]:
    out: Set[str] = set()
    for key, value in doc.items():
        path = f"{prefix}.{key}" if prefix else key
        out.add(path)
        if isinstance(value, dict):
            out.update(_flatten_keys(value, path))
    return out


def _coerce_object_id(value: Any) -> Tuple[Optional[ObjectId], bool]:
    if isinstance(value, ObjectId):
        return value, True
    if isinstance(value, str) and ObjectId.is_valid(value):
        return ObjectId(value), True
    return None, False


def _is_scalar_bson_type_match(bson_type: str, value: Any) -> bool:
    if value is None:
        return False
    if bson_type == "string":
        return isinstance(value, str)
    if bson_type == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if bson_type == "long":
        return isinstance(value, int) and not isinstance(value, bool)
    if bson_type in {"double", "decimal"}:
        return (isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)
    if bson_type == "bool":
        return isinstance(value, bool)
    if bson_type == "date":
        return isinstance(value, datetime)
    if bson_type == "objectId":
        _obj_id, ok = _coerce_object_id(value)
        return ok
    return True


# ============================================================
# Constraint engine
# ============================================================


class CreateConstraintEngine:
    """
    Blueprint-aware validation for create-instance operations.

    Main design choices for v1:
    - schema changes are NOT allowed here
    - this validator only checks instance creation against the generated blueprint
    - class-level checks: existence, abstract class, equivalent/union subtype routing,
      vocabulary collection policy, basic disjoint-type conflicts
    - data-property checks: allowed fields, bson types, required fields
    - object-property checks: allowed reference fields, cardinality, ObjectId shape,
      target document existence, target collection correctness

    What this file intentionally does NOT fully enforce yet:
    - cross-document inverse maintenance
    - full disjointness inference across existing persisted type assertions
    - full `only` / value restriction semantic checking when target typing is ambiguous
    - cross-document inverse cardinality counts
    """

    def __init__(
            self,
            ontology_blueprint_path: str | Path,
            *,
            mongo_uri: Optional[str] = None,
            db_name: Optional[str] = None,
            allow_create_in_vocabulary_collections: bool = False,
            strict_unknown_fields: bool = True,
            auto_add_metadata: bool = True,
            auto_add_object_id: bool = True,
    ) -> None:
        self.ontology_blueprint_path = str(ontology_blueprint_path)
        self.ontology_bp: Dict[str, Any] = load_ontology_blueprint(self.ontology_blueprint_path)
        self.mongo_bp: MongoDBBlueprint = build_mongodb_blueprint(
            self.ontology_bp,
            source_blueprint_path=self.ontology_blueprint_path,
        )
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.allow_create_in_vocabulary_collections = allow_create_in_vocabulary_collections
        self.strict_unknown_fields = strict_unknown_fields
        self.auto_add_metadata = auto_add_metadata
        self.auto_add_object_id = auto_add_object_id

        self.class_to_collection: Dict[str, str] = {
            col.source_class: col.name for col in self.mongo_bp.collections.values()
        }
        self.collection_to_class: Dict[str, str] = {
            col.name: col.source_class for col in self.mongo_bp.collections.values()
        }
        self.abstract_classes: Set[str] = set(self.mongo_bp.abstract_classes)
        self.disjoint_groups: List[Set[str]] = [set(g) for g in self.mongo_bp.disjoint_groups]
        self.union_axioms: Dict[str, List[str]] = self._build_union_axioms()
        self.parent_map, self.child_map = self._build_parent_child_maps()

    # -------------------------
    # blueprint navigation
    # -------------------------

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

    def _build_parent_child_maps(self) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        parents: Dict[str, Set[str]] = {}
        children: Dict[str, Set[str]] = {}
        axioms = self.ontology_bp.get("axioms", {}).get("subclass_axioms", []) or []
        for ax in axioms:
            sub = ax["sub_class"]
            sup = ax["super_class"]
            parents.setdefault(sub, set()).add(sup)
            parents.setdefault(sup, set())
            children.setdefault(sup, set()).add(sub)
            children.setdefault(sub, set())
        return parents, children

    def _all_ancestors(self, class_name: str) -> Set[str]:
        out: Set[str] = set()
        stack = list(self.parent_map.get(class_name, set()))
        while stack:
            item = stack.pop()
            if item in out:
                continue
            out.add(item)
            stack.extend(self.parent_map.get(item, set()))
        return out

    def _is_descendant_of(self, child: str, parent: str) -> bool:
        return parent in self._all_ancestors(child)

    def _get_collection_by_class(self, class_name: str) -> Optional[MongoCollectionBlueprint]:
        col_name = self.class_to_collection.get(class_name)
        return self.mongo_bp.collections.get(col_name) if col_name else None

    def _allowed_target_collections(self, ref: MongoReferenceBlueprint) -> List[str]:
        out: List[str] = []
        if ref.target_collection:
            out.append(ref.target_collection)
        for cls in ref.target_classes:
            col = self.class_to_collection.get(cls)
            if col and col not in out:
                out.append(col)
        return out

    # -------------------------
    # DB handling
    # -------------------------

    def _open_db(self, db: Optional[Database] = None) -> Tuple[Optional[MongoClient], Optional[Database]]:
        if db is not None:
            return None, db
        if not self.mongo_uri or not self.db_name:
            return None, None
        if MongoClient is None:
            raise CreateValidationError(
                "pymongo is not installed in the current environment. DB-bound validation requires pymongo."
            )
        client = MongoClient(self.mongo_uri)
        return client, client[self.db_name]

    # -------------------------
    # public API
    # -------------------------

    def validate_create(
            self,
            requested_class: str,
            payload: Dict[str, Any],
            *,
            subtype: Optional[str] = None,
            db: Optional[Database] = None,
            created_by: str = "create_instance",
            strict: bool = False,
            check_db_collections: bool = True,
            check_reference_targets: bool = True,
    ) -> CreateValidationResult:
        errors: List[ValidationIssue] = []
        warnings: List[ValidationIssue] = []
        infos: List[ValidationIssue] = []

        final_class, collection = self._resolve_final_class(
            requested_class=requested_class,
            subtype=subtype,
            errors=errors,
            warnings=warnings,
            infos=infos,
        )

        if final_class is None or collection is None:
            result = CreateValidationResult(
                ok=False,
                requested_class=requested_class,
                final_class=final_class,
                collection_name=collection.name if collection else None,
                prepared_document=None,
                errors=errors,
                warnings=warnings,
                infos=infos,
            )
            if strict:
                raise CreateValidationError(self._format_errors(result))
            return result

        prepared = self._prepare_document(
            payload=payload,
            collection=collection,
            final_class=final_class,
            created_by=created_by,
        )

        self._validate_allowed_fields(collection, prepared, errors, warnings)
        self._validate_scalar_fields(collection, prepared, errors)
        self._validate_reference_fields(collection, prepared, errors)
        self._validate_required_fields(collection, prepared, errors)
        self._validate_basic_disjointness(final_class, requested_class, subtype, warnings)

        client, active_db = self._open_db(db)
        try:
            if active_db is not None:
                if check_db_collections:
                    self._validate_collection_deployed(collection, active_db, errors)
                if check_reference_targets:
                    self._validate_reference_targets(collection, prepared, active_db, errors, warnings)
            else:
                warnings.append(
                    ValidationIssue(
                        level="warning",
                        code="db_not_bound",
                        message="No database connection was provided. Reference existence and deployed-collection checks were skipped.",
                    )
                )
        finally:
            if client is not None:
                client.close()

        result = CreateValidationResult(
            ok=not errors,
            requested_class=requested_class,
            final_class=final_class,
            collection_name=collection.name,
            prepared_document=prepared if not errors else None,
            errors=errors,
            warnings=warnings,
            infos=infos,
        )

        if strict and errors:
            raise CreateValidationError(self._format_errors(result))
        return result

    # -------------------------
    # create-time resolution
    # -------------------------

    def _resolve_final_class(
            self,
            *,
            requested_class: str,
            subtype: Optional[str],
            errors: List[ValidationIssue],
            warnings: List[ValidationIssue],
            infos: List[ValidationIssue],
    ) -> Tuple[Optional[str], Optional[MongoCollectionBlueprint]]:
        classes = self.ontology_bp.get("classes", {}) or {}

        if requested_class not in classes:
            errors.append(
                ValidationIssue(
                    level="error",
                    code="unknown_class",
                    message=f"Requested class '{requested_class}' does not exist in the ontology blueprint.",
                )
            )
            return None, None

        # Equivalent/union parent class: require a concrete subtype.
        if requested_class in self.union_axioms:
            allowed_subtypes = self.union_axioms[requested_class]
            if not subtype:
                errors.append(
                    ValidationIssue(
                        level="error",
                        code="concrete_subtype_required",
                        message=(
                            f"Class '{requested_class}' is defined as a union-equivalent class and cannot be created directly. "
                            f"Choose one concrete subtype: {', '.join(allowed_subtypes)}."
                        ),
                        details={"allowed_subtypes": allowed_subtypes},
                    )
                )
                return None, None
            if subtype not in allowed_subtypes:
                errors.append(
                    ValidationIssue(
                        level="error",
                        code="invalid_union_subtype",
                        message=(
                            f"Subtype '{subtype}' is not allowed for union-equivalent parent '{requested_class}'. "
                            f"Allowed: {', '.join(allowed_subtypes)}."
                        ),
                        details={"allowed_subtypes": allowed_subtypes},
                    )
                )
                return None, None
            requested_class = subtype
            infos.append(
                ValidationIssue(
                    level="info",
                    code="parent_routed_to_subtype",
                    message=f"Create request was routed from '{self.union_axioms and next((k for k, v in self.union_axioms.items() if subtype in v and k == requested_class), requested_class)}' to concrete subtype '{subtype}'.",
                )
            )

        if subtype and requested_class not in self.union_axioms:
            if subtype not in classes:
                errors.append(
                    ValidationIssue(
                        level="error",
                        code="unknown_subtype",
                        message=f"Provided subtype '{subtype}' does not exist in the ontology blueprint.",
                    )
                )
                return None, None
            if subtype in self.abstract_classes:
                errors.append(
                    ValidationIssue(
                        level="error",
                        code="abstract_subtype",
                        message=f"Provided subtype '{subtype}' is abstract and cannot be instantiated.",
                    )
                )
                return None, None
            if subtype == requested_class or self._is_descendant_of(subtype, requested_class):
                requested_class = subtype
                infos.append(
                    ValidationIssue(
                        level="info",
                        code="subtype_selected",
                        message=f"Concrete subtype '{subtype}' will be used for the create operation.",
                    )
                )
            else:
                errors.append(
                    ValidationIssue(
                        level="error",
                        code="subtype_not_descendant",
                        message=(
                            f"Subtype '{subtype}' is not a descendant of requested class '{requested_class}'."
                        ),
                    )
                )
                return None, None

        if requested_class in self.abstract_classes:
            errors.append(
                ValidationIssue(
                    level="error",
                    code="abstract_class",
                    message=f"Class '{requested_class}' is abstract and cannot be instantiated directly.",
                )
            )
            return None, None

        collection = self._get_collection_by_class(requested_class)
        if collection is None:
            errors.append(
                ValidationIssue(
                    level="error",
                    code="class_not_materialized",
                    message=(
                        f"Class '{requested_class}' does not materialize to a MongoDB collection in the generated blueprint."
                    ),
                )
            )
            return None, None

        if (
                collection.materialization == "vocabulary_collection"
                and not self.allow_create_in_vocabulary_collections
        ):
            errors.append(
                ValidationIssue(
                    level="error",
                    code="vocabulary_create_blocked",
                    message=(
                        f"Class '{requested_class}' maps to controlled vocabulary collection '{collection.name}'. "
                        "Runtime create is blocked by policy; seed/update vocabulary through the generation layer instead."
                    ),
                )
            )
            return None, None

        return requested_class, collection

    def _prepare_document(
            self,
            *,
            payload: Dict[str, Any],
            collection: MongoCollectionBlueprint,
            final_class: str,
            created_by: str,
    ) -> Dict[str, Any]:
        doc = json.loads(json.dumps(payload, default=str)) if False else dict(payload)

        if self.auto_add_object_id and "_id" not in doc:
            doc["_id"] = ObjectId()

        if collection.subtype_field:
            doc.setdefault(collection.subtype_field, final_class)

        if self.auto_add_metadata:
            if not _has_dotted(doc, "metadata.createdAt"):
                _set_dotted(doc, "metadata.createdAt", _now_utc())
            if not _has_dotted(doc, "metadata.createdBy"):
                _set_dotted(doc, "metadata.createdBy", created_by)
            if not _has_dotted(doc, "metadata.updatedAt"):
                _set_dotted(doc, "metadata.updatedAt", _now_utc())
            if not _has_dotted(doc, "metadata.notes"):
                _set_dotted(doc, "metadata.notes", f"created via create_class validator for {final_class}")

        return doc

    # -------------------------
    # static validation
    # -------------------------

    def _allowed_field_names(self, collection: MongoCollectionBlueprint) -> Tuple[Set[str], Set[str]]:
        scalar_names = {field.name for field in collection.fields}
        reference_names = {ref.name for ref in collection.references}
        return scalar_names, reference_names

    def _validate_allowed_fields(
            self,
            collection: MongoCollectionBlueprint,
            doc: Dict[str, Any],
            errors: List[ValidationIssue],
            warnings: List[ValidationIssue],
    ) -> None:
        scalar_names, reference_names = self._allowed_field_names(collection)
        allowed_roots = {
            "_id",
            collection.subtype_field or "_ontology_type",
            "metadata",
        }
        allowed = scalar_names | reference_names | allowed_roots

        payload_roots = set(doc.keys())
        unknown = sorted([k for k in payload_roots if k not in allowed])
        if not unknown:
            return

        issue = ValidationIssue(
            level="error" if self.strict_unknown_fields else "warning",
            code="unknown_fields",
            message=f"Unknown field(s) for collection '{collection.name}': {', '.join(unknown)}",
            details={"unknown_fields": unknown},
        )
        if self.strict_unknown_fields:
            errors.append(issue)
        else:
            warnings.append(issue)

    def _validate_scalar_fields(
            self,
            collection: MongoCollectionBlueprint,
            doc: Dict[str, Any],
            errors: List[ValidationIssue],
    ) -> None:
        for field in collection.fields:
            if not _has_dotted(doc, field.name):
                continue
            value = _get_dotted(doc, field.name)
            if field.is_array:
                if not isinstance(value, list):
                    errors.append(
                        ValidationIssue(
                            level="error",
                            code="data_array_expected",
                            message=f"Field '{field.name}' expects an array value.",
                            path=field.name,
                        )
                    )
                    continue
                for idx, item in enumerate(value):
                    if not _is_scalar_bson_type_match(field.bson_type, item):
                        errors.append(
                            ValidationIssue(
                                level="error",
                                code="invalid_data_item_type",
                                message=(
                                    f"Field '{field.name}[{idx}]' does not match expected BSON type '{field.bson_type}'."
                                ),
                                path=f"{field.name}[{idx}]",
                                details={"expected_bson_type": field.bson_type, "actual_value": repr(item)},
                            )
                        )
            else:
                if isinstance(value, list):
                    errors.append(
                        ValidationIssue(
                            level="error",
                            code="data_scalar_expected",
                            message=f"Field '{field.name}' is single-valued and cannot be an array.",
                            path=field.name,
                        )
                    )
                    continue
                if not _is_scalar_bson_type_match(field.bson_type, value):
                    errors.append(
                        ValidationIssue(
                            level="error",
                            code="invalid_data_type",
                            message=f"Field '{field.name}' does not match expected BSON type '{field.bson_type}'.",
                            path=field.name,
                            details={"expected_bson_type": field.bson_type, "actual_value": repr(value)},
                        )
                    )

    def _validate_reference_fields(
            self,
            collection: MongoCollectionBlueprint,
            doc: Dict[str, Any],
            errors: List[ValidationIssue],
    ) -> None:
        for ref in collection.references:
            if not _has_dotted(doc, ref.name):
                continue
            value = _get_dotted(doc, ref.name)

            if ref.is_array:
                if not isinstance(value, list):
                    errors.append(
                        ValidationIssue(
                            level="error",
                            code="reference_array_expected",
                            message=f"Reference '{ref.name}' expects an array of ObjectId values.",
                            path=ref.name,
                        )
                    )
                    continue
                if ref.min_items is not None and len(value) < ref.min_items:
                    errors.append(
                        ValidationIssue(
                            level="error",
                            code="reference_min_items",
                            message=(
                                f"Reference '{ref.name}' requires at least {ref.min_items} item(s), got {len(value)}."
                            ),
                            path=ref.name,
                        )
                    )
                if ref.max_items is not None and len(value) > ref.max_items:
                    errors.append(
                        ValidationIssue(
                            level="error",
                            code="reference_max_items",
                            message=(
                                f"Reference '{ref.name}' allows at most {ref.max_items} item(s), got {len(value)}."
                            ),
                            path=ref.name,
                        )
                    )
                for idx, item in enumerate(value):
                    _obj_id, ok = _coerce_object_id(item)
                    if not ok:
                        errors.append(
                            ValidationIssue(
                                level="error",
                                code="invalid_reference_id",
                                message=f"Reference '{ref.name}[{idx}]' is not a valid ObjectId or ObjectId string.",
                                path=f"{ref.name}[{idx}]",
                            )
                        )
            else:
                if isinstance(value, list):
                    errors.append(
                        ValidationIssue(
                            level="error",
                            code="reference_scalar_expected",
                            message=f"Reference '{ref.name}' is single-valued and cannot be an array.",
                            path=ref.name,
                        )
                    )
                    continue
                _obj_id, ok = _coerce_object_id(value)
                if not ok:
                    errors.append(
                        ValidationIssue(
                            level="error",
                            code="invalid_reference_id",
                            message=f"Reference '{ref.name}' is not a valid ObjectId or ObjectId string.",
                            path=ref.name,
                        )
                    )

    def _validate_required_fields(
            self,
            collection: MongoCollectionBlueprint,
            doc: Dict[str, Any],
            errors: List[ValidationIssue],
    ) -> None:
        for field in collection.fields:
            if field.required and not _has_dotted(doc, field.name):
                errors.append(
                    ValidationIssue(
                        level="error",
                        code="required_field_missing",
                        message=f"Required field '{field.name}' is missing.",
                        path=field.name,
                    )
                )
        for ref in collection.references:
            if ref.required and not _has_dotted(doc, ref.name):
                errors.append(
                    ValidationIssue(
                        level="error",
                        code="required_reference_missing",
                        message=f"Required reference '{ref.name}' is missing.",
                        path=ref.name,
                    )
                )

    def _validate_basic_disjointness(
            self,
            final_class: str,
            requested_class: str,
            subtype: Optional[str],
            warnings: List[ValidationIssue],
    ) -> None:
        # v1 only warns when user requested a parent class plus subtype that belongs
        # to a disjoint set with some sibling classes. Full persisted-type conflict
        # checking requires a stronger runtime type model.
        active_types = {final_class}
        if requested_class != final_class:
            active_types.add(requested_class)
        if subtype:
            active_types.add(subtype)
        for group in self.disjoint_groups:
            overlap = active_types & group
            if len(overlap) > 1:
                warnings.append(
                    ValidationIssue(
                        level="warning",
                        code="possible_disjoint_type_overlap",
                        message=(
                            f"Type selection touches multiple classes from a disjoint group: {', '.join(sorted(overlap))}. "
                            "This usually means the caller should choose a single concrete class only."
                        ),
                        details={"overlap": sorted(overlap)},
                    )
                )

    # -------------------------
    # DB-aware validation
    # -------------------------

    def _validate_collection_deployed(
            self,
            collection: MongoCollectionBlueprint,
            db: Database,
            errors: List[ValidationIssue],
    ) -> None:
        existing = set(db.list_collection_names())
        if collection.name not in existing:
            errors.append(
                ValidationIssue(
                    level="error",
                    code="collection_not_deployed",
                    message=(
                        f"Target collection '{collection.name}' does not exist in the database. "
                        "Run the data-layer generation step first."
                    ),
                )
            )

    def _validate_reference_targets(
            self,
            collection: MongoCollectionBlueprint,
            doc: Dict[str, Any],
            db: Database,
            errors: List[ValidationIssue],
            warnings: List[ValidationIssue],
    ) -> None:
        existing_collections = set(db.list_collection_names())
        for ref in collection.references:
            if not _has_dotted(doc, ref.name):
                continue
            value = _get_dotted(doc, ref.name)
            allowed_collections = self._allowed_target_collections(ref)
            if not allowed_collections:
                warnings.append(
                    ValidationIssue(
                        level="warning",
                        code="target_collection_unresolved",
                        message=(
                            f"Reference '{ref.name}' has no resolved target collection in the blueprint. "
                            "Static shape validation succeeded, but semantic target validation was skipped."
                        ),
                        path=ref.name,
                        details={"target_classes": ref.target_classes},
                    )
                )
                continue

            for target_col in allowed_collections:
                if target_col not in existing_collections:
                    warnings.append(
                        ValidationIssue(
                            level="warning",
                            code="target_collection_missing",
                            message=(
                                f"Expected target collection '{target_col}' for reference '{ref.name}' does not exist in the bound database."
                            ),
                            path=ref.name,
                        )
                    )

            raw_values = value if ref.is_array else [value]
            for idx, raw in enumerate(raw_values):
                obj_id, ok = _coerce_object_id(raw)
                if not ok:
                    # shape error already reported earlier
                    continue
                matched_collections: List[str] = []
                for target_col in allowed_collections:
                    if target_col not in existing_collections:
                        continue
                    if db[target_col].find_one({"_id": obj_id}, {"_id": 1}):
                        matched_collections.append(target_col)
                if not matched_collections:
                    errors.append(
                        ValidationIssue(
                            level="error",
                            code="missing_reference_target",
                            message=(
                                f"Reference '{ref.name}' points to ObjectId '{obj_id}', but no matching target document "
                                f"was found in allowed collection(s): {', '.join(allowed_collections)}."
                            ),
                            path=f"{ref.name}[{idx}]" if ref.is_array else ref.name,
                            details={"allowed_collections": allowed_collections, "object_id": str(obj_id)},
                        )
                    )
                elif len(matched_collections) > 1:
                    warnings.append(
                        ValidationIssue(
                            level="warning",
                            code="ambiguous_reference_target",
                            message=(
                                f"Reference '{ref.name}' target ObjectId '{obj_id}' was found in multiple allowed collections: "
                                f"{', '.join(matched_collections)}."
                            ),
                            path=f"{ref.name}[{idx}]" if ref.is_array else ref.name,
                            details={"matched_collections": matched_collections},
                        )
                    )

    # -------------------------
    # convenience helpers for create_instance implementation
    # -------------------------

    def assert_create_valid(
            self,
            requested_class: str,
            payload: Dict[str, Any],
            **kwargs: Any,
    ) -> Dict[str, Any]:
        result = self.validate_create(requested_class, payload, strict=True, **kwargs)
        assert result.prepared_document is not None
        return result.prepared_document

    def create_instance(
            self,
            requested_class: str,
            payload: Dict[str, Any],
            *,
            subtype: Optional[str] = None,
            db: Optional[Database] = None,
            created_by: str = "create_instance",
    ) -> Dict[str, Any]:
        """
        Validate and insert a document.

        This helper is intentionally thin so later DAL code can either call:
        - validate_create(...) and decide itself, or
        - create_instance(...) directly.
        """
        result = self.validate_create(
            requested_class,
            payload,
            subtype=subtype,
            db=db,
            created_by=created_by,
            strict=True,
        )
        if result.prepared_document is None or result.collection_name is None:
            raise CreateValidationError("Create validation unexpectedly returned no prepared document.")

        client, active_db = self._open_db(db)
        try:
            if active_db is None:
                raise CreateValidationError(
                    "No MongoDB Database handle is available. Pass db=... or initialize the engine with mongo_uri and db_name."
                )
            active_db[result.collection_name].insert_one(result.prepared_document)
            return result.prepared_document
        finally:
            if client is not None:
                client.close()

    # -------------------------
    # formatting
    # -------------------------

    @staticmethod
    def _format_errors(result: CreateValidationResult) -> str:
        lines = [
            f"Create validation failed for requested class '{result.requested_class}'.",
            f"Final class: {result.final_class}",
            f"Collection: {result.collection_name}",
            "Errors:",
        ]
        for err in result.errors:
            path = f" [{err.path}]" if err.path else ""
            lines.append(f"- {err.code}{path}: {err.message}")
        if result.warnings:
            lines.append("Warnings:")
            for warn in result.warnings:
                path = f" [{warn.path}]" if warn.path else ""
                lines.append(f"- {warn.code}{path}: {warn.message}")
        return "\n".join(lines)


# ============================================================
# Factory helper
# ============================================================


def build_create_constraint_engine(
        ontology_blueprint_path: str | Path,
        *,
        mongo_uri: Optional[str] = None,
        db_name: Optional[str] = None,
        allow_create_in_vocabulary_collections: bool = False,
        strict_unknown_fields: bool = True,
        auto_add_metadata: bool = True,
        auto_add_object_id: bool = True,
) -> CreateConstraintEngine:
    return CreateConstraintEngine(
        ontology_blueprint_path=ontology_blueprint_path,
        mongo_uri=mongo_uri,
        db_name=db_name,
        allow_create_in_vocabulary_collections=allow_create_in_vocabulary_collections,
        strict_unknown_fields=strict_unknown_fields,
        auto_add_metadata=auto_add_metadata,
        auto_add_object_id=auto_add_object_id,
    )


# ============================================================
# Minimal CLI for quick manual testing
# ============================================================


def _load_json_file(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate create-instance payloads against the generated ontology/Mongo blueprint.")
    parser.add_argument("class_name", type=str, help="Requested class name")
    parser.add_argument("payload_json", type=str, help="Path to a JSON payload file")
    parser.add_argument("--ontology_blueprint", type=str, default="ontology_blueprint.json")
    parser.add_argument("--subtype", type=str, default=None)
    parser.add_argument("--mongo_uri", type=str, default=None)
    parser.add_argument("--db", type=str, default=None)
    parser.add_argument("--allow_vocab_create", action="store_true")
    parser.add_argument("--loose_unknown_fields", action="store_true")
    args = parser.parse_args()

    engine = build_create_constraint_engine(
        args.ontology_blueprint,
        mongo_uri=args.mongo_uri,
        db_name=args.db,
        allow_create_in_vocabulary_collections=args.allow_vocab_create,
        strict_unknown_fields=not args.loose_unknown_fields,
    )

    payload = _load_json_file(args.payload_json)
    result = engine.validate_create(
        args.class_name,
        payload,
        subtype=args.subtype,
        strict=False,
    )

    print(json.dumps({
        "ok": result.ok,
        "requested_class": result.requested_class,
        "final_class": result.final_class,
        "collection_name": result.collection_name,
        "errors": [issue.__dict__ for issue in result.errors],
        "warnings": [issue.__dict__ for issue in result.warnings],
        "infos": [issue.__dict__ for issue in result.infos],
        "prepared_document": result.prepared_document,
    }, indent=2, default=str, ensure_ascii=False))
