from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
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

from mapping.mongodb_blueprint import (
    MongoCollectionBlueprint,
    MongoDBBlueprint,
    MongoFieldBlueprint,
    build_mongodb_blueprint,
    load_ontology_blueprint,
)


@dataclass
class DataValidationIssue:
    level: str  # error | warning | info
    code: str
    message: str
    path: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataValidationResult:
    ok: bool
    requested_class: str
    final_class: Optional[str]
    collection_name: Optional[str]
    normalized_data: Optional[Dict[str, Any]]
    errors: List[DataValidationIssue] = field(default_factory=list)
    warnings: List[DataValidationIssue] = field(default_factory=list)
    infos: List[DataValidationIssue] = field(default_factory=list)


class CreateDataValidationError(ValueError):
    pass


class CreateDataValidator:
    """
    Validate only the *data-property / scalar-field* part of a create payload.

    Intended usage:
    - create_class.py decides the final concrete class
    - create_data.py validates the non-reference fields for that class
    - create_object.py (future) validates reference/object-property fields

    This module intentionally focuses on:
    - allowed scalar fields
    - scalar BSON/datatype checks
    - required data fields
    - single-value vs array checks for scalar fields
    - optional automatic metadata filling
    - optional automatic _id generation
    """

    def __init__(
            self,
            ontology_blueprint_path: str | Path,
            *,
            strict_unknown_fields: bool = True,
            auto_add_metadata: bool = True,
            auto_add_object_id: bool = True,
            include_metadata_fields_in_validation: bool = True,
    ) -> None:
        self.ontology_blueprint_path = str(ontology_blueprint_path)
        self.ontology_bp: Dict[str, Any] = load_ontology_blueprint(self.ontology_blueprint_path)
        self.mongo_bp: MongoDBBlueprint = build_mongodb_blueprint(
            self.ontology_bp,
            source_blueprint_path=self.ontology_blueprint_path,
        )
        self.strict_unknown_fields = strict_unknown_fields
        self.auto_add_metadata = auto_add_metadata
        self.auto_add_object_id = auto_add_object_id
        self.include_metadata_fields_in_validation = include_metadata_fields_in_validation

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

    def _is_descendant_of(self, child: str, parent: str) -> bool:
        seen: set[str] = set()
        stack = list(self.parent_map.get(child, set()))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if cur == parent:
                return True
            stack.extend(self.parent_map.get(cur, set()))
        return False

    def _get_collection(self, class_name: str) -> Optional[MongoCollectionBlueprint]:
        col_name = self.class_to_collection.get(class_name)
        return self.mongo_bp.collections.get(col_name) if col_name else None

    # --------------------------------------------------
    # generic helpers
    # --------------------------------------------------

    @staticmethod
    def _now_utc() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _set_dotted(doc: Dict[str, Any], dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        cur = doc
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value

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
    def _coerce_object_id(value: Any) -> Tuple[Optional[ObjectId], bool]:
        if isinstance(value, ObjectId):
            return value, True
        if isinstance(value, str) and ObjectId.is_valid(value):
            return ObjectId(value), True
        return None, False

    @staticmethod
    def _is_scalar_bson_type_match(bson_type: str, value: Any) -> bool:
        if value is None:
            return False
        if bson_type == "string":
            return isinstance(value, str)
        if bson_type in {"int", "long"}:
            return isinstance(value, int) and not isinstance(value, bool)
        if bson_type in {"double", "decimal"}:
            return (isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)
        if bson_type == "bool":
            return isinstance(value, bool)
        if bson_type == "date":
            return isinstance(value, datetime)
        if bson_type == "objectId":
            _obj, ok = CreateDataValidator._coerce_object_id(value)
            return ok
        return True

    # --------------------------------------------------
    # class resolution
    # --------------------------------------------------

    def _resolve_final_class(
            self,
            requested_class: str,
            subtype: Optional[str],
            errors: List[DataValidationIssue],
            infos: List[DataValidationIssue],
    ) -> Tuple[Optional[str], Optional[MongoCollectionBlueprint]]:
        classes = self.ontology_bp.get("classes", {}) or {}

        if requested_class not in classes:
            errors.append(DataValidationIssue(
                level="error",
                code="unknown_class",
                message=f"Requested class '{requested_class}' does not exist.",
            ))
            return None, None

        if requested_class in self.union_axioms:
            allowed_subtypes = self.union_axioms[requested_class]
            if not subtype:
                errors.append(DataValidationIssue(
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
                errors.append(DataValidationIssue(
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
            infos.append(DataValidationIssue(
                level="info",
                code="parent_routed_to_subtype",
                message=f"Using concrete subtype '{subtype}' for create-data validation.",
            ))

        elif subtype:
            if subtype not in classes:
                errors.append(DataValidationIssue(
                    level="error",
                    code="unknown_subtype",
                    message=f"Provided subtype '{subtype}' does not exist.",
                ))
                return None, None
            if subtype in self.abstract_classes:
                errors.append(DataValidationIssue(
                    level="error",
                    code="abstract_subtype",
                    message=f"Provided subtype '{subtype}' is abstract and cannot be instantiated.",
                ))
                return None, None
            if subtype == requested_class or self._is_descendant_of(subtype, requested_class):
                requested_class = subtype
                infos.append(DataValidationIssue(
                    level="info",
                    code="subtype_selected",
                    message=f"Concrete subtype '{subtype}' will be used.",
                ))
            else:
                errors.append(DataValidationIssue(
                    level="error",
                    code="subtype_not_descendant",
                    message=f"Subtype '{subtype}' is not a descendant of '{requested_class}'.",
                ))
                return None, None

        if requested_class in self.abstract_classes:
            errors.append(DataValidationIssue(
                level="error",
                code="abstract_class",
                message=f"Class '{requested_class}' is abstract and cannot be instantiated directly.",
            ))
            return None, None

        collection = self._get_collection(requested_class)
        if collection is None:
            errors.append(DataValidationIssue(
                level="error",
                code="class_not_materialized",
                message=f"Class '{requested_class}' does not materialize to a MongoDB collection.",
            ))
            return None, None

        return requested_class, collection

    # --------------------------------------------------
    # data normalization
    # --------------------------------------------------

    def _prepare_document(
            self,
            payload: Dict[str, Any],
            collection: MongoCollectionBlueprint,
            final_class: str,
            created_by: str,
    ) -> Dict[str, Any]:
        doc = dict(payload)

        if self.auto_add_object_id and "_id" not in doc:
            doc["_id"] = ObjectId()

        if collection.subtype_field:
            doc.setdefault(collection.subtype_field, final_class)

        if self.auto_add_metadata:
            if not self._has_dotted(doc, "metadata.createdAt"):
                self._set_dotted(doc, "metadata.createdAt", self._now_utc())
            if not self._has_dotted(doc, "metadata.createdBy"):
                self._set_dotted(doc, "metadata.createdBy", created_by)
            if not self._has_dotted(doc, "metadata.updatedAt"):
                self._set_dotted(doc, "metadata.updatedAt", self._now_utc())
            if not self._has_dotted(doc, "metadata.notes"):
                self._set_dotted(doc, "metadata.notes", f"created via create_data validator for {final_class}")

        return doc

    # --------------------------------------------------
    # scalar/data validation
    # --------------------------------------------------

    def _scalar_fields(self, collection: MongoCollectionBlueprint) -> List[MongoFieldBlueprint]:
        fields = list(collection.fields)
        if not self.include_metadata_fields_in_validation:
            fields = [f for f in fields if not f.name.startswith("metadata.")]
        return fields

    def _validate_allowed_scalar_fields(
            self,
            collection: MongoCollectionBlueprint,
            doc: Dict[str, Any],
            errors: List[DataValidationIssue],
            warnings: List[DataValidationIssue],
    ) -> None:
        allowed_scalar_roots = {f.name.split(".")[0] for f in self._scalar_fields(collection)}
        allowed_aux = {"_id", collection.subtype_field or "_ontology_type"}
        if self.include_metadata_fields_in_validation:
            allowed_aux.add("metadata")

        # Root-level object-property fields are intentionally ignored here;
        # this validator only complains about unknown roots after excluding known refs.
        known_ref_roots = {r.name.split(".")[0] for r in collection.references}

        unknown = sorted(
            k for k in doc.keys()
            if k not in allowed_scalar_roots and k not in allowed_aux and k not in known_ref_roots
        )
        if not unknown:
            return

        issue = DataValidationIssue(
            level="error" if self.strict_unknown_fields else "warning",
            code="unknown_scalar_fields",
            message=f"Unknown data/scalar field root(s) for '{collection.name}': {', '.join(unknown)}",
            details={"unknown_fields": unknown},
        )
        if self.strict_unknown_fields:
            errors.append(issue)
        else:
            warnings.append(issue)

    def _validate_scalar_types(
            self,
            collection: MongoCollectionBlueprint,
            doc: Dict[str, Any],
            errors: List[DataValidationIssue],
    ) -> None:
        for field in self._scalar_fields(collection):
            if not self._has_dotted(doc, field.name):
                continue
            value = self._get_dotted(doc, field.name)
            if field.is_array:
                if not isinstance(value, list):
                    errors.append(DataValidationIssue(
                        level="error",
                        code="data_array_expected",
                        message=f"Field '{field.name}' expects an array value.",
                        path=field.name,
                    ))
                    continue
                for idx, item in enumerate(value):
                    if not self._is_scalar_bson_type_match(field.bson_type, item):
                        errors.append(DataValidationIssue(
                            level="error",
                            code="invalid_data_item_type",
                            message=(
                                f"Field '{field.name}[{idx}]' does not match expected BSON type '{field.bson_type}'."
                            ),
                            path=f"{field.name}[{idx}]",
                            details={"expected_bson_type": field.bson_type, "actual_value": repr(item)},
                        ))
            else:
                if isinstance(value, list):
                    errors.append(DataValidationIssue(
                        level="error",
                        code="data_scalar_expected",
                        message=f"Field '{field.name}' is single-valued and cannot be an array.",
                        path=field.name,
                    ))
                    continue
                if not self._is_scalar_bson_type_match(field.bson_type, value):
                    errors.append(DataValidationIssue(
                        level="error",
                        code="invalid_data_type",
                        message=f"Field '{field.name}' does not match expected BSON type '{field.bson_type}'.",
                        path=field.name,
                        details={"expected_bson_type": field.bson_type, "actual_value": repr(value)},
                    ))

    def _validate_required_scalar_fields(
            self,
            collection: MongoCollectionBlueprint,
            doc: Dict[str, Any],
            errors: List[DataValidationIssue],
    ) -> None:
        for field in self._scalar_fields(collection):
            if field.required and not self._has_dotted(doc, field.name):
                errors.append(DataValidationIssue(
                    level="error",
                    code="required_data_field_missing",
                    message=f"Required data field '{field.name}' is missing.",
                    path=field.name,
                ))

    # --------------------------------------------------
    # public API
    # --------------------------------------------------

    def validate_data(
            self,
            requested_class: str,
            payload: Dict[str, Any],
            *,
            subtype: Optional[str] = None,
            created_by: str = "create_instance",
            strict: bool = False,
    ) -> DataValidationResult:
        errors: List[DataValidationIssue] = []
        warnings: List[DataValidationIssue] = []
        infos: List[DataValidationIssue] = []

        final_class, collection = self._resolve_final_class(requested_class, subtype, errors, infos)
        if final_class is None or collection is None:
            result = DataValidationResult(
                ok=False,
                requested_class=requested_class,
                final_class=final_class,
                collection_name=collection.name if collection else None,
                normalized_data=None,
                errors=errors,
                warnings=warnings,
                infos=infos,
            )
            if strict:
                raise CreateDataValidationError(self._format_errors(result))
            return result

        normalized = self._prepare_document(payload, collection, final_class, created_by)
        self._validate_allowed_scalar_fields(collection, normalized, errors, warnings)
        self._validate_scalar_types(collection, normalized, errors)
        self._validate_required_scalar_fields(collection, normalized, errors)

        result = DataValidationResult(
            ok=not errors,
            requested_class=requested_class,
            final_class=final_class,
            collection_name=collection.name,
            normalized_data=normalized if not errors else None,
            errors=errors,
            warnings=warnings,
            infos=infos,
        )
        if strict and errors:
            raise CreateDataValidationError(self._format_errors(result))
        return result

    def assert_data_valid(
            self,
            requested_class: str,
            payload: Dict[str, Any],
            **kwargs: Any,
    ) -> Dict[str, Any]:
        result = self.validate_data(requested_class, payload, strict=True, **kwargs)
        assert result.normalized_data is not None
        return result.normalized_data

    @staticmethod
    def _format_errors(result: DataValidationResult) -> str:
        lines = [
            f"Create-data validation failed for requested class '{result.requested_class}'.",
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


def build_create_data_validator(
        ontology_blueprint_path: str | Path,
        *,
        strict_unknown_fields: bool = True,
        auto_add_metadata: bool = True,
        auto_add_object_id: bool = True,
        include_metadata_fields_in_validation: bool = True,
) -> CreateDataValidator:
    return CreateDataValidator(
        ontology_blueprint_path=ontology_blueprint_path,
        strict_unknown_fields=strict_unknown_fields,
        auto_add_metadata=auto_add_metadata,
        auto_add_object_id=auto_add_object_id,
        include_metadata_fields_in_validation=include_metadata_fields_in_validation,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate create payload data-property/scalar-field constraints.")
    parser.add_argument("class_name", type=str, help="Requested class name")
    parser.add_argument("payload_json", type=str, help="Path to a JSON payload file")
    parser.add_argument("--ontology_blueprint", type=str, default="ontology_blueprint.json")
    parser.add_argument("--subtype", type=str, default=None)
    parser.add_argument("--loose_unknown_fields", action="store_true")
    parser.add_argument("--skip_metadata_validation", action="store_true")
    args = parser.parse_args()

    payload = json.loads(Path(args.payload_json).read_text(encoding="utf-8"))
    validator = build_create_data_validator(
        args.ontology_blueprint,
        strict_unknown_fields=not args.loose_unknown_fields,
        include_metadata_fields_in_validation=not args.skip_metadata_validation,
    )
    result = validator.validate_data(args.class_name, payload, subtype=args.subtype, strict=False)
    print(json.dumps({
        "ok": result.ok,
        "requested_class": result.requested_class,
        "final_class": result.final_class,
        "collection_name": result.collection_name,
        "errors": [issue.__dict__ for issue in result.errors],
        "warnings": [issue.__dict__ for issue in result.warnings],
        "infos": [issue.__dict__ for issue in result.infos],
        "normalized_data": result.normalized_data,
    }, indent=2, default=str, ensure_ascii=False))
