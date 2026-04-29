from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import MongoClient

from blueprint_registry import BlueprintRegistry, ReferenceFieldSpec, ScalarFieldSpec


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ValidationError:
    code: str
    message: str
    field_name: Optional[str] = None
    property_name: Optional[str] = None
    details: Dict[str, Any] = dc_field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        out = {"code": self.code, "message": self.message}
        if self.field_name is not None:
            out["field"] = self.field_name
        if self.property_name is not None:
            out["property_name"] = self.property_name
        if self.details:
            out["details"] = dict(self.details)
        return out


class UpdateValidator:
    SYSTEM_FIELDS = {"_id", "_ontology_type"}

    def __init__(self, registry: BlueprintRegistry, db: Any | None = None):
        self.registry = registry
        self.db = db

    def validate_update(
            self,
            class_name: str,
            old_doc: Dict[str, Any],
            patch: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        errors: List[ValidationError] = []

        if not isinstance(old_doc, dict):
            return [ValidationError(
                code="invalid_old_document",
                message="Existing document must be a dictionary-like document.",
            ).as_dict()]
        if not isinstance(patch, dict):
            return [ValidationError(
                code="invalid_patch",
                message="Update patch must be a dictionary-like document.",
            ).as_dict()]

        errors.extend(self._validate_class_request(class_name))
        if errors:
            return [e.as_dict() for e in errors]

        errors.extend(self._validate_update_patch_shape(class_name, old_doc, patch))
        merged_doc = self._merge_for_update(old_doc, patch)
        errors.extend(self._validate_document_state(class_name, merged_doc))
        return [e.as_dict() for e in errors]

    def _validate_document_state(self, class_name: str, document: Dict[str, Any]) -> List[ValidationError]:
        errors: List[ValidationError] = []

        if not isinstance(document, dict):
            return [ValidationError(code="invalid_payload", message="Payload must be a dictionary-like document.")]

        errors.extend(self._validate_class_request(class_name))
        collection_name = self.registry.get_collection_for_class(class_name)
        if not collection_name:
            errors.append(
                ValidationError(
                    code="no_collection_for_class",
                    message=f"Class '{class_name}' does not map to a MongoDB collection.",
                )
            )
            return errors

        allowed_fields = self.registry.get_allowed_fields_for_class(class_name)
        scalar_specs = self.registry.get_scalar_fields(collection_name)
        reference_specs = self.registry.get_reference_fields(collection_name)

        errors.extend(self._validate_unknown_fields(class_name, document, allowed_fields))
        errors.extend(self._validate_required_rules(class_name, document))

        for field_name, value in document.items():
            if field_name in self.SYSTEM_FIELDS:
                continue

            if field_name == "metadata":
                errors.extend(self._validate_metadata_field(field_name, value))
                continue

            if field_name in scalar_specs:
                errors.extend(self._validate_scalar_field(class_name, field_name, value, scalar_specs[field_name]))
            elif field_name in reference_specs:
                errors.extend(
                    self._validate_reference_field(
                        class_name=class_name,
                        collection_name=collection_name,
                        field_name=field_name,
                        value=value,
                        spec=reference_specs[field_name],
                    )
                )

        return errors

    def _validate_class_request(self, class_name: str) -> List[ValidationError]:
        errors: List[ValidationError] = []

        if not self.registry.class_exists(class_name):
            errors.append(
                ValidationError(
                    code="unknown_class",
                    message=f"Requested class '{class_name}' does not exist in the ontology blueprint.",
                )
            )
            return errors

        if self.registry.is_abstract_class(class_name):
            errors.append(
                ValidationError(
                    code="abstract_class",
                    message=f"Class '{class_name}' is abstract and cannot be instantiated directly.",
                )
            )

        if self.registry.is_union_parent(class_name):
            members = self.registry.get_union_members(class_name)
            errors.append(
                ValidationError(
                    code="concrete_subtype_required",
                    message=(
                        f"Class '{class_name}' is defined as a union-equivalent class and cannot be created directly. "
                        f"Choose one concrete subtype: {', '.join(members)}."
                    ),
                    details={"allowed_subtypes": members},
                )
            )

        return errors

    def _validate_update_patch_shape(
            self,
            class_name: str,
            old_doc: Dict[str, Any],
            patch: Dict[str, Any],
    ) -> List[ValidationError]:
        errors: List[ValidationError] = []

        if "_id" in patch and old_doc.get("_id") is not None and patch["_id"] != old_doc.get("_id"):
            errors.append(
                ValidationError(
                    code="immutable_field",
                    field_name="_id",
                    message="Field '_id' is immutable and cannot be changed during update.",
                    details={"old_value": str(old_doc.get("_id")), "new_value": str(patch.get("_id"))},
                )
            )

        if "_ontology_type" in patch:
            old_type = old_doc.get("_ontology_type")
            new_type = patch.get("_ontology_type")
            if old_type is not None and new_type != old_type:
                errors.append(
                    ValidationError(
                        code="immutable_field",
                        field_name="_ontology_type",
                        message="Field '_ontology_type' is immutable and cannot be changed during update.",
                        details={"old_value": old_type, "new_value": new_type},
                    )
                )
            elif old_type is None and new_type is not None and new_type != class_name:
                errors.append(
                    ValidationError(
                        code="invalid_ontology_type",
                        field_name="_ontology_type",
                        message=(
                            f"Patched '_ontology_type' value '{new_type}' "
                            f"does not match the validator class '{class_name}'."
                        ),
                        details={"expected_type": class_name, "patched_type": new_type},
                    )
                )

        if "_ontology_type" in old_doc:
            actual_old_type = old_doc.get("_ontology_type")
            if actual_old_type and actual_old_type != class_name:
                if (
                        not self.registry.is_type_compatible(actual_old_type, [class_name])
                        and not self.registry.is_type_compatible(class_name, [actual_old_type])
                ):
                    errors.append(
                        ValidationError(
                            code="class_document_type_mismatch",
                            field_name="_ontology_type",
                            message=(
                                f"Existing document has ontology type '{actual_old_type}', "
                                f"which is incompatible with validator class '{class_name}'."
                            ),
                            details={"document_type": actual_old_type, "validator_class": class_name},
                        )
                    )

        return errors

    def _validate_unknown_fields(
            self,
            class_name: str,
            payload: Dict[str, Any],
            allowed_fields: set[str],
    ) -> List[ValidationError]:
        errors: List[ValidationError] = []
        allowed_top_level = set(allowed_fields) | {"metadata"} | self.SYSTEM_FIELDS

        for field_name in payload.keys():
            if field_name not in allowed_top_level:
                origin = self.registry.get_field_origin_for_class(class_name, field_name)
                details: Dict[str, Any] = {"known_fields": sorted(allowed_fields)}
                if origin:
                    details["field_origin"] = origin
                errors.append(
                    ValidationError(
                        code="unknown_field",
                        field_name=field_name,
                        message=f"Field '{field_name}' is not allowed for class '{class_name}'.",
                        details=details,
                    )
                )

        return errors

    def _validate_metadata_field(self, field_name: str, value: Any) -> List[ValidationError]:
        if value is None or isinstance(value, dict):
            return []
        return [
            ValidationError(
                code="invalid_metadata",
                field_name=field_name,
                message="Field 'metadata' must be an object/document when provided.",
            )
        ]

    def _validate_required_rules(self, class_name: str, payload: Dict[str, Any]) -> List[ValidationError]:
        errors: List[ValidationError] = []
        required_rule_map = self.registry.get_required_rule_map_for_class(class_name)

        for _, rule in required_rule_map.items():
            mongo_field = rule.get("mongo_field")
            source_property = rule.get("source_property")
            if not mongo_field:
                continue

            present = mongo_field in payload and not self._is_effectively_empty(payload.get(mongo_field))
            if present:
                continue

            message_bits: List[str] = []
            if rule.get("required_by_validator"):
                message_bits.append("validator-required")
            if rule.get("required_by_ontology"):
                kind = rule.get("ontology_rule_kind") or "ontology-required"
                message_bits.append(f"ontology:{kind}")
            source_desc = ", ".join(message_bits) if message_bits else "required"

            errors.append(
                ValidationError(
                    code="missing_required_field",
                    field_name=mongo_field,
                    property_name=source_property,
                    message=(
                        f"Missing required field '{mongo_field}' for class '{class_name}' "
                        f"(source property '{source_property}', {source_desc})."
                    ),
                    details=rule,
                )
            )

        return errors

    def _validate_scalar_field(
            self,
            class_name: str,
            field_name: str,
            value: Any,
            spec: ScalarFieldSpec,
    ) -> List[ValidationError]:
        errors: List[ValidationError] = []
        if self._is_effectively_empty(value):
            return errors

        if spec.is_array:
            if not isinstance(value, list):
                errors.append(
                    ValidationError(
                        code="scalar_array_expected",
                        field_name=field_name,
                        property_name=spec.source_property,
                        message=f"Field '{field_name}' must be an array.",
                        details={"expected_bson_type": spec.bson_type},
                    )
                )
                return errors

            errors.extend(self._validate_cardinality_list(field_name, spec.min_items, spec.max_items, value))
            for idx, item in enumerate(value):
                if not self._matches_bson_type(item, spec.bson_type):
                    errors.append(
                        ValidationError(
                            code="scalar_type_mismatch",
                            field_name=field_name,
                            property_name=spec.source_property,
                            message=f"Field '{field_name}[{idx}]' does not match expected type '{spec.bson_type}'.",
                            details={
                                "expected_bson_type": spec.bson_type,
                                "actual_python_type": type(item).__name__,
                            },
                        )
                    )
            return errors

        if not self._matches_bson_type(value, spec.bson_type):
            errors.append(
                ValidationError(
                    code="scalar_type_mismatch",
                    field_name=field_name,
                    property_name=spec.source_property,
                    message=f"Field '{field_name}' does not match expected type '{spec.bson_type}'.",
                    details={"expected_bson_type": spec.bson_type, "actual_python_type": type(value).__name__},
                )
            )

        return errors

    def _validate_reference_field(
            self,
            class_name: str,
            collection_name: str,
            field_name: str,
            value: Any,
            spec: ReferenceFieldSpec,
    ) -> List[ValidationError]:
        errors: List[ValidationError] = []
        if self._is_effectively_empty(value):
            return errors

        raw_values: List[Any]
        if spec.is_array:
            if not isinstance(value, list):
                errors.append(
                    ValidationError(
                        code="reference_array_expected",
                        field_name=field_name,
                        property_name=spec.source_property,
                        message=f"Reference field '{field_name}' must be an array of ObjectIds.",
                    )
                )
                return errors
            raw_values = value
            errors.extend(self._validate_cardinality_list(field_name, spec.min_items, spec.max_items, raw_values))
        else:
            if isinstance(value, list):
                errors.append(
                    ValidationError(
                        code="single_reference_expected",
                        field_name=field_name,
                        property_name=spec.source_property,
                        message=f"Reference field '{field_name}' expects a single ObjectId, not a list.",
                    )
                )
                return errors
            raw_values = [value]

        object_ids: List[ObjectId] = []
        for idx, item in enumerate(raw_values):
            oid = self._coerce_object_id(item)
            if oid is None:
                label = field_name if not spec.is_array else f"{field_name}[{idx}]"
                errors.append(
                    ValidationError(
                        code="invalid_object_id",
                        field_name=field_name,
                        property_name=spec.source_property,
                        message=f"Reference value at '{label}' is not a valid Mongo ObjectId.",
                        details={"value": item},
                    )
                )
            else:
                object_ids.append(oid)

        if not object_ids or self.db is None:
            return errors

        target_collections = self.registry.get_target_collections(collection_name, field_name)
        if not target_collections:
            errors.append(
                ValidationError(
                    code="unresolved_target_collection",
                    field_name=field_name,
                    property_name=spec.source_property,
                    message=(
                        f"Reference field '{field_name}' does not resolve to any target collection "
                        f"for class '{class_name}'."
                    ),
                    details={"target_classes": list(spec.target_classes)},
                )
            )
            return errors

        for oid in object_ids:
            found_doc, found_collection = self._find_reference_target(oid, target_collections)
            if found_doc is None:
                errors.append(
                    ValidationError(
                        code="reference_target_missing",
                        field_name=field_name,
                        property_name=spec.source_property,
                        message=(
                            f"Referenced document '{oid}' for field '{field_name}' "
                            f"was not found in allowed target collections."
                        ),
                        details={"target_collections": list(target_collections)},
                    )
                )
                continue

            if spec.target_classes:
                actual_type = found_doc.get("_ontology_type")
                if actual_type and not self.registry.is_type_compatible(actual_type, spec.target_classes):
                    errors.append(
                        ValidationError(
                            code="reference_target_type_mismatch",
                            field_name=field_name,
                            property_name=spec.source_property,
                            message=(
                                f"Referenced document '{oid}' has ontology type '{actual_type}', "
                                f"which is incompatible with allowed target classes {list(spec.target_classes)}."
                            ),
                            details={
                                "target_collection": found_collection,
                                "actual_type": actual_type,
                                "allowed_types": list(spec.target_classes),
                            },
                        )
                    )

        return errors

    @staticmethod
    def _merge_for_update(old_doc: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
        base = deepcopy(old_doc)
        for key, value in patch.items():
            if key == "metadata" and isinstance(base.get("metadata"), dict) and isinstance(value, dict):
                merged_metadata = deepcopy(base["metadata"])
                merged_metadata.update(value)
                base["metadata"] = merged_metadata
            else:
                base[key] = value
        return base

    @staticmethod
    def _is_effectively_empty(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str) and value.strip() == "":
            return True
        if isinstance(value, (list, tuple, set, dict)) and len(value) == 0:
            return True
        return False

    @staticmethod
    def _coerce_object_id(value: Any) -> Optional[ObjectId]:
        if isinstance(value, ObjectId):
            return value
        if isinstance(value, str):
            try:
                return ObjectId(value)
            except (InvalidId, TypeError):
                return None
        return None

    @staticmethod
    def _matches_bson_type(value: Any, bson_type: str) -> bool:
        if bson_type == "string":
            return isinstance(value, str)
        if bson_type == "bool":
            return isinstance(value, bool)
        if bson_type in {"int", "long"}:
            return isinstance(value, int) and not isinstance(value, bool)
        if bson_type in {"double", "decimal"}:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if bson_type == "date":
            import datetime as _dt
            return isinstance(value, (_dt.datetime, _dt.date))
        if bson_type == "object":
            return isinstance(value, dict)
        if bson_type == "objectId":
            return isinstance(value, ObjectId)
        return True

    @staticmethod
    def _validate_cardinality_list(
            field_name: str,
            min_items: Optional[int],
            max_items: Optional[int],
            value: Sequence[Any],
    ) -> List[ValidationError]:
        errors: List[ValidationError] = []
        size = len(value)
        if min_items is not None and size < min_items:
            errors.append(
                ValidationError(
                    code="cardinality_too_small",
                    field_name=field_name,
                    message=f"Field '{field_name}' requires at least {min_items} value(s), got {size}.",
                    details={"min_items": min_items, "actual_items": size},
                )
            )
        if max_items is not None and size > max_items:
            errors.append(
                ValidationError(
                    code="cardinality_too_large",
                    field_name=field_name,
                    message=f"Field '{field_name}' allows at most {max_items} value(s), got {size}.",
                    details={"max_items": max_items, "actual_items": size},
                )
            )
        return errors

    def _find_reference_target(
            self,
            oid: ObjectId,
            target_collections: Sequence[str],
    ) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        if self.db is None:
            return None, None
        for coll_name in target_collections:
            doc = self.db[coll_name].find_one({"_id": oid})
            if doc is not None:
                return doc, coll_name
        return None, None

    @classmethod
    def from_files(
            cls,
            ontology_blueprint_path: str | Path,
            validators_path: str | Path,
            db: Any | None = None,
    ) -> "UpdateValidator":
        ontology_bp = json.loads(Path(ontology_blueprint_path).read_text(encoding="utf-8"))
        validators = json.loads(Path(validators_path).read_text(encoding="utf-8"))
        registry = BlueprintRegistry(ontology_bp, validators)
        return cls(registry=registry, db=db)


class UpdateDAL:
    def __init__(self, registry: BlueprintRegistry, db: Any):
        self.registry = registry
        self.db = db
        self.validator = UpdateValidator(registry=registry, db=db)

    def update_by_id(
            self,
            class_name_or_collection: str,
            document_id: str | ObjectId,
            patch: Dict[str, Any],
            *,
            return_document: bool = True,
    ) -> Dict[str, Any]:
        collection_name = self._resolve_collection_name(class_name_or_collection)
        if not collection_name:
            return {
                "ok": False,
                "errors": [
                    {
                        "code": "unknown_class_or_collection",
                        "message": (
                            f"'{class_name_or_collection}' is neither a known ontology class "
                            f"nor a known MongoDB collection."
                        ),
                    }
                ],
            }

        oid = self._coerce_object_id(document_id)
        if oid is None:
            return {
                "ok": False,
                "errors": [
                    {
                        "code": "invalid_object_id",
                        "message": f"Update target '{document_id}' is not a valid Mongo ObjectId.",
                        "field": "_id",
                    }
                ],
            }

        old_doc = self.db[collection_name].find_one({"_id": oid})
        if old_doc is None:
            return {
                "ok": False,
                "errors": [
                    {
                        "code": "document_not_found",
                        "message": f"No document with _id '{oid}' exists in collection '{collection_name}'.",
                        "field": "_id",
                        "details": {"collection": collection_name, "_id": str(oid)},
                    }
                ],
            }

        class_name = self._resolve_effective_class_name(class_name_or_collection, collection_name, old_doc)
        if not class_name:
            return {
                "ok": False,
                "errors": [
                    {
                        "code": "cannot_resolve_effective_class",
                        "message": (
                            f"Could not resolve an effective ontology class for document '{oid}' "
                            f"in collection '{collection_name}'."
                        ),
                    }
                ],
            }

        errors = self.validator.validate_update(class_name, old_doc, patch)
        if errors:
            return {"ok": False, "errors": errors}

        merged_doc = self.validator._merge_for_update(old_doc, patch)
        normalized = self._normalize_document_for_update(class_name, merged_doc, old_doc)

        replace_result = self.db[collection_name].replace_one({"_id": oid}, normalized)
        if replace_result.matched_count != 1:
            return {
                "ok": False,
                "errors": [
                    {
                        "code": "update_failed",
                        "message": f"Update operation did not match a document in '{collection_name}'.",
                        "details": {"collection": collection_name, "_id": str(oid)},
                    }
                ],
            }

        result: Dict[str, Any] = {
            "ok": True,
            "collection": collection_name,
            "class_name": class_name,
            "_id": str(oid),
            "matched_count": replace_result.matched_count,
            "modified_count": replace_result.modified_count,
        }

        if return_document:
            stored = self.db[collection_name].find_one({"_id": oid})
            result["document"] = stored

        return result

    def _normalize_document_for_update(
            self,
            class_name: str,
            merged_doc: Dict[str, Any],
            old_doc: Dict[str, Any],
    ) -> Dict[str, Any]:
        doc = deepcopy(merged_doc)

        doc["_id"] = old_doc["_id"]
        doc["_ontology_type"] = old_doc.get("_ontology_type") or class_name

        metadata = doc.get("metadata")
        if metadata is None or not isinstance(metadata, dict):
            metadata = {}
            doc["metadata"] = metadata

        old_metadata = old_doc.get("metadata")
        if isinstance(old_metadata, dict) and "createdAt" in old_metadata:
            metadata.setdefault("createdAt", old_metadata["createdAt"])

        metadata["updatedAt"] = utcnow()

        return doc

    def _resolve_effective_class_name(
            self,
            class_name_or_collection: str,
            collection_name: str,
            old_doc: Dict[str, Any],
    ) -> Optional[str]:
        if class_name_or_collection and self.registry.class_exists(class_name_or_collection):
            return class_name_or_collection

        old_type = old_doc.get("_ontology_type")
        if isinstance(old_type, str) and self.registry.class_exists(old_type):
            return old_type

        source_classes = sorted(self.registry.collection_to_classes.get(collection_name, set()))
        return source_classes[0] if source_classes else None

    def _resolve_collection_name(self, class_name_or_collection: str) -> Optional[str]:
        if not class_name_or_collection:
            return None
        if class_name_or_collection in self.registry.collection_specs:
            return class_name_or_collection
        return self.registry.get_collection_for_class(class_name_or_collection)

    @staticmethod
    def _coerce_object_id(value: Any) -> Optional[ObjectId]:
        if isinstance(value, ObjectId):
            return value
        if isinstance(value, str):
            try:
                return ObjectId(value)
            except (InvalidId, TypeError):
                return None
        return None

    @classmethod
    def from_files(
            cls,
            ontology_blueprint_path: str | Path,
            validators_path: str | Path,
            mongo_uri: str,
            db_name: str,
    ) -> "UpdateDAL":
        ontology_bp = json.loads(Path(ontology_blueprint_path).read_text(encoding="utf-8"))
        validators = json.loads(Path(validators_path).read_text(encoding="utf-8"))
        registry = BlueprintRegistry(ontology_bp, validators)
        client = MongoClient(mongo_uri)
        db = client[db_name]
        return cls(registry=registry, db=db)


def _load_json_arg(value: str) -> Any:
    stripped = value.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return json.loads(stripped)
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Update validation / DAL demo")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_validate = sub.add_parser("validate")
    p_validate.add_argument("ontology_blueprint", type=str)
    p_validate.add_argument("validators", type=str)
    p_validate.add_argument("class_name", type=str)
    p_validate.add_argument("old_doc_json", type=str)
    p_validate.add_argument("patch_json", type=str)

    p_execute = sub.add_parser("execute")
    p_execute.add_argument("ontology_blueprint", type=str)
    p_execute.add_argument("validators", type=str)
    p_execute.add_argument("mongo_uri", type=str)
    p_execute.add_argument("db_name", type=str)
    p_execute.add_argument("class_or_collection", type=str)
    p_execute.add_argument("document_id", type=str)
    p_execute.add_argument("patch_json", type=str)
    p_execute.add_argument("--no-return-document", action="store_true")

    args = parser.parse_args()

    if args.mode == "validate":
        validator = UpdateValidator.from_files(args.ontology_blueprint, args.validators)
        old_doc = _load_json_arg(args.old_doc_json)
        patch = _load_json_arg(args.patch_json)
        result = validator.validate_update(args.class_name, old_doc, patch)
    else:
        dal = UpdateDAL.from_files(args.ontology_blueprint, args.validators, args.mongo_uri, args.db_name)
        patch = _load_json_arg(args.patch_json)
        result = dal.update_by_id(
            args.class_or_collection,
            args.document_id,
            patch,
            return_document=not args.no_return_document,
        )

    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    _cli()
