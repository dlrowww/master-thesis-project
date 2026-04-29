from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import MongoClient

from blueprint_registry import BlueprintRegistry

try:
    from crud_update import UpdateValidator
except Exception:  # pragma: no cover - package-path fallback
    from scripts.crud_update import UpdateValidator

DELETE_MODES = {"restrict", "detach_if_valid", "force_detach"}
DETACH_POLICIES = {"restrict", "unset", "pull", "cascade"}


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


class DeleteValidator:
    def __init__(self, registry: BlueprintRegistry, db: Any | None = None):
        self.registry = registry
        self.db = db

    def validate_delete(self, class_name_or_collection: str, document_id: str | ObjectId) -> List[Dict[str, Any]]:
        errors: List[ValidationError] = []

        collection_name = self._resolve_collection_name(class_name_or_collection)
        if not collection_name:
            errors.append(
                ValidationError(
                    code="unknown_class_or_collection",
                    message=f"'{class_name_or_collection}' is neither a known ontology class nor a known MongoDB collection in the registry.",
                )
            )
            return [e.as_dict() for e in errors]

        oid = self._coerce_object_id(document_id)
        if oid is None:
            errors.append(
                ValidationError(
                    code="invalid_object_id",
                    field_name="_id",
                    message=f"Delete target '{document_id}' is not a valid Mongo ObjectId.",
                )
            )
            return [e.as_dict() for e in errors]

        if self.db is None:
            errors.append(
                ValidationError(
                    code="db_required_for_delete_validation",
                    message="Delete validation requires a live database connection to inspect references.",
                    details={"collection": collection_name, "_id": str(oid)},
                )
            )
            return [e.as_dict() for e in errors]

        doc = self.db[collection_name].find_one({"_id": oid}, {"_id": 1, "_ontology_type": 1, "name": 1, "iri": 1})
        if doc is None:
            errors.append(
                ValidationError(
                    code="document_not_found",
                    field_name="_id",
                    message=f"No document with _id '{oid}' exists in collection '{collection_name}'.",
                    details={"collection": collection_name, "_id": str(oid)},
                )
            )
            return [e.as_dict() for e in errors]

        incoming_refs = self.registry.get_incoming_references(collection_name)
        for ref in incoming_refs:
            source_collection = ref.get("source_collection")
            source_field = ref.get("source_field")
            source_property = ref.get("source_property")
            configured_policy = ref.get("on_delete") or "restrict"
            is_array = bool(ref.get("is_array"))

            if not source_collection or not source_field:
                continue

            query = {source_field: oid}
            count = self.db[source_collection].count_documents(query)
            if count <= 0:
                continue

            example_docs = list(
                self.db[source_collection].find(query, {"_id": 1, "_ontology_type": 1, "name": 1, "iri": 1}).limit(3)
            )
            examples = [self._summarize_doc_identity(d) for d in example_docs]

            errors.append(
                ValidationError(
                    code="incoming_reference_exists",
                    field_name=source_field,
                    property_name=source_property,
                    message=(
                        f"Cannot delete document '{oid}' from '{collection_name}' because it is still "
                        f"referenced by {count} document(s) through '{source_collection}.{source_field}'."
                    ),
                    details={
                        "target_collection": collection_name,
                        "target_id": str(oid),
                        "source_collection": source_collection,
                        "source_field": source_field,
                        "is_array": is_array,
                        "configured_policy": configured_policy,
                        "referencing_count": count,
                        "examples": examples,
                    },
                )
            )

        return [e.as_dict() for e in errors]

    def _resolve_collection_name(self, class_name_or_collection: str) -> Optional[str]:
        if not class_name_or_collection:
            return None
        if class_name_or_collection in self.registry.collection_specs:
            return class_name_or_collection
        return self.registry.get_collection_for_class(class_name_or_collection)

    @staticmethod
    def _summarize_doc_identity(doc: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "_id": str(doc.get("_id")),
            "_ontology_type": doc.get("_ontology_type"),
            "name": doc.get("name"),
            "iri": doc.get("iri"),
        }

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
    ) -> "DeleteValidator":
        ontology_bp = json.loads(Path(ontology_blueprint_path).read_text(encoding="utf-8"))
        validators = json.loads(Path(validators_path).read_text(encoding="utf-8"))
        registry = BlueprintRegistry(ontology_bp, validators)
        client = MongoClient(mongo_uri)
        db = client[db_name]
        return cls(registry=registry, db=db)


class DeleteDAL:
    def __init__(self, registry: BlueprintRegistry, db: Any):
        self.registry = registry
        self.db = db
        self.validator = DeleteValidator(registry=registry, db=db)
        self.update_validator = UpdateValidator(registry=registry, db=db)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preview_delete(
            self,
            class_name_or_collection: str,
            document_id: str | ObjectId,
            *,
            delete_mode: str = "restrict",
    ) -> Dict[str, Any]:
        delete_mode = self._normalize_delete_mode(delete_mode)
        if not delete_mode:
            return {
                "ok": False,
                "errors": [{
                    "code": "unknown_delete_mode",
                    "message": f"Delete mode must be one of {sorted(DELETE_MODES)}.",
                    "details": {"requested_mode": delete_mode},
                }],
            }

        collection_name = self._resolve_collection_name(class_name_or_collection)
        if not collection_name:
            return {
                "ok": False,
                "errors": [{
                    "code": "unknown_class_or_collection",
                    "message": f"'{class_name_or_collection}' is neither a known ontology class nor a known collection.",
                }],
            }

        oid = self._coerce_object_id(document_id)
        if oid is None:
            return {
                "ok": False,
                "errors": [{
                    "code": "invalid_object_id",
                    "message": f"Delete target '{document_id}' is not a valid Mongo ObjectId.",
                    "field": "_id",
                }],
            }

        target_doc = self.db[collection_name].find_one({"_id": oid})
        if target_doc is None:
            return {
                "ok": False,
                "errors": [{
                    "code": "document_not_found",
                    "message": f"No document with _id '{oid}' exists in collection '{collection_name}'.",
                    "field": "_id",
                    "details": {"collection": collection_name, "_id": str(oid)},
                }],
            }

        plan = self._build_delete_plan(collection_name=collection_name, oid=oid, delete_mode=delete_mode)
        return self._serialize_plan(plan)

    def delete_by_id(
            self,
            class_name_or_collection: str,
            document_id: str | ObjectId,
            *,
            delete_mode: str = "restrict",
            detach_incoming: Optional[bool] = None,
    ) -> Dict[str, Any]:
        # backward compatibility with old boolean API
        if detach_incoming is not None:
            delete_mode = "force_detach" if detach_incoming else "restrict"

        preview = self.preview_delete(class_name_or_collection, document_id, delete_mode=delete_mode)
        if not preview.get("ok"):
            preview["executed"] = False
            return preview

        collection_name = preview["collection"]
        oid = self._coerce_object_id(preview["_id"])
        assert oid is not None
        internal_plan = self._build_delete_plan(collection_name=collection_name, oid=oid, delete_mode=delete_mode)
        if internal_plan["errors"]:
            result = self._serialize_plan(internal_plan)
            result["executed"] = False
            return result

        return self._execute_delete_plan(internal_plan)

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _build_delete_plan(
            self,
            *,
            collection_name: str,
            oid: ObjectId,
            delete_mode: str,
    ) -> Dict[str, Any]:
        target_doc = self.db[collection_name].find_one({"_id": oid})
        plan: Dict[str, Any] = {
            "ok": True,
            "delete_mode": delete_mode,
            "target_collection": collection_name,
            "target_id": oid,
            "target_document": target_doc,
            "updates": [],
            "deletes": [{"collection": collection_name, "_id": oid}],
            "warnings": [],
            "errors": [],
            "affected_reference_groups": [],
        }
        if target_doc is None:
            plan["ok"] = False
            plan["errors"].append({
                "code": "document_not_found",
                "message": f"No document with _id '{oid}' exists in collection '{collection_name}'.",
                "field": "_id",
                "details": {"collection": collection_name, "_id": str(oid)},
            })
            return plan

        incoming_refs = self.registry.get_incoming_references(collection_name)
        for ref in incoming_refs:
            source_collection = ref.get("source_collection")
            source_field = ref.get("source_field")
            source_property = ref.get("source_property")
            is_array = bool(ref.get("is_array"))
            configured_policy = self._normalize_policy(ref.get("on_delete")) or ("pull" if is_array else "unset")

            if not source_collection or not source_field:
                continue

            source_docs = list(self.db[source_collection].find({source_field: oid}))
            if not source_docs:
                continue

            ref_group = {
                "source_collection": source_collection,
                "source_field": source_field,
                "source_property": source_property,
                "is_array": is_array,
                "configured_policy": configured_policy,
                "referencing_count": len(source_docs),
                "documents": [],
            }

            if delete_mode == "restrict":
                plan["ok"] = False
                examples = [self._summarize_doc_identity(d) for d in source_docs[:3]]
                plan["errors"].append({
                    "code": "incoming_reference_exists",
                    "message": (
                        f"Cannot delete document '{oid}' from '{collection_name}' because it is still referenced by "
                        f"{len(source_docs)} document(s) through '{source_collection}.{source_field}'."
                    ),
                    "field": source_field,
                    "property_name": source_property,
                    "details": {
                        "target_collection": collection_name,
                        "target_id": str(oid),
                        "source_collection": source_collection,
                        "source_field": source_field,
                        "configured_policy": configured_policy,
                        "referencing_count": len(source_docs),
                        "examples": examples,
                    },
                })
                ref_group["documents"] = examples
                plan["affected_reference_groups"].append(ref_group)
                continue

            for source_doc in source_docs:
                source_id = source_doc.get("_id")
                effective_policy, policy_warning = self._effective_policy_for_mode(
                    delete_mode=delete_mode,
                    configured_policy=configured_policy,
                    is_array=is_array,
                )
                doc_entry = {
                    "source_document": self._summarize_doc_identity(source_doc),
                    "configured_policy": configured_policy,
                    "effective_policy": effective_policy,
                    "validation_errors": [],
                    "ok": True,
                }
                if policy_warning:
                    warning_payload = {
                        "code": "force_detach_policy_override",
                        "message": policy_warning,
                        "details": {
                            "source_collection": source_collection,
                            "source_field": source_field,
                            "source_document_id": str(source_id),
                            "configured_policy": configured_policy,
                            "effective_policy": effective_policy,
                        },
                    }
                    plan["warnings"].append(warning_payload)
                    doc_entry.setdefault("warnings", []).append(warning_payload)

                if effective_policy == "cascade":
                    plan["ok"] = False
                    doc_entry["ok"] = False
                    doc_entry["validation_errors"].append({
                        "code": "cascade_not_supported",
                        "message": (
                            f"Cascade delete policy is not implemented for '{source_collection}.{source_field}'."
                        ),
                    })
                    plan["errors"].append({
                        "code": "cascade_not_supported",
                        "message": (
                            f"Cascade delete policy is not implemented for '{source_collection}.{source_field}'."
                        ),
                        "details": {
                            "source_collection": source_collection,
                            "source_field": source_field,
                            "source_document_id": str(source_id),
                        },
                    })
                    ref_group["documents"].append(doc_entry)
                    continue

                if effective_policy == "restrict":
                    plan["ok"] = False
                    doc_entry["ok"] = False
                    doc_entry["validation_errors"].append({
                        "code": "incoming_reference_restricts_delete",
                        "message": (
                            f"Delete is blocked by policy 'restrict' on '{source_collection}.{source_field}'."
                        ),
                    })
                    plan["errors"].append({
                        "code": "incoming_reference_restricts_delete",
                        "message": (
                            f"Delete is blocked by policy 'restrict' on '{source_collection}.{source_field}'."
                        ),
                        "field": source_field,
                        "property_name": source_property,
                        "details": {
                            "source_collection": source_collection,
                            "source_document_id": str(source_id),
                            "configured_policy": configured_policy,
                        },
                    })
                    ref_group["documents"].append(doc_entry)
                    continue

                simulated_doc = self._simulate_reference_removal(
                    source_doc=source_doc,
                    source_field=source_field,
                    target_oid=oid,
                    effective_policy=effective_policy,
                )

                validation_errors: List[Dict[str, Any]] = []
                if delete_mode == "detach_if_valid":
                    validation_errors = self._validate_simulated_document(source_collection, simulated_doc)
                    if validation_errors:
                        plan["ok"] = False
                        doc_entry["ok"] = False
                        doc_entry["validation_errors"] = validation_errors
                        plan["errors"].append({
                            "code": "detach_would_violate_constraints",
                            "message": (
                                f"Detaching '{source_collection}.{source_field}' from document '{source_id}' would violate "
                                f"document constraints."
                            ),
                            "field": source_field,
                            "property_name": source_property,
                            "details": {
                                "source_collection": source_collection,
                                "source_document_id": str(source_id),
                                "configured_policy": configured_policy,
                                "effective_policy": effective_policy,
                                "validation_errors": validation_errors,
                            },
                        })
                else:
                    doc_entry.setdefault("warnings", []).append({
                        "code": "force_detach_skips_validation",
                        "message": "Force-detach mode skips post-detach constraint validation for affected source documents.",
                    })

                plan["updates"].append({
                    "source_collection": source_collection,
                    "source_field": source_field,
                    "source_property": source_property,
                    "source_id": source_id,
                    "is_array": is_array,
                    "configured_policy": configured_policy,
                    "effective_policy": effective_policy,
                    "before_document": source_doc,
                    "simulated_document": simulated_doc,
                })
                ref_group["documents"].append(doc_entry)

            plan["affected_reference_groups"].append(ref_group)

        return plan

    def _simulate_reference_removal(
            self,
            *,
            source_doc: Dict[str, Any],
            source_field: str,
            target_oid: ObjectId,
            effective_policy: str,
    ) -> Dict[str, Any]:
        doc = deepcopy(source_doc)
        if effective_policy == "unset":
            doc.pop(source_field, None)
        elif effective_policy == "pull":
            current = doc.get(source_field)
            if isinstance(current, list):
                doc[source_field] = [v for v in current if self._coerce_object_id(v) != target_oid]
            else:
                doc.pop(source_field, None)
        else:
            raise ValueError(f"Unsupported effective delete policy '{effective_policy}'.")

        metadata = doc.get("metadata")
        if isinstance(metadata, dict):
            metadata = deepcopy(metadata)
            metadata["updatedAt"] = utcnow()
            doc["metadata"] = metadata
        return doc

    def _validate_simulated_document(self, source_collection: str, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        class_name = document.get("_ontology_type") or self.registry.get_primary_class_for_collection(source_collection)
        if not class_name:
            return [{
                "code": "unknown_source_document_type",
                "message": f"Unable to resolve ontology type for affected document in '{source_collection}'.",
            }]
        errors = self.update_validator._validate_document_state(class_name, document)
        return [e.as_dict() for e in errors]

    def _effective_policy_for_mode(
            self,
            *,
            delete_mode: str,
            configured_policy: str,
            is_array: bool,
    ) -> Tuple[str, Optional[str]]:
        configured_policy = self._normalize_policy(configured_policy) or ("pull" if is_array else "unset")
        if delete_mode == "force_detach" and configured_policy == "restrict":
            effective_policy = "pull" if is_array else "unset"
            return effective_policy, (
                f"Delete mode 'force_detach' overrides configured policy 'restrict' with '{effective_policy}'."
            )
        return configured_policy, None

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute_delete_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        detached_summary: List[Dict[str, Any]] = []

        def _apply(session: Any | None) -> None:
            detached_summary.clear()
            for update in plan["updates"]:
                simulated_doc = deepcopy(update["simulated_document"])
                result = self.db[update["source_collection"]].replace_one(
                    {"_id": update["source_id"]},
                    simulated_doc,
                    session=session,
                )
                if result.matched_count != 1:
                    raise RuntimeError(
                        f"Failed to update affected document '{update['source_id']}' in '{update['source_collection']}'."
                    )
                detached_summary.append({
                    "source_collection": update["source_collection"],
                    "source_field": update["source_field"],
                    "source_document_id": str(update["source_id"]),
                    "configured_policy": update["configured_policy"],
                    "effective_policy": update["effective_policy"],
                    "matched_count": result.matched_count,
                    "modified_count": result.modified_count,
                })

            delete_result = self.db[plan["target_collection"]].delete_one({"_id": plan["target_id"]}, session=session)
            if delete_result.deleted_count != 1:
                raise RuntimeError(
                    f"Delete operation did not remove document '{plan['target_id']}' from '{plan['target_collection']}'."
                )

        transaction_meta = self._run_with_best_effort_transaction(_apply)

        return {
            "ok": True,
            "executed": True,
            "delete_mode": plan["delete_mode"],
            "collection": plan["target_collection"],
            "_id": str(plan["target_id"]),
            "deleted_count": 1,
            "detached_summary": detached_summary,
            "warnings": list(plan["warnings"]),
            "transaction": transaction_meta,
        }

    def _run_with_best_effort_transaction(self, fn) -> Dict[str, Any]:
        client = getattr(self.db, "client", None)
        if client is None:
            fn(None)
            return {
                "attempted": False,
                "used": False,
                "fallback_used": True,
                "reason": "Database handle does not expose a Mongo client for sessions.",
            }

        try:
            with client.start_session() as session:
                with session.start_transaction():
                    fn(session)
                return {
                    "attempted": True,
                    "used": True,
                    "fallback_used": False,
                    "reason": None,
                }
        except Exception as exc:
            fn(None)
            return {
                "attempted": True,
                "used": False,
                "fallback_used": True,
                "reason": f"Transaction unavailable or failed to start cleanly: {exc}",
            }

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def _serialize_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ok": not plan["errors"],
            "preview_only": True,
            "delete_mode": plan["delete_mode"],
            "collection": plan["target_collection"],
            "_id": str(plan["target_id"]),
            "target_document": self._summarize_doc_identity(plan["target_document"]),
            "plan": {
                "affected_reference_groups": deepcopy(plan["affected_reference_groups"]),
                "update_count": len(plan["updates"]),
                "warning_count": len(plan["warnings"]),
            },
            "warnings": list(plan["warnings"]),
            "errors": list(plan["errors"]),
        }

    @staticmethod
    def _summarize_doc_identity(doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if doc is None:
            return None
        return {
            "_id": str(doc.get("_id")),
            "_ontology_type": doc.get("_ontology_type"),
            "name": doc.get("name"),
            "iri": doc.get("iri"),
        }

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

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

    @staticmethod
    def _normalize_delete_mode(delete_mode: str) -> Optional[str]:
        mode = str(delete_mode or "").strip().lower()
        return mode if mode in DELETE_MODES else None

    @staticmethod
    def _normalize_policy(policy: Any) -> Optional[str]:
        value = str(policy or "").strip().lower()
        return value if value in DETACH_POLICIES else None

    @classmethod
    def from_files(
            cls,
            ontology_blueprint_path: str | Path,
            validators_path: str | Path,
            mongo_uri: str,
            db_name: str,
    ) -> "DeleteDAL":
        ontology_bp = json.loads(Path(ontology_blueprint_path).read_text(encoding="utf-8"))
        validators = json.loads(Path(validators_path).read_text(encoding="utf-8"))
        registry = BlueprintRegistry(ontology_bp, validators)
        client = MongoClient(mongo_uri)
        db = client[db_name]
        return cls(registry=registry, db=db)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Delete validation / DAL demo")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_validate = sub.add_parser("validate")
    p_validate.add_argument("ontology_blueprint", type=str)
    p_validate.add_argument("validators", type=str)
    p_validate.add_argument("mongo_uri", type=str)
    p_validate.add_argument("db_name", type=str)
    p_validate.add_argument("class_or_collection", type=str)
    p_validate.add_argument("document_id", type=str)

    p_preview = sub.add_parser("preview")
    p_preview.add_argument("ontology_blueprint", type=str)
    p_preview.add_argument("validators", type=str)
    p_preview.add_argument("mongo_uri", type=str)
    p_preview.add_argument("db_name", type=str)
    p_preview.add_argument("class_or_collection", type=str)
    p_preview.add_argument("document_id", type=str)
    p_preview.add_argument("--delete-mode", default="restrict", choices=sorted(DELETE_MODES))

    p_execute = sub.add_parser("execute")
    p_execute.add_argument("ontology_blueprint", type=str)
    p_execute.add_argument("validators", type=str)
    p_execute.add_argument("mongo_uri", type=str)
    p_execute.add_argument("db_name", type=str)
    p_execute.add_argument("class_or_collection", type=str)
    p_execute.add_argument("document_id", type=str)
    p_execute.add_argument("--delete-mode", default="restrict", choices=sorted(DELETE_MODES))
    p_execute.add_argument("--detach-incoming", action="store_true")

    args = parser.parse_args()

    if args.mode == "validate":
        validator = DeleteValidator.from_files(args.ontology_blueprint, args.validators, args.mongo_uri, args.db_name)
        result = validator.validate_delete(args.class_or_collection, args.document_id)
    else:
        dal = DeleteDAL.from_files(args.ontology_blueprint, args.validators, args.mongo_uri, args.db_name)
        if args.mode == "preview":
            result = dal.preview_delete(args.class_or_collection, args.document_id, delete_mode=args.delete_mode)
        else:
            result = dal.delete_by_id(
                args.class_or_collection,
                args.document_id,
                delete_mode=args.delete_mode,
                detach_incoming=args.detach_incoming if getattr(args, "detach_incoming", False) else None,
            )

    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    _cli()
