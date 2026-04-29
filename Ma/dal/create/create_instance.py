from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from bson import ObjectId  # type: ignore
except Exception:  # pragma: no cover
    ObjectId = str  # type: ignore

try:
    from pymongo import MongoClient  # type: ignore
    from pymongo.database import Database  # type: ignore
except Exception:  # pragma: no cover
    MongoClient = None  # type: ignore
    Database = Any  # type: ignore

# Support both package-style imports and standalone file usage.
try:  # pragma: no cover
    from .create_class import CreateConstraintEngine, ValidationIssue
    from .create_data import CreateDataValidator, DataValidationIssue
    from .create_object import CreateObjectValidator, ObjectValidationIssue
except Exception:  # pragma: no cover
    from create_class import CreateConstraintEngine, ValidationIssue
    from create_data import CreateDataValidator, DataValidationIssue
    from create_object import CreateObjectValidator, ObjectValidationIssue


@dataclass
class InstanceValidationIssue:
    level: str  # error | warning | info
    code: str
    message: str
    path: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CreateInstanceResult:
    ok: bool
    requested_class: str
    final_class: Optional[str]
    collection_name: Optional[str]
    normalized_document: Optional[Dict[str, Any]]
    inserted_id: Optional[Any] = None
    errors: List[InstanceValidationIssue] = field(default_factory=list)
    warnings: List[InstanceValidationIssue] = field(default_factory=list)
    infos: List[InstanceValidationIssue] = field(default_factory=list)


class CreateInstanceError(ValueError):
    pass


class CreateInstanceService:
    """
    Orchestrates create-instance validation and insertion.

    Responsibilities:
    - class-level resolution and policy checks
    - scalar/data-property validation
    - object-property/reference validation
    - collection deployment check
    - final insert into MongoDB

    Non-goals for v1:
    - inverse back-reference updates
    - full value-restriction (only) enforcement
    - cross-document inverse cardinality maintenance
    - transactional cascade logic
    """

    def __init__(
            self,
            ontology_blueprint_path: str | Path,
            *,
            mongo_uri: str,
            db_name: str,
            allow_create_in_vocabulary_collections: bool = False,
            strict_unknown_fields: bool = True,
            auto_add_metadata: bool = True,
            auto_add_object_id: bool = True,
            require_reference_targets_to_exist: bool = True,
            check_collection_deployed: bool = True,
    ) -> None:
        self.ontology_blueprint_path = str(ontology_blueprint_path)
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.check_collection_deployed = check_collection_deployed

        self.class_validator = CreateConstraintEngine(
            self.ontology_blueprint_path,
            mongo_uri=mongo_uri,
            db_name=db_name,
            allow_create_in_vocabulary_collections=allow_create_in_vocabulary_collections,
            strict_unknown_fields=strict_unknown_fields,
            auto_add_metadata=auto_add_metadata,
            auto_add_object_id=auto_add_object_id,
        )
        self.data_validator = CreateDataValidator(
            self.ontology_blueprint_path,
            strict_unknown_fields=strict_unknown_fields,
            auto_add_metadata=auto_add_metadata,
            auto_add_object_id=auto_add_object_id,
            include_metadata_fields_in_validation=True,
        )
        self.object_validator = CreateObjectValidator(
            self.ontology_blueprint_path,
            mongo_uri=mongo_uri,
            db_name=db_name,
            strict_unknown_fields=strict_unknown_fields,
            require_reference_targets_to_exist=require_reference_targets_to_exist,
            coerce_string_object_ids=True,
        )

    # --------------------------------------------------
    # DB helpers
    # --------------------------------------------------

    def _open_db(self) -> Tuple[Any, Any]:
        if MongoClient is None:
            raise CreateInstanceError(
                "pymongo is not installed in the current environment, so create_instance cannot open MongoDB."
            )
        client = MongoClient(self.mongo_uri)
        return client, client[self.db_name]

    @staticmethod
    def _merge_documents(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(left)
        for key, value in right.items():
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key] = CreateInstanceService._merge_documents(out[key], value)
            else:
                out[key] = value
        return out

    @staticmethod
    def _convert_issue(issue: Any) -> InstanceValidationIssue:
        return InstanceValidationIssue(
            level=getattr(issue, "level", "error"),
            code=getattr(issue, "code", "unknown"),
            message=getattr(issue, "message", str(issue)),
            path=getattr(issue, "path", None),
            details=dict(getattr(issue, "details", {}) or {}),
        )

    def _ensure_collection_exists(
            self,
            collection_name: str,
            db: Database,
            errors: List[InstanceValidationIssue],
    ) -> None:
        if not self.check_collection_deployed:
            return
        if collection_name not in db.list_collection_names():
            errors.append(
                InstanceValidationIssue(
                    level="error",
                    code="collection_not_deployed",
                    message=(
                        f"Collection '{collection_name}' does not exist in database '{self.db_name}'. "
                        f"Create/apply the data layer first."
                    ),
                    path=None,
                )
            )

    def _optionally_check_individual_identity(
            self,
            collection_name: str,
            document: Dict[str, Any],
            db: Database,
            warnings: List[InstanceValidationIssue],
    ) -> None:
        """
        Optional helper only. No hard enforcement by default.

        If the payload contains `individual_id`, we warn when the same semantic id
        already exists in a different collection. This is useful for your ontology-level
        concern around one semantic instance accidentally being created in multiple places.
        """
        individual_id = document.get("individual_id")
        if not individual_id:
            return

        for other_collection in db.list_collection_names():
            if other_collection == collection_name:
                continue
            hit = db[other_collection].find_one({"individual_id": individual_id}, {"_id": 1})
            if hit is not None:
                warnings.append(
                    InstanceValidationIssue(
                        level="warning",
                        code="individual_id_seen_in_other_collection",
                        message=(
                            f"individual_id '{individual_id}' already exists in collection "
                            f"'{other_collection}'. Make sure this is semantically intended."
                        ),
                        path="individual_id",
                        details={"other_collection": other_collection, "other_id": str(hit.get("_id"))},
                    )
                )

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def validate_instance(
            self,
            requested_class: str,
            payload: Dict[str, Any],
            *,
            subtype: Optional[str] = None,
            created_by: str = "create_instance",
            db: Optional[Database] = None,
    ) -> CreateInstanceResult:
        errors: List[InstanceValidationIssue] = []
        warnings: List[InstanceValidationIssue] = []
        infos: List[InstanceValidationIssue] = []

        # 1) class-level policy / routing checks
        class_result = self.class_validator.validate_create(
            requested_class,
            payload,
            subtype=subtype,
            db=db,
            created_by=created_by,
            strict=False,
            check_db_collections=False,
            check_reference_targets=False,
        )
        errors.extend(self._convert_issue(x) for x in class_result.errors)
        warnings.extend(self._convert_issue(x) for x in class_result.warnings)
        infos.extend(self._convert_issue(x) for x in class_result.infos)

        if not class_result.final_class or not class_result.collection_name:
            return CreateInstanceResult(
                ok=False,
                requested_class=requested_class,
                final_class=class_result.final_class,
                collection_name=class_result.collection_name,
                normalized_document=None,
                errors=errors,
                warnings=warnings,
                infos=infos,
            )

        # 2) scalar/data-property validation
        data_result = self.data_validator.validate_data(
            requested_class,
            payload,
            subtype=subtype,
            created_by=created_by,
            strict=False,
        )
        errors.extend(self._convert_issue(x) for x in data_result.errors)
        warnings.extend(self._convert_issue(x) for x in data_result.warnings)
        infos.extend(self._convert_issue(x) for x in data_result.infos)

        # 3) object/reference validation
        object_result = self.object_validator.validate_object_fields(
            requested_class,
            payload,
            subtype=subtype,
        )
        errors.extend(self._convert_issue(x) for x in object_result.errors)
        warnings.extend(self._convert_issue(x) for x in object_result.warnings)
        infos.extend(self._convert_issue(x) for x in object_result.infos)

        # 4) consistency checks between the three validators
        final_class = data_result.final_class or object_result.final_class or class_result.final_class
        collection_name = (
                data_result.collection_name or object_result.collection_name or class_result.collection_name
        )

        if class_result.final_class and final_class and class_result.final_class != final_class:
            errors.append(
                InstanceValidationIssue(
                    level="error",
                    code="final_class_mismatch",
                    message=(
                        f"Validators resolved inconsistent final classes: class={class_result.final_class}, "
                        f"data={data_result.final_class}, object={object_result.final_class}."
                    ),
                )
            )

        if class_result.collection_name and collection_name and class_result.collection_name != collection_name:
            errors.append(
                InstanceValidationIssue(
                    level="error",
                    code="collection_name_mismatch",
                    message=(
                        f"Validators resolved inconsistent collection names: class={class_result.collection_name}, "
                        f"data={data_result.collection_name}, object={object_result.collection_name}."
                    ),
                )
            )

        if not data_result.normalized_data:
            final_doc = None
        else:
            final_doc = dict(data_result.normalized_data)
            if object_result.normalized_object_fields:
                final_doc = self._merge_documents(final_doc, object_result.normalized_object_fields)

        # 5) optional database checks that require collection access
        client = None
        active_db = db
        if active_db is None:
            client, active_db = self._open_db()
        try:
            if active_db is not None and collection_name:
                self._ensure_collection_exists(collection_name, active_db, errors)
                if final_doc is not None:
                    self._optionally_check_individual_identity(collection_name, final_doc, active_db, warnings)
        finally:
            if client is not None:
                client.close()

        return CreateInstanceResult(
            ok=len(errors) == 0,
            requested_class=requested_class,
            final_class=final_class,
            collection_name=collection_name,
            normalized_document=final_doc if len(errors) == 0 else None,
            errors=errors,
            warnings=warnings,
            infos=infos,
        )

    def assert_instance_valid(
            self,
            requested_class: str,
            payload: Dict[str, Any],
            *,
            subtype: Optional[str] = None,
            created_by: str = "create_instance",
            db: Optional[Database] = None,
    ) -> Dict[str, Any]:
        result = self.validate_instance(
            requested_class,
            payload,
            subtype=subtype,
            created_by=created_by,
            db=db,
        )
        if not result.ok or result.normalized_document is None:
            raise CreateInstanceError(self._format_result(result))
        return result.normalized_document

    def create_instance(
            self,
            requested_class: str,
            payload: Dict[str, Any],
            *,
            subtype: Optional[str] = None,
            created_by: str = "create_instance",
            db: Optional[Database] = None,
    ) -> Dict[str, Any]:
        client = None
        active_db = db
        if active_db is None:
            client, active_db = self._open_db()

        try:
            validated = self.validate_instance(
                requested_class,
                payload,
                subtype=subtype,
                created_by=created_by,
                db=active_db,
            )
            if not validated.ok or validated.normalized_document is None or not validated.collection_name:
                raise CreateInstanceError(self._format_result(validated))

            insert_result = active_db[validated.collection_name].insert_one(validated.normalized_document)
            inserted_doc = dict(validated.normalized_document)
            inserted_doc["_id"] = inserted_doc.get("_id", insert_result.inserted_id)
            return inserted_doc
        finally:
            if client is not None:
                client.close()

    @staticmethod
    def _format_result(result: CreateInstanceResult) -> str:
        lines = [
            f"Create-instance validation failed for requested class '{result.requested_class}'.",
            f"Final class: {result.final_class}",
            f"Collection: {result.collection_name}",
        ]
        if result.errors:
            lines.append("Errors:")
            for err in result.errors:
                path = f" [{err.path}]" if err.path else ""
                lines.append(f"- {err.code}{path}: {err.message}")
        if result.warnings:
            lines.append("Warnings:")
            for warn in result.warnings:
                path = f" [{warn.path}]" if warn.path else ""
                lines.append(f"- {warn.code}{path}: {warn.message}")
        if result.infos:
            lines.append("Infos:")
            for info in result.infos:
                path = f" [{info.path}]" if info.path else ""
                lines.append(f"- {info.code}{path}: {info.message}")
        return "\n".join(lines)


def build_create_instance_service(
        ontology_blueprint_path: str | Path,
        *,
        mongo_uri: str,
        db_name: str,
        allow_create_in_vocabulary_collections: bool = False,
        strict_unknown_fields: bool = True,
        auto_add_metadata: bool = True,
        auto_add_object_id: bool = True,
        require_reference_targets_to_exist: bool = True,
        check_collection_deployed: bool = True,
) -> CreateInstanceService:
    return CreateInstanceService(
        ontology_blueprint_path=ontology_blueprint_path,
        mongo_uri=mongo_uri,
        db_name=db_name,
        allow_create_in_vocabulary_collections=allow_create_in_vocabulary_collections,
        strict_unknown_fields=strict_unknown_fields,
        auto_add_metadata=auto_add_metadata,
        auto_add_object_id=auto_add_object_id,
        require_reference_targets_to_exist=require_reference_targets_to_exist,
        check_collection_deployed=check_collection_deployed,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate and create a runtime instance document.")
    parser.add_argument("class_name", type=str, help="Requested class name")
    parser.add_argument("payload_json", type=str, help="Path to a JSON payload file")
    parser.add_argument("--ontology_blueprint", type=str, default="ontology_blueprint.json")
    parser.add_argument("--mongo_uri", type=str, default="mongodb://localhost:27017")
    parser.add_argument("--db", type=str, default="owl_ac")
    parser.add_argument("--subtype", type=str, default=None)
    parser.add_argument("--validate_only", action="store_true")
    args = parser.parse_args()

    payload = json.loads(Path(args.payload_json).read_text(encoding="utf-8"))
    service = build_create_instance_service(
        args.ontology_blueprint,
        mongo_uri=args.mongo_uri,
        db_name=args.db,
    )

    if args.validate_only:
        result = service.validate_instance(args.class_name, payload, subtype=args.subtype)
        print(json.dumps({
            "ok": result.ok,
            "requested_class": result.requested_class,
            "final_class": result.final_class,
            "collection_name": result.collection_name,
            "errors": [x.__dict__ for x in result.errors],
            "warnings": [x.__dict__ for x in result.warnings],
            "infos": [x.__dict__ for x in result.infos],
            "normalized_document": result.normalized_document,
        }, indent=2, default=str, ensure_ascii=False))
    else:
        doc = service.create_instance(args.class_name, payload, subtype=args.subtype)
        print(json.dumps(doc, indent=2, default=str, ensure_ascii=False))
