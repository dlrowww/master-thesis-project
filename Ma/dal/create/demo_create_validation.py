from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

# --------------------------------------------------
# Import helpers
# --------------------------------------------------

THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent
PROJECT_ROOT = THIS_DIR

for candidate in [THIS_DIR, PROJECT_ROOT, PROJECT_ROOT / "dal" / "create"]:
    p = str(candidate)
    if p not in sys.path:
        sys.path.insert(0, p)


def _import_modules():
    """
    Try both package-style and standalone-style imports so the demo can be
    copied either into your project (recommended: dal/create/) or run directly
    from /mnt/data during quick testing.
    """
    try:
        from dal.create.create_class import build_create_constraint_engine  # type: ignore
        from dal.create.create_data import build_create_data_validator  # type: ignore
        from dal.create.create_object import build_create_object_validator  # type: ignore
        from dal.create.create_instance import build_create_instance_service  # type: ignore
        return (
            build_create_constraint_engine,
            build_create_data_validator,
            build_create_object_validator,
            build_create_instance_service,
        )
    except Exception:
        from create_class import build_create_constraint_engine  # type: ignore
        from create_data import build_create_data_validator  # type: ignore
        from create_object import build_create_object_validator  # type: ignore
        from create_instance import build_create_instance_service  # type: ignore
        return (
            build_create_constraint_engine,
            build_create_data_validator,
            build_create_object_validator,
            build_create_instance_service,
        )


(
    build_create_constraint_engine,
    build_create_data_validator,
    build_create_object_validator,
    build_create_instance_service,
) = _import_modules()


# --------------------------------------------------
# Small utilities
# --------------------------------------------------


def oid(n: int) -> str:
    """Return a valid 24-hex fake ObjectId string."""
    return f"{n:024x}"[-24:]


class DemoPrinter:
    @staticmethod
    def title(text: str) -> None:
        print("\n" + "=" * 88)
        print(text)
        print("=" * 88)

    @staticmethod
    def subtitle(text: str) -> None:
        print("\n" + "-" * 88)
        print(text)
        print("-" * 88)

    @staticmethod
    def show_result(name: str, result: Any) -> None:
        ok = getattr(result, "ok", False)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

        errors = getattr(result, "errors", []) or []
        warnings = getattr(result, "warnings", []) or []
        infos = getattr(result, "infos", []) or []

        if errors:
            print("  Errors:")
            for issue in errors:
                path = f" [{issue.path}]" if getattr(issue, "path", None) else ""
                print(f"    - {issue.code}{path}: {issue.message}")
        if warnings:
            print("  Warnings:")
            for issue in warnings:
                path = f" [{issue.path}]" if getattr(issue, "path", None) else ""
                print(f"    - {issue.code}{path}: {issue.message}")
        if infos:
            print("  Infos:")
            for issue in infos:
                path = f" [{issue.path}]" if getattr(issue, "path", None) else ""
                print(f"    - {issue.code}{path}: {issue.message}")

        final_class = getattr(result, "final_class", None)
        collection_name = getattr(result, "collection_name", None)
        if final_class or collection_name:
            print(f"  Final class: {final_class}")
            print(f"  Collection : {collection_name}")

        for attr_name in ["prepared_document", "normalized_data", "normalized_object_fields", "normalized_document"]:
            value = getattr(result, attr_name, None)
            if value is not None:
                print(f"  {attr_name}:")
                print(json.dumps(value, indent=2, default=str, ensure_ascii=False))
                break


# --------------------------------------------------
# Demo runner
# --------------------------------------------------


class CreateValidationDemo:
    def __init__(
            self,
            ontology_blueprint_path: str | Path,
            *,
            mongo_uri: Optional[str] = None,
            db_name: Optional[str] = None,
    ) -> None:
        self.ontology_blueprint_path = str(ontology_blueprint_path)
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.with_db = bool(mongo_uri and db_name)

        self.class_engine = build_create_constraint_engine(
            self.ontology_blueprint_path,
            mongo_uri=mongo_uri,
            db_name=db_name,
            strict_unknown_fields=True,
            auto_add_metadata=True,
            auto_add_object_id=True,
        )

        self.data_validator = build_create_data_validator(
            self.ontology_blueprint_path,
            strict_unknown_fields=True,
            auto_add_metadata=True,
            auto_add_object_id=True,
            include_metadata_fields_in_validation=True,
        )

        self.object_validator = build_create_object_validator(
            self.ontology_blueprint_path,
            mongo_uri=mongo_uri,
            db_name=db_name,
            strict_unknown_fields=True,
            require_reference_targets_to_exist=self.with_db,
            coerce_string_object_ids=True,
        )

        if self.with_db:
            self.instance_service = build_create_instance_service(
                self.ontology_blueprint_path,
                mongo_uri=mongo_uri,
                db_name=db_name,
                require_reference_targets_to_exist=True,
                check_collection_deployed=True,
            )
        else:
            self.instance_service = build_create_instance_service(
                self.ontology_blueprint_path,
                mongo_uri="mongodb://localhost:27017",
                db_name="owl_ac",
                require_reference_targets_to_exist=False,
                check_collection_deployed=False,
            )

    # ------------------------------
    # class-level examples
    # ------------------------------

    def run_class_cases(self) -> None:
        DemoPrinter.subtitle("CLASS-LEVEL CONSTRAINTS")

        cases = [
            (
                "Unknown class should fail",
                lambda: self.class_engine.validate_create("NotARealClass", {}),
            ),
            (
                "Abstract class Arrangement should fail",
                lambda: self.class_engine.validate_create("Arrangement", {}),
            ),
            (
                "Union-equivalent parent Activity without subtype should fail",
                lambda: self.class_engine.validate_create("Activity", {"name": "demo_activity"}),
            ),
            (
                "Concrete subtype GroupActivity should pass",
                lambda: self.class_engine.validate_create("Activity", {"name": "demo_group_activity"},
                                                          subtype="GroupActivity"),
            ),
            (
                "Vocabulary class Sex should fail by default for runtime create",
                lambda: self.class_engine.validate_create("Sex", {"name": "sexOther"}),
            ),
        ]

        for name, fn in cases:
            DemoPrinter.show_result(name, fn())

    # ------------------------------
    # data-level examples
    # ------------------------------

    def run_data_cases(self) -> None:
        DemoPrinter.subtitle("DATA-PROPERTY / SCALAR CONSTRAINTS")

        cases = [
            (
                "Participant without required name should fail",
                lambda: self.data_validator.validate_data("Participant", {}),
            ),
            (
                "Participant with unknown scalar field should fail",
                lambda: self.data_validator.validate_data(
                    "Participant",
                    {"name": "Alice", "unknown_scalar": 123},
                ),
            ),
            (
                "ParticipantState with age as string should fail",
                lambda: self.data_validator.validate_data(
                    "ParticipantState",
                    {"age": "twenty"},
                ),
            ),
            (
                "Participant with valid scalar fields should pass",
                lambda: self.data_validator.validate_data(
                    "Participant",
                    {
                        "name": "Alice",
                        "disorder": "none",
                    },
                ),
            ),
        ]

        for name, fn in cases:
            DemoPrinter.show_result(name, fn())

    # ------------------------------
    # object-level examples
    # ------------------------------

    def run_object_cases(self) -> None:
        DemoPrinter.subtitle("OBJECT-PROPERTY / REFERENCE CONSTRAINTS")

        cases = [
            (
                "Participation without required references should fail",
                lambda: self.object_validator.validate_object_fields("Participation", {}),
            ),
            (
                "Participation with array where single ref is expected should fail",
                lambda: self.object_validator.validate_object_fields(
                    "Participation",
                    {
                        "has_activity_execution_id": [oid(1)],
                        "has_participant_state_id": oid(2),
                    },
                ),
            ),
            (
                "GroupActivityExecution with too few refs for min cardinality should fail",
                lambda: self.object_validator.validate_object_fields(
                    "GroupActivityExecution",
                    {
                        "has_activity_id": oid(10),
                        "has_activity_execution_ids": [oid(11), oid(12)],
                    },
                ),
            ),
            (
                "Participation with well-formed refs should pass shape/cardinality checks",
                lambda: self.object_validator.validate_object_fields(
                    "Participation",
                    {
                        "has_activity_execution_id": oid(21),
                        "has_participant_state_id": oid(22),
                    },
                ),
            ),
        ]

        for name, fn in cases:
            DemoPrinter.show_result(name, fn())

        if not self.with_db:
            print(
                "\nNote: DB-backed target existence checks are currently skipped because no --mongo_uri/--db was provided.")

    # ------------------------------
    # full instance-level examples
    # ------------------------------

    def run_instance_cases(self) -> None:
        DemoPrinter.subtitle("FULL CREATE_INSTANCE VALIDATION (CLASS + DATA + OBJECT)")

        cases = [
            (
                "Create Participant valid instance should pass",
                lambda: self.instance_service.validate_instance(
                    "Participant",
                    {
                        "individual_id": "participant_001",
                        "name": "Alice",
                        "disorder": "none",
                    },
                ),
            ),
            (
                "Create Activity without subtype should fail at class layer",
                lambda: self.instance_service.validate_instance(
                    "Activity",
                    {"individual_id": "activity_001", "name": "ambiguous_activity"},
                ),
            ),
            (
                "Create Participant with bad scalar type should fail at data layer",
                lambda: self.instance_service.validate_instance(
                    "ParticipantState",
                    {
                        "individual_id": "participant_state_001",
                        "age": "bad_int",
                        "has_participant_id": oid(31),
                    },
                ),
            ),
            (
                "Create Participation with missing required object refs should fail at object layer",
                lambda: self.instance_service.validate_instance(
                    "Participation",
                    {
                        "individual_id": "participation_001",
                        "name": "p1",
                    },
                ),
            ),
            (
                "Create GroupActivityExecution with subtype, scalar, and refs should pass basic validation",
                lambda: self.instance_service.validate_instance(
                    "GroupActivityExecution",
                    {
                        "individual_id": "group_exec_001",
                        "name": "group_exec_demo",
                        "has_activity_id": oid(41),
                        "has_activity_execution_ids": [oid(42), oid(43), oid(44)],
                    },
                ),
            ),
        ]

        for name, fn in cases:
            DemoPrinter.show_result(name, fn())

    def run_all(self) -> None:
        DemoPrinter.title("DEMO: CREATE CONSTRAINT VALIDATION FROM CLASS → DATA → OBJECT → INSTANCE")
        print(f"Ontology blueprint: {self.ontology_blueprint_path}")
        if self.with_db:
            print(f"MongoDB target    : {self.mongo_uri} / {self.db_name}")
        else:
            print("MongoDB target    : not provided (reference-existence checks are skipped)")

        self.run_class_cases()
        self.run_data_cases()
        self.run_object_cases()
        self.run_instance_cases()


# --------------------------------------------------
# CLI
# --------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo for create-time ontology-aware validation from class to data to object."
    )
    parser.add_argument(
        "--ontology_blueprint",
        type=str,
        default=str(THIS_DIR / "ontology_blueprint.json") if (
                    THIS_DIR / "ontology_blueprint.json").exists() else "ontology_blueprint.json",
        help="Path to ontology_blueprint.json",
    )
    parser.add_argument("--mongo_uri", type=str, default=None, help="Optional Mongo URI for target-existence checks")
    parser.add_argument("--db", type=str, default=None, help="Optional DB name for target-existence checks")
    args = parser.parse_args()

    demo = CreateValidationDemo(
        args.ontology_blueprint,
        mongo_uri=args.mongo_uri,
        db_name=args.db,
    )
    demo.run_all()


if __name__ == "__main__":
    main()
