from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional


# ======================================================================================
# Import helpers
# ======================================================================================


def import_project_symbols() -> Dict[str, Any]:
    """
    Import the project symbols needed by the unified demo.

    Supported input modes:
      - mixed: one OWL file containing schema + data
      - separated: schema OWL + instance OWL

    Step 4 is interactive CRUD over the generated blueprint/validators/MongoDB data layer.
    """
    symbols: Dict[str, Any] = {}

    # Step 1A: schema extraction / mixed extraction
    try:
        from mapping.mapping_rules import MappingConfig, MappingOverrides
        from mapping.ontology_blueprint import (
            build_ontology_blueprint,
            ontology_blueprint_to_json,
        )
        from ontology.owl_parser import parse_owl

        symbols.update(
            {
                "MappingConfig": MappingConfig,
                "MappingOverrides": MappingOverrides,
                "build_ontology_blueprint": build_ontology_blueprint,
                "ontology_blueprint_to_json": ontology_blueprint_to_json,
                "parse_owl": parse_owl,
            }
        )
    except Exception as exc:
        raise ImportError(
            "Failed to import blueprint extraction modules using package paths. "
            "Expected modules like mapping.mapping_rules, mapping.ontology_blueprint, ontology.owl_parser."
        ) from exc

    # Step 1B: schema/data separated enrichment
    augmenter_import_errors = []
    for mod_name in [
        "scripts.augment_ontology_blueprint_with_instances",
        "augment_ontology_blueprint_with_instances",
    ]:
        try:
            module = __import__(mod_name, fromlist=["enrich_blueprint"])
            symbols["enrich_blueprint"] = getattr(module, "enrich_blueprint")
            break
        except Exception as exc:
            augmenter_import_errors.append((mod_name, exc))
    else:
        raise ImportError(
            "Failed to import enrich_blueprint from augment_ontology_blueprint_with_instances. "
            f"Tried: {[name for name, _ in augmenter_import_errors]}"
        )

    # Step 2: ontology blueprint -> mongo validators
    validator_import_errors = []
    for mod_name in [
        "scripts.generate_mongo_validators_from_ontology_blueprint",
        "generate_mongo_validators_from_ontology_blueprint",
    ]:
        try:
            module = __import__(mod_name, fromlist=["build_mongodb_validators_from_ontology_blueprint"])
            symbols["build_mongodb_validators_from_ontology_blueprint"] = getattr(
                module, "build_mongodb_validators_from_ontology_blueprint"
            )
            break
        except Exception as exc:
            validator_import_errors.append((mod_name, exc))
    else:
        raise ImportError(
            "Failed to import build_mongodb_validators_from_ontology_blueprint. "
            f"Tried: {[name for name, _ in validator_import_errors]}"
        )

    # Step 3: build MongoDB data layer / import instances
    data_layer_import_errors = []
    for mod_name in [
        "scripts.create_mongodb_data_layer",
        "create_mongodb_data_layer",
    ]:
        try:
            module = __import__(
                mod_name,
                fromlist=[
                    "ensure_collection_block",
                    "write_ontology_meta",
                    "seed_or_import_instances",
                ],
            )
            symbols.update(
                {
                    "ensure_collection_block": getattr(module, "ensure_collection_block"),
                    "write_ontology_meta": getattr(module, "write_ontology_meta"),
                    "seed_or_import_instances": getattr(module, "seed_or_import_instances"),
                }
            )
            break
        except Exception as exc:
            data_layer_import_errors.append((mod_name, exc))
    else:
        raise ImportError(
            "Failed to import MongoDB data-layer builder helpers. "
            f"Tried: {[name for name, _ in data_layer_import_errors]}"
        )

    # Step 4: CRUD DAL
    crud_import_attempts = [
        (
            "scripts.crud_create",
            "scripts.crud_read",
            "scripts.crud_update",
            "scripts.crud_delete",
            "scripts.blueprint_registry",
        ),
        (
            "crud_create",
            "crud_read",
            "crud_update",
            "crud_delete",
            "blueprint_registry",
        ),
    ]
    crud_import_errors = []
    for create_mod, read_mod, update_mod, delete_mod, registry_mod in crud_import_attempts:
        try:
            create_module = __import__(create_mod, fromlist=["CreateDAL"])
            read_module = __import__(read_mod, fromlist=["ReadDAL"])
            update_module = __import__(update_mod, fromlist=["UpdateDAL"])
            delete_module = __import__(delete_mod, fromlist=["DeleteDAL"])
            registry_module = __import__(registry_mod, fromlist=["BlueprintRegistry"])
            symbols.update(
                {
                    "CreateDAL": getattr(create_module, "CreateDAL"),
                    "ReadDAL": getattr(read_module, "ReadDAL"),
                    "UpdateDAL": getattr(update_module, "UpdateDAL"),
                    "DeleteDAL": getattr(delete_module, "DeleteDAL"),
                    "BlueprintRegistry": getattr(registry_module, "BlueprintRegistry"),
                }
            )
            break
        except Exception as exc:
            crud_import_errors.append(((create_mod, read_mod, update_mod, delete_mod, registry_mod), exc))
    else:
        raise ImportError(
            "Failed to import CRUD DAL modules / BlueprintRegistry. "
            f"Tried package sets: {[mods for mods, _ in crud_import_errors]}"
        )

    # MongoDB / BSON
    try:
        from pymongo import MongoClient
        from bson import ObjectId

        symbols["MongoClient"] = MongoClient
        symbols["ObjectId"] = ObjectId
    except Exception as exc:
        raise ImportError("pymongo / bson is required for this demo.") from exc

    return symbols


# ======================================================================================
# Utilities
# ======================================================================================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def print_json_block(title: str, data: Any) -> None:
    print(title)
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def prompt_non_empty(prompt: str) -> str:
    while True:
        raw = input(prompt).strip()
        if raw:
            return raw
        print("Input cannot be empty.")


def prompt_optional(prompt: str, default: Optional[str] = None) -> str:
    raw = input(prompt).strip()
    if raw:
        return raw
    return default or ""


def prompt_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    raw = input(prompt + suffix).strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes", "true", "1"}


def prompt_json(prompt: str, allow_empty: bool = False, default: Optional[Any] = None) -> Any:
    while True:
        raw = input(prompt).strip()
        if not raw:
            if allow_empty:
                return default
            print("Input cannot be empty. Please enter valid JSON.")
            continue
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"Invalid JSON: {exc}")


def collection_for_class(ontology_blueprint: Dict[str, Any], class_name: str) -> Optional[str]:
    classes = ontology_blueprint.get("classes") or {}
    named_individuals = ontology_blueprint.get("named_individuals") or {}

    vocabulary_types = set()
    for ind in named_individuals.values():
        for t in ind.get("asserted_types", []) or []:
            if t in classes:
                vocabulary_types.add(t)

    if class_name in vocabulary_types and class_name == "Sex":
        return "sexes"

    import re

    def camel_to_snake(name: str) -> str:
        s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
        return s2.replace("-", "_").lower()

    def pluralize(name: str) -> str:
        if name.endswith("y") and len(name) > 1 and name[-2] not in "aeiou":
            return name[:-1] + "ies"
        if name.endswith(("s", "x", "z", "ch", "sh")):
            return name + "es"
        return name + "s"

    return pluralize(camel_to_snake(class_name))


# ======================================================================================
# STEP 1 - unified input handling
# ======================================================================================


def _build_blueprint_from_schema_owl(
        *,
        owl_path: Path,
        symbols: Dict[str, Any],
) -> Dict[str, Any]:
    parse_owl = symbols["parse_owl"]
    MappingConfig = symbols["MappingConfig"]
    MappingOverrides = symbols["MappingOverrides"]
    build_ontology_blueprint = symbols["build_ontology_blueprint"]
    ontology_blueprint_to_json = symbols["ontology_blueprint_to_json"]

    model = parse_owl(str(owl_path))
    overrides = MappingOverrides(ignore_classes={"owl:Thing"})
    cfg = MappingConfig(
        improved_naming=True,
        canonicalize_inverse=True,
        ignore_owl_thing=True,
        default_relation="reference",
        functional_implies_single=True,
        object_relation_mode="reference_only",
        max_embed_depth=None,
    )
    ontology_blueprint_obj = build_ontology_blueprint(
        model,
        overrides=overrides,
        cfg=cfg,
        owl_path=str(owl_path),
    )
    return json.loads(ontology_blueprint_to_json(ontology_blueprint_obj))


def step1_prepare_ontology_blueprint(
        *,
        input_mode: str,
        mixed_owl_path: Optional[Path],
        schema_owl_path: Optional[Path],
        instance_owl_path: Optional[Path],
        workdir: Path,
        symbols: Dict[str, Any],
) -> Dict[str, Any]:
    print("\n" + "=" * 90)
    print("STEP 1 - PREPARE ontology_blueprint.json")
    print("=" * 90)

    ontology_blueprint_path = workdir / "ontology_blueprint.json"

    if input_mode == "mixed":
        assert mixed_owl_path is not None
        enrich_blueprint = symbols["enrich_blueprint"]

        print("Input mode               : mixed (schema + data in one OWL)")

        # First build the pure ontology blueprint from the OWL schema view.
        # Then enrich that blueprint from the SAME mixed OWL so that
        # named_individuals also carry data_assertions/object_assertions.
        base_blueprint = _build_blueprint_from_schema_owl(
            owl_path=mixed_owl_path,
            symbols=symbols,
        )
        ontology_blueprint = enrich_blueprint(base_blueprint, str(mixed_owl_path))
        write_json(ontology_blueprint_path, ontology_blueprint)

        object_assertion_individuals = 0
        data_assertion_individuals = 0
        for ind in (ontology_blueprint.get("named_individuals") or {}).values():
            if ind.get("object_assertions"):
                object_assertion_individuals += 1
            if ind.get("data_assertions"):
                data_assertion_individuals += 1

        print(f"Input OWL                : {mixed_owl_path}")
        print(f"Generated file           : {ontology_blueprint_path}")
        print(f"Classes                  : {len(ontology_blueprint.get('classes', {}))}")
        print(f"Data properties          : {len(ontology_blueprint.get('data_properties', {}))}")
        print(f"Object properties        : {len(ontology_blueprint.get('object_properties', {}))}")
        print(f"Named individuals        : {len(ontology_blueprint.get('named_individuals', {}))}")
        print(f"Subclass axioms          : {len((ontology_blueprint.get('axioms') or {}).get('subclass_axioms', []))}")
        print(f"Individuals with object assertions : {object_assertion_individuals}")
        print(f"Individuals with data assertions   : {data_assertion_individuals}")

        return {
            "ontology_blueprint": ontology_blueprint,
            "ontology_blueprint_path": ontology_blueprint_path,
            "base_blueprint_path": None,
            "input_summary": {
                "input_mode": "mixed",
                "mixed_owl": str(mixed_owl_path),
                "object_assertion_individuals": object_assertion_individuals,
                "data_assertion_individuals": data_assertion_individuals,
            },
        }

    assert schema_owl_path is not None and instance_owl_path is not None
    enrich_blueprint = symbols["enrich_blueprint"]
    base_blueprint_path = workdir / "ontology_blueprint_base.json"

    print("Input mode               : separated (schema OWL + instance OWL)")
    base_blueprint = _build_blueprint_from_schema_owl(
        owl_path=schema_owl_path,
        symbols=symbols,
    )
    write_json(base_blueprint_path, base_blueprint)

    enriched_blueprint = enrich_blueprint(base_blueprint, str(instance_owl_path))
    write_json(ontology_blueprint_path, enriched_blueprint)

    print(f"Schema OWL               : {schema_owl_path}")
    print(f"Instance OWL             : {instance_owl_path}")
    print(f"Base blueprint file      : {base_blueprint_path}")
    print(f"Generated file           : {ontology_blueprint_path}")
    print(f"Classes                  : {len(enriched_blueprint.get('classes', {}))}")
    print(f"Data properties          : {len(enriched_blueprint.get('data_properties', {}))}")
    print(f"Object properties        : {len(enriched_blueprint.get('object_properties', {}))}")
    print(f"Named individuals        : {len(enriched_blueprint.get('named_individuals', {}))}")
    print(f"Subclass axioms          : {len((enriched_blueprint.get('axioms') or {}).get('subclass_axioms', []))}")

    return {
        "ontology_blueprint": enriched_blueprint,
        "ontology_blueprint_path": ontology_blueprint_path,
        "base_blueprint_path": base_blueprint_path,
        "input_summary": {
            "input_mode": "separated",
            "schema_owl": str(schema_owl_path),
            "instance_owl": str(instance_owl_path),
        },
    }


# ======================================================================================
# STEP 2
# ======================================================================================


def step2_generate_validators(
        *,
        ontology_blueprint: Dict[str, Any],
        workdir: Path,
        symbols: Dict[str, Any],
) -> Dict[str, Any]:
    print("\n" + "=" * 90)
    print("STEP 2 - GENERATE mongo_validators.json FROM ontology_blueprint.json")
    print("=" * 90)

    build_validators = symbols["build_mongodb_validators_from_ontology_blueprint"]
    validators_path = workdir / "mongo_validators.json"

    validators = build_validators(ontology_blueprint)
    write_json(validators_path, validators)

    print(f"Generated file           : {validators_path}")
    print(f"Collections with schema  : {len(validators)}")

    return {
        "validators": validators,
        "validators_path": validators_path,
    }


# ======================================================================================
# STEP 3
# ======================================================================================


def step3_build_mongodb_data_layer(
        *,
        ontology_blueprint: Dict[str, Any],
        validators: Dict[str, Any],
        mongo_uri: str,
        db_name: str,
        workdir: Path,
        symbols: Dict[str, Any],
        drop_collections: bool,
) -> Dict[str, Any]:
    print("\n" + "=" * 90)
    print("STEP 3 - BUILD MONGODB DATA LAYER AND IMPORT INSTANCES")
    print("=" * 90)

    MongoClient = symbols["MongoClient"]
    ensure_collection_block = symbols["ensure_collection_block"]
    write_ontology_meta = symbols["write_ontology_meta"]
    seed_or_import_instances = symbols["seed_or_import_instances"]

    effective_blueprint_path = workdir / "ontology_blueprint_effective.json"
    import_summary_path = workdir / "import_summary.json"

    client = MongoClient(mongo_uri)
    db = client[db_name]

    if drop_collections:
        try:
            db.client.drop_database(db_name)
            print(f"Dropped database         : {db_name}")
        except Exception as exc:
            print(f"[WARN] Could not drop database '{db_name}': {exc}")

    try:
        ensure_collection_block(db, validators, False)
    except TypeError:
        try:
            ensure_collection_block(db=db, validators=validators, drop_collections=False)
        except Exception as exc:
            raise RuntimeError(
                "ensure_collection_block exists but could not be called with known signatures."
            ) from exc

    print(f"Validator-applied collections: {len(validators)}")

    try:
        write_ontology_meta(db, ontology_blueprint, validators)
    except TypeError:
        try:
            write_ontology_meta(db=db, ontology_bp=ontology_blueprint, validators=validators)
        except Exception as exc:
            print(f"[WARN] write_ontology_meta signature mismatch, skipped: {exc}")
    except Exception as exc:
        print(f"[WARN] write_ontology_meta failed, skipped: {exc}")

    import_summary: Dict[str, Any] = {}
    try:
        import_summary = seed_or_import_instances(
            db=db,
            validators=validators,
            ontology_bp=ontology_blueprint,
        )
    except TypeError:
        try:
            import_summary = seed_or_import_instances(
                db,
                validators,
                ontology_blueprint,
            )
        except TypeError:
            try:
                import_summary = seed_or_import_instances(
                    db,
                    validators,
                    ontology_blueprint,
                    [],
                )
            except Exception as exc:
                raise RuntimeError(
                    "seed_or_import_instances exists but could not be called with known signatures."
                ) from exc
        except Exception as exc:
            raise RuntimeError(
                "seed_or_import_instances exists but could not be called with known signatures."
            ) from exc

    write_json(effective_blueprint_path, ontology_blueprint)
    write_json(import_summary_path, import_summary)

    print(f"Effective blueprint file : {effective_blueprint_path}")
    print(f"Import summary file      : {import_summary_path}")

    inserted_or_upserted = import_summary.get("inserted_or_upserted")
    delayed = import_summary.get("delayed_inserted")
    ref_updates = import_summary.get("reference_updates")
    resolution_rounds = import_summary.get("resolution_rounds")
    skipped_no_type = import_summary.get("skipped_no_type")
    skipped_no_collection = import_summary.get("skipped_no_collection")
    skipped_incomplete = import_summary.get("skipped_still_incomplete")
    unresolved_import_refs = import_summary.get("unresolved_import_references")

    if inserted_or_upserted is not None:
        print(f"Inserted / upserted      : {inserted_or_upserted}")
    if delayed is not None:
        print(f"Delayed inserted         : {delayed}")
    if ref_updates is not None:
        print(f"Reference updates        : {ref_updates}")
    if resolution_rounds is not None:
        print(f"Resolution rounds        : {resolution_rounds}")
    if skipped_no_type is not None:
        print(f"Skipped (no type)        : {skipped_no_type}")
    if skipped_no_collection is not None:
        print(f"Skipped (no collection)  : {skipped_no_collection}")
    if skipped_incomplete is not None:
        print(f"Skipped (incomplete)     : {skipped_incomplete}")
    if unresolved_import_refs is not None:
        print(f"Unresolved import refs   : {unresolved_import_refs}")

    return {
        "effective_blueprint_path": effective_blueprint_path,
        "import_summary": import_summary,
        "import_summary_path": import_summary_path,
    }


# ======================================================================================
# STEP 4 - interactive CRUD validation
# ======================================================================================


def _print_interactive_help() -> None:
    print("\nAvailable actions:")
    print("  create  - create one document from a JSON payload")
    print("  read    - read by id / get_one(query) / list(query)")
    print("  update  - update one document by id using a JSON patch")
    print("  delete  - preview or execute one document delete by id")
    print("  help    - show this help")
    print("  exit    - finish the interactive session")


def _interactive_create(create_dal: Any) -> Dict[str, Any]:
    class_name = prompt_non_empty("Class name for create: ")
    payload = prompt_json("Create payload JSON: ")
    return create_dal.create_document(class_name, payload, return_document=True)


def _interactive_read(read_dal: Any) -> Dict[str, Any]:
    mode = prompt_optional("Read mode [id/one/list] (default=id): ", default="id").lower() or "id"
    target = prompt_non_empty("Class or collection: ")
    depth_raw = prompt_optional("Depth (default=0): ", default="0") or "0"
    include_metadata = prompt_yes_no("Include metadata?", default=True)

    try:
        depth = int(depth_raw)
    except Exception:
        depth = 0

    if mode == "id":
        document_id = prompt_non_empty("Document _id: ")
        doc = read_dal.get_by_id(target, document_id, depth=depth, include_metadata=include_metadata)
        return {"ok": doc is not None, "mode": "id", "result": doc}

    if mode == "one":
        query = prompt_json("Query JSON: ")
        doc = read_dal.get_one(target, query, depth=depth, include_metadata=include_metadata)
        return {"ok": doc is not None, "mode": "one", "query": query, "result": doc}

    query = prompt_json("Query JSON (empty object {} for all): ")
    limit_raw = prompt_optional("Limit (default=20): ", default="20") or "20"
    try:
        limit = int(limit_raw)
    except Exception:
        limit = 20
    docs = read_dal.list_documents(target, query=query, limit=limit, depth=depth, include_metadata=include_metadata)
    return {"ok": True, "mode": "list", "query": query, "limit": limit, "count": len(docs), "result": docs}


def _interactive_update(update_dal: Any) -> Dict[str, Any]:
    class_name = prompt_non_empty("Class or collection for update: ")
    document_id = prompt_non_empty("Document _id to update: ")
    patch = prompt_json("Patch JSON: ")
    return update_dal.update_by_id(class_name, document_id, patch, return_document=True)


def _choose_delete_mode_with_preview(delete_dal: Any, class_name: str, document_id: str) -> Dict[str, Any]:
    candidate_modes = ["restrict", "detach_if_valid", "force_detach"]
    previews: Dict[str, Any] = {}

    print("\nDelete preview (automatic):")
    print("-" * 90)
    for mode in candidate_modes:
        preview = delete_dal.preview_delete(class_name, document_id, delete_mode=mode)
        previews[mode] = preview

        status = "OK" if preview.get("ok") else "BLOCKED"
        warning_count = len(preview.get("warnings") or [])
        error_count = len(preview.get("errors") or [])
        print(f"[{mode}] {status} | warnings={warning_count} | errors={error_count}")

        target_document = preview.get("target_document") or {}
        if target_document:
            print(
                "  target:",
                f"type={target_document.get('_ontology_type')}",
                f"name={target_document.get('name')}",
                f"_id={target_document.get('_id')}",
            )

        plan = preview.get("plan") or {}
        if plan:
            affected_groups = plan.get("affected_reference_groups") or []
            print(f"  affected_reference_groups={len(affected_groups)} | update_count={plan.get('update_count', 0)}")

        first_error = (preview.get("errors") or [None])[0]
        if first_error:
            print(f"  first_error: {first_error.get('code')} - {first_error.get('message')}")

        first_warning = (preview.get("warnings") or [None])[0]
        if first_warning:
            print(f"  first_warning: {first_warning.get('code')} - {first_warning.get('message')}")

        print()

    suggested_mode = next((mode for mode in candidate_modes if previews[mode].get("ok")), "restrict")
    print(f"Suggested delete mode: {suggested_mode}")

    return {"previews": previews, "suggested_mode": suggested_mode}


def _interactive_delete(delete_dal: Any) -> Dict[str, Any]:
    class_name = prompt_non_empty("Class or collection for delete: ")
    document_id = prompt_non_empty("Document _id to delete: ")

    preview_bundle = _choose_delete_mode_with_preview(delete_dal, class_name, document_id)
    previews = preview_bundle["previews"]
    suggested_mode = preview_bundle["suggested_mode"]

    preview_ok = any((preview or {}).get("ok") for preview in previews.values())
    if not preview_ok:
        print("Delete is blocked in all modes. Returning the restrict preview result.")
        result = deepcopy(previews.get("restrict") or {})
        result["executed"] = False
        result["cancelled"] = True
        result["preview_bundle"] = previews
        return result

    proceed = prompt_yes_no("Proceed with delete after preview?", default=False)
    if not proceed:
        result = deepcopy(previews.get(suggested_mode) or {})
        result["executed"] = False
        result["cancelled"] = True
        result["preview_bundle"] = previews
        return result

    delete_mode = prompt_optional(
        f"Delete mode [restrict/detach_if_valid/force_detach] (default={suggested_mode}): ",
        default=suggested_mode,
    ).lower() or suggested_mode

    selected_preview = delete_dal.preview_delete(class_name, document_id, delete_mode=delete_mode)
    print_json_block("Selected delete-mode preview:", selected_preview)

    if not selected_preview.get("ok"):
        print("Selected delete mode is not allowed. Returning preview result without execution.")
        selected_preview["executed"] = False
        selected_preview["cancelled"] = True
        selected_preview["preview_bundle"] = previews
        return selected_preview

    confirm_execute = prompt_yes_no("Confirm execution?", default=False)
    if not confirm_execute:
        selected_preview["executed"] = False
        selected_preview["cancelled"] = True
        selected_preview["preview_bundle"] = previews
        return selected_preview

    result = delete_dal.delete_by_id(class_name, document_id, delete_mode=delete_mode)
    result["preview_bundle"] = previews
    return result


def step4_interactive_dal_validation(
        *,
        ontology_blueprint_path: Path,
        validators_path: Path,
        mongo_uri: str,
        db_name: str,
        workdir: Path,
        symbols: Dict[str, Any],
) -> Dict[str, Any]:
    print("\n" + "=" * 90)
    print("STEP 4 - INTERACTIVE DAL VALIDATION")
    print("=" * 90)

    CreateDAL = symbols["CreateDAL"]
    ReadDAL = symbols["ReadDAL"]
    UpdateDAL = symbols["UpdateDAL"]
    DeleteDAL = symbols["DeleteDAL"]

    session_log_path = workdir / "crud_session_log.json"

    create_dal = CreateDAL.from_files(ontology_blueprint_path, validators_path, mongo_uri, db_name)
    read_dal = ReadDAL.from_files(ontology_blueprint_path, validators_path, mongo_uri, db_name)
    update_dal = UpdateDAL.from_files(ontology_blueprint_path, validators_path, mongo_uri, db_name)
    delete_dal = DeleteDAL.from_files(ontology_blueprint_path, validators_path, mongo_uri, db_name)

    session_log: List[Dict[str, Any]] = []
    _print_interactive_help()

    while True:
        action = prompt_optional("\nChoose action (create/read/update/delete/help/exit): ",
                                 default="help").lower().strip()
        if action == "exit":
            break
        if action == "help" or not action:
            _print_interactive_help()
            continue

        try:
            if action == "create":
                result = _interactive_create(create_dal)
            elif action == "read":
                result = _interactive_read(read_dal)
            elif action == "update":
                result = _interactive_update(update_dal)
            elif action == "delete":
                result = _interactive_delete(delete_dal)
            else:
                print(f"Unknown action: {action}")
                continue
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            continue
        except Exception as exc:
            result = {"ok": False, "action": action, "unexpected_exception": str(exc)}

        entry = {
            "action": action,
            "result": deepcopy(result),
        }
        session_log.append(entry)
        write_json(session_log_path, session_log)

        if isinstance(result, dict) and result.get("ok"):
            print_json_block("Operation result (success):", result)
        else:
            print_json_block("Operation result (failure / validation feedback):", result)

    print(f"CRUD session log file    : {session_log_path}")
    return {
        "crud_session_log_path": session_log_path,
        "operation_count": len(session_log),
        "last_result": session_log[-1]["result"] if session_log else None,
    }


# ======================================================================================
# MANIFEST
# ======================================================================================


def write_manifest(
        *,
        input_summary: Dict[str, Any],
        mongo_uri: str,
        db_name: str,
        workdir: Path,
) -> Path:
    manifest = {
        "demo": "unified_pipeline_with_interactive_dal",
        "input_mode": input_summary.get("input_mode"),
        "input_summary": input_summary,
        "mongo_uri": mongo_uri,
        "db_name": db_name,
        "workdir": str(workdir),
        "generated_files": {
            "ontology_blueprint": str(workdir / "ontology_blueprint.json"),
            "ontology_blueprint_base": str(workdir / "ontology_blueprint_base.json"),
            "mongo_validators": str(workdir / "mongo_validators.json"),
            "ontology_blueprint_effective": str(workdir / "ontology_blueprint_effective.json"),
            "import_summary": str(workdir / "import_summary.json"),
            "crud_session_log": str(workdir / "crud_session_log.json"),
        },
        "steps": [
            "STEP 1 - Prepare ontology_blueprint.json (mixed or separated input mode)",
            "STEP 2 - Generate mongo_validators.json from ontology_blueprint.json",
            "STEP 3 - Build MongoDB data layer and import instances",
            "STEP 4 - Interactive CRUD validation through the DAL",
        ],
    }
    manifest_path = workdir / "demo_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


# ======================================================================================
# MAIN
# ======================================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Unified demo: mixed/separated OWL -> ontology blueprint -> Mongo validators -> "
            "MongoDB data layer -> interactive DAL validation"
        )
    )
    parser.add_argument(
        "--input-mode",
        choices=["mixed", "separated"],
        default="mixed",
        help="Choose one mixed OWL input or separated schema/data inputs",
    )
    parser.add_argument(
        "owl_path",
        nargs="?",
        help="Path to mixed OWL when --input-mode=mixed",
    )
    parser.add_argument("--schema-owl", type=str, help="Path to schema OWL when --input-mode=separated")
    parser.add_argument("--instance-owl", type=str, help="Path to instance OWL when --input-mode=separated")
    parser.add_argument(
        "--mongo_uri",
        type=str,
        default="mongodb://localhost:27017",
        help="MongoDB connection URI",
    )
    parser.add_argument(
        "--db",
        dest="db_name",
        type=str,
        default="demo_db",
        help="Database name used for this demo run",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default="demo_outputs",
        help="Directory where generated JSON artifacts will be written",
    )
    parser.add_argument(
        "--keep-db",
        action="store_true",
        help="Do not drop the target database before building the MongoDB data layer",
    )
    args = parser.parse_args()

    mixed_owl_path: Optional[Path] = None
    schema_owl_path: Optional[Path] = None
    instance_owl_path: Optional[Path] = None

    if args.input_mode == "mixed":
        if not args.owl_path:
            raise ValueError("When --input-mode=mixed, you must provide the mixed OWL path as the positional owl_path.")
        mixed_owl_path = Path(args.owl_path).resolve()
        if not mixed_owl_path.exists():
            raise FileNotFoundError(f"OWL file not found: {mixed_owl_path}")
    else:
        if not args.schema_owl or not args.instance_owl:
            raise ValueError("When --input-mode=separated, you must provide --schema-owl and --instance-owl.")
        schema_owl_path = Path(args.schema_owl).resolve()
        instance_owl_path = Path(args.instance_owl).resolve()
        if not schema_owl_path.exists():
            raise FileNotFoundError(f"Schema OWL file not found: {schema_owl_path}")
        if not instance_owl_path.exists():
            raise FileNotFoundError(f"Instance OWL file not found: {instance_owl_path}")

    workdir = Path(args.workdir).resolve()
    ensure_dir(workdir)

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    symbols = import_project_symbols()

    step1 = step1_prepare_ontology_blueprint(
        input_mode=args.input_mode,
        mixed_owl_path=mixed_owl_path,
        schema_owl_path=schema_owl_path,
        instance_owl_path=instance_owl_path,
        workdir=workdir,
        symbols=symbols,
    )

    manifest_path = write_manifest(
        input_summary=step1["input_summary"],
        mongo_uri=args.mongo_uri,
        db_name=args.db_name,
        workdir=workdir,
    )
    print(f"Manifest file            : {manifest_path}")

    step2 = step2_generate_validators(
        ontology_blueprint=step1["ontology_blueprint"],
        workdir=workdir,
        symbols=symbols,
    )

    step3 = step3_build_mongodb_data_layer(
        ontology_blueprint=step1["ontology_blueprint"],
        validators=step2["validators"],
        mongo_uri=args.mongo_uri,
        db_name=args.db_name,
        workdir=workdir,
        symbols=symbols,
        drop_collections=not args.keep_db,
    )

    step4 = step4_interactive_dal_validation(
        ontology_blueprint_path=step1["ontology_blueprint_path"],
        validators_path=step2["validators_path"],
        mongo_uri=args.mongo_uri,
        db_name=args.db_name,
        workdir=workdir,
        symbols=symbols,
    )

    print("\n" + "=" * 90)
    print("UNIFIED DEMO FINISHED")
    print("=" * 90)
    print(f"Workdir                  : {workdir}")
    if step1.get("base_blueprint_path"):
        print(f"Base blueprint           : {step1['base_blueprint_path']}")
    print(f"Ontology blueprint       : {step1['ontology_blueprint_path']}")
    print(f"Mongo validators         : {step2['validators_path']}")
    print(f"Effective blueprint      : {step3['effective_blueprint_path']}")
    print(f"Import summary           : {step3['import_summary_path']}")
    print(f"CRUD session log         : {step4['crud_session_log_path']}")
    print(f"Operations executed      : {step4['operation_count']}")


if __name__ == "__main__":
    main()
