from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ============================================================
# Helpers
# ============================================================

SPECIAL_NORMALIZATION = {
    "Apperance": "appearance",
}

VOCAB_COLLECTION_OVERRIDES = {
    "Sex": "sexes",
    "Channel": "channels",
    "LifeActivity": "life_activities",
    "Modality": "modalities",
}


def _to_snake(name: str) -> str:
    out: List[str] = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0 and (
                name[i - 1].islower() or (i + 1 < len(name) and name[i + 1].islower())
        ):
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def normalize_name(name: str) -> str:
    return SPECIAL_NORMALIZATION.get(name, _to_snake(name))


def pluralize(name: str) -> str:
    if name.endswith("s"):
        return name + "es"
    if name.endswith("y") and len(name) > 1 and name[-2] not in "aeiou":
        return name[:-1] + "ies"
    return name + "s"


def collection_name_for_class(class_name: str, concrete_named_individual_types: Set[str]) -> str:
    if class_name in concrete_named_individual_types:
        return VOCAB_COLLECTION_OVERRIDES.get(class_name, pluralize(normalize_name(class_name)))
    return pluralize(normalize_name(class_name))


def bson_type_for_range(dt: Optional[str]) -> str:
    if not dt:
        return "string"
    lower = dt.lower()
    if "objectid" in lower:
        return "objectId"
    if "int" in lower:
        return "int"
    if "float" in lower or "double" in lower or "decimal" in lower:
        return "double"
    if "bool" in lower:
        return "bool"
    if "date" in lower or "time" in lower:
        return "date"
    return "string"


def annotation_values(obj: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    return (obj.get("annotations", {}) or {}).get(key, []) or []


def first_annotation_text(obj: Dict[str, Any], key: str) -> Optional[str]:
    vals = annotation_values(obj, key)
    if not vals:
        return None
    return vals[0].get("value")


def is_abstract_class(cls_obj: Dict[str, Any]) -> bool:
    for ann in annotation_values(cls_obj, "conceptType"):
        if str(ann.get("value", "")).lower() == "abstract":
            return True
    return False


def ensure_object_schema(node: Dict[str, Any]) -> None:
    if node.get("bsonType") != "object":
        node["bsonType"] = "object"
    node.setdefault("properties", {})


def mark_required(obj_node: Dict[str, Any], field: str) -> None:
    req = obj_node.setdefault("required", [])
    if field not in req:
        req.append(field)


def set_schema_at_path(root: Dict[str, Any], path: str, schema: Dict[str, Any], required: bool) -> None:
    parts = [p for p in path.split(".") if p]
    if not parts:
        return

    cur = root
    for i, part in enumerate(parts):
        ensure_object_schema(cur)
        props = cur["properties"]
        is_last = i == len(parts) - 1

        if is_last:
            props[part] = schema
            if required:
                mark_required(cur, part)
        else:
            if part not in props or props[part].get("bsonType") != "object":
                props[part] = {"bsonType": "object", "properties": {}}
            cur = props[part]


def prune_empty_required(node: Any) -> None:
    if isinstance(node, dict):
        for v in list(node.values()):
            prune_empty_required(v)
        if "required" in node and isinstance(node["required"], list) and not node["required"]:
            node.pop("required", None)
    elif isinstance(node, list):
        for item in node:
            prune_empty_required(item)


# ============================================================
# Model
# ============================================================

@dataclass
class ScalarField:
    name: str
    bson_type: str
    required: bool = False


@dataclass
class RefField:
    name: str
    bson_type: str = "objectId"
    is_array: bool = False
    required: bool = False
    min_items: Optional[int] = None
    max_items: Optional[int] = None
    target_collection: Optional[str] = None
    source_property: Optional[str] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class CollectionSpec:
    name: str
    source_class: str
    fields: Dict[str, ScalarField] = field(default_factory=dict)
    refs: Dict[str, RefField] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


@dataclass
class ResolvedConstraint:
    required: bool = False
    min_items: Optional[int] = None
    max_items: Optional[int] = None
    allowed_target_classes: Set[str] = field(default_factory=set)


# ============================================================
# Ontology helpers
# ============================================================

def build_parent_child_maps(axioms: Dict[str, Any]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    parents: Dict[str, Set[str]] = {}
    children: Dict[str, Set[str]] = {}
    for ax in axioms.get("subclass_axioms", []):
        sub = ax["sub_class"]
        sup = ax["super_class"]
        parents.setdefault(sub, set()).add(sup)
        children.setdefault(sup, set()).add(sub)
        parents.setdefault(sup, set())
        children.setdefault(sub, set())
    return parents, children


def transitive_closure(seed: str, relation: Dict[str, Set[str]]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    stack = list(relation.get(seed, set()))
    while stack:
        item = stack.pop()
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
        stack.extend(relation.get(item, set()))
    return sorted(out)


def all_ancestors(cls_name: str, parents_map: Dict[str, Set[str]]) -> List[str]:
    return transitive_closure(cls_name, parents_map)


def concrete_descendants(
        domain: str,
        classes: Dict[str, Any],
        children_map: Dict[str, Set[str]],
) -> List[str]:
    descendants = transitive_closure(domain, children_map)
    out: List[str] = []
    seen: Set[str] = set()

    if domain in classes and not is_abstract_class(classes[domain]):
        out.append(domain)
        seen.add(domain)

    for c in descendants:
        if c in classes and not is_abstract_class(classes[c]) and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def resolve_target_classes(
        declared_ranges: List[str],
        allowed_target_classes: Set[str],
        classes: Dict[str, Any],
        children_map: Dict[str, Set[str]],
) -> List[str]:
    targets = set(allowed_target_classes) if allowed_target_classes else set(declared_ranges)
    expanded: List[str] = []
    for t in sorted(targets):
        if t in classes and is_abstract_class(classes[t]):
            expanded.extend(concrete_descendants(t, classes, children_map))
        else:
            expanded.append(t)

    out: List[str] = []
    seen: Set[str] = set()
    for t in expanded:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ============================================================
# Constraint resolution
# ============================================================

def merge_min(current: Optional[int], new: Optional[int]) -> Optional[int]:
    if new is None:
        return current
    if current is None:
        return new
    return max(current, new)


def merge_max(current: Optional[int], new: Optional[int]) -> Optional[int]:
    if new is None:
        return current
    if current is None:
        return new
    return min(current, new)


def resolve_forward_constraints_for_property(
        class_name: str,
        prop_name: str,
        prop_obj: Dict[str, Any],
        ontology_bp: Dict[str, Any],
        parents_map: Dict[str, Set[str]],
) -> ResolvedConstraint:
    rc = ResolvedConstraint()

    if prop_obj.get("is_functional"):
        rc.max_items = 1

    cards = prop_obj.get("forward_cardinality_by_domain", {}) or {}
    for ancestor in [class_name] + all_ancestors(class_name, parents_map):
        card = cards.get(ancestor)
        if card:
            rc.min_items = merge_min(rc.min_items, card.get("min"))
            rc.max_items = merge_max(rc.max_items, card.get("max"))

    all_subjects = [class_name] + all_ancestors(class_name, parents_map)
    for r in ontology_bp.get("axioms", {}).get("restriction_axioms", []):
        if r.get("subject_class") not in all_subjects:
            continue
        if r.get("property_name") != prop_name:
            continue
        if r.get("inverse"):
            continue

        rtype = r.get("restriction_type")
        filler = r.get("filler")

        if rtype in {"object_some_values_from", "data_some_values_from"}:
            rc.required = True
            rc.min_items = merge_min(rc.min_items, 1)
            if filler:
                rc.allowed_target_classes.add(filler)
        elif rtype == "object_exact_cardinality":
            n = r.get("cardinality")
            if n is not None and n >= 1:
                rc.required = True
            rc.min_items = merge_min(rc.min_items, n)
            rc.max_items = merge_max(rc.max_items, n)
            if filler:
                rc.allowed_target_classes.add(filler)
        elif rtype == "object_min_cardinality":
            n = r.get("cardinality")
            if n is not None and n >= 1:
                rc.required = True
            rc.min_items = merge_min(rc.min_items, n)
            if filler:
                rc.allowed_target_classes.add(filler)
        elif rtype == "object_max_cardinality":
            n = r.get("cardinality")
            rc.max_items = merge_max(rc.max_items, n)
            if filler:
                rc.allowed_target_classes.add(filler)
        elif rtype == "object_all_values_from":
            if filler:
                rc.allowed_target_classes.add(filler)

    return rc


def resolve_inverse_constraints_for_property(
        target_class: str,
        prop_name: str,
        prop_obj: Dict[str, Any],
        ontology_bp: Dict[str, Any],
        parents_map: Dict[str, Set[str]],
) -> ResolvedConstraint:
    rc = ResolvedConstraint()

    inv_cards = prop_obj.get("inverse_cardinality_by_domain", {}) or {}
    for ancestor in [target_class] + all_ancestors(target_class, parents_map):
        card = inv_cards.get(ancestor)
        if card:
            rc.min_items = merge_min(rc.min_items, card.get("min"))
            rc.max_items = merge_max(rc.max_items, card.get("max"))
            if (card.get("min") or 0) >= 1:
                rc.required = True

    all_subjects = [target_class] + all_ancestors(target_class, parents_map)
    for r in ontology_bp.get("axioms", {}).get("restriction_axioms", []):
        if r.get("subject_class") not in all_subjects:
            continue
        if r.get("property_name") != prop_name:
            continue
        if not r.get("inverse"):
            continue

        rtype = r.get("restriction_type")
        filler = r.get("filler")

        if rtype == "object_some_values_from":
            rc.required = True
            rc.min_items = merge_min(rc.min_items, 1)
            if filler:
                rc.allowed_target_classes.add(filler)
        elif rtype == "object_exact_cardinality":
            n = r.get("cardinality")
            if n is not None and n >= 1:
                rc.required = True
            rc.min_items = merge_min(rc.min_items, n)
            rc.max_items = merge_max(rc.max_items, n)
            if filler:
                rc.allowed_target_classes.add(filler)
        elif rtype == "object_min_cardinality":
            n = r.get("cardinality")
            if n is not None and n >= 1:
                rc.required = True
            rc.min_items = merge_min(rc.min_items, n)
            if filler:
                rc.allowed_target_classes.add(filler)
        elif rtype == "object_max_cardinality":
            n = r.get("cardinality")
            rc.max_items = merge_max(rc.max_items, n)
            if filler:
                rc.allowed_target_classes.add(filler)
        elif rtype == "object_all_values_from":
            if filler:
                rc.allowed_target_classes.add(filler)

    return rc


def data_property_is_required(
        concrete_cls: str,
        prop_name: str,
        ontology_bp: Dict[str, Any],
        parents_map: Dict[str, Set[str]],
) -> bool:
    all_subjects = [concrete_cls] + all_ancestors(concrete_cls, parents_map)
    for r in ontology_bp.get("axioms", {}).get("restriction_axioms", []):
        if r.get("subject_class") not in all_subjects:
            continue
        if r.get("property_name") != prop_name:
            continue
        if r.get("restriction_type") == "data_some_values_from" and not r.get("inverse", False):
            return True
    return False


# ============================================================
# Main builder
# ============================================================

def build_collection_specs(ontology_bp: Dict[str, Any]) -> Dict[str, CollectionSpec]:
    classes: Dict[str, Any] = ontology_bp.get("classes", {}) or {}
    data_properties: Dict[str, Any] = ontology_bp.get("data_properties", {}) or {}
    object_properties: Dict[str, Any] = ontology_bp.get("object_properties", {}) or {}
    named_individuals: Dict[str, Any] = ontology_bp.get("named_individuals", {}) or {}
    axioms: Dict[str, Any] = ontology_bp.get("axioms", {}) or {}

    parents_map, children_map = build_parent_child_maps(axioms)

    named_individuals_by_type: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for ind_name, ind in named_individuals.items():
        for asserted_type in ind.get("asserted_types", []) or []:
            named_individuals_by_type.setdefault(asserted_type, []).append((ind_name, ind))

    concrete_named_individual_types = {
        cls_name
        for cls_name, inds in named_individuals_by_type.items()
        if inds and cls_name in classes and not is_abstract_class(classes[cls_name])
    }

    class_to_collection: Dict[str, str] = {}
    specs: Dict[str, CollectionSpec] = {}

    # Create collections for concrete classes and vocabulary classes.
    for class_name, cls_obj in sorted(classes.items()):
        if is_abstract_class(cls_obj):
            continue

        col_name = collection_name_for_class(class_name, concrete_named_individual_types)
        class_to_collection[class_name] = col_name
        if col_name not in specs:
            specs[col_name] = CollectionSpec(name=col_name, source_class=class_name)
        else:
            # defensive only; should not occur in this ontology
            specs[col_name].notes.append(f"Multiple source classes mapped to same collection: {class_name}")

    # Add default metadata fields to all collections.
    for spec in specs.values():
        spec.fields["metadata.createdAt"] = ScalarField("metadata.createdAt", "date", required=False)
        spec.fields["metadata.createdBy"] = ScalarField("metadata.createdBy", "string", required=False)
        spec.fields["metadata.updatedAt"] = ScalarField("metadata.updatedAt", "date", required=False)
        spec.fields["metadata.notes"] = ScalarField("metadata.notes", "string", required=False)

    # Add vocabulary-friendly scalar fields to seeded collections.
    for vocab_class in sorted(concrete_named_individual_types):
        col_name = class_to_collection[vocab_class]
        spec = specs[col_name]
        spec.fields["name"] = ScalarField("name", "string", required=True)
        spec.fields["iri"] = ScalarField("iri", "string", required=False)
        spec.fields["label"] = ScalarField("label", "string", required=False)

    # Data properties.
    for prop_name, dp in sorted(data_properties.items()):
        domains = dp.get("domains", []) or []
        ranges = dp.get("ranges", []) or []
        bson_type = bson_type_for_range(ranges[0] if ranges else None)
        field_name = normalize_name(prop_name)

        if "owl:Thing" in domains:
            for concrete_cls, col_name in sorted(class_to_collection.items()):
                required = data_property_is_required(concrete_cls, prop_name, ontology_bp, parents_map)
                specs[col_name].fields[field_name] = ScalarField(field_name, bson_type, required=required)

        for domain in domains:
            if domain == "owl:Thing":
                continue
            for concrete_cls in concrete_descendants(domain, classes, children_map):
                col_name = class_to_collection.get(concrete_cls)
                if not col_name:
                    continue
                required = data_property_is_required(concrete_cls, prop_name, ontology_bp, parents_map)
                specs[col_name].fields[field_name] = ScalarField(field_name, bson_type, required=required)

    # Forward object properties.
    for prop_name, op in sorted(object_properties.items()):
        domains = op.get("domains", []) or []
        ranges = op.get("ranges", []) or []
        if not domains:
            continue

        logical_name = normalize_name(prop_name)
        for domain in domains:
            for concrete_cls in concrete_descendants(domain, classes, children_map):
                source_col = class_to_collection.get(concrete_cls)
                if not source_col:
                    continue

                rc = resolve_forward_constraints_for_property(concrete_cls, prop_name, op, ontology_bp, parents_map)
                target_classes = resolve_target_classes(ranges, rc.allowed_target_classes, classes, children_map)

                target_collection = None
                if len(target_classes) == 1:
                    target_collection = class_to_collection.get(target_classes[0])

                is_array = not (op.get("is_functional") or rc.max_items == 1)
                ref_name = f"{logical_name}_ids" if is_array else f"{logical_name}_id"

                specs[source_col].refs[ref_name] = RefField(
                    name=ref_name,
                    is_array=is_array,
                    required=rc.required,
                    min_items=rc.min_items if is_array else None,
                    max_items=rc.max_items if is_array else None,
                    target_collection=target_collection,
                    source_property=prop_name,
                )

    # Reverse refs driven by inverse restrictions / inverse cardinalities.
    # This matches the current project style where, for example,
    # GroupActivityExecution gets has_activity_execution_ids.
    for prop_name, op in sorted(object_properties.items()):
        inverse_cards = op.get("inverse_cardinality_by_domain", {}) or {}
        relevant_subjects: Set[str] = set(inverse_cards.keys())

        for r in axioms.get("restriction_axioms", []):
            if r.get("property_name") == prop_name and r.get("inverse"):
                subj = r.get("subject_class")
                if subj:
                    relevant_subjects.add(subj)

        if not relevant_subjects:
            continue

        source_domains = op.get("domains", []) or []
        if not source_domains:
            continue

        logical_name = normalize_name(prop_name)

        for subject_class in sorted(relevant_subjects):
            if subject_class not in classes:
                continue
            if is_abstract_class(classes[subject_class]):
                continue

            for concrete_target_cls in concrete_descendants(subject_class, classes, children_map):
                target_col_name = class_to_collection.get(concrete_target_cls)
                if not target_col_name:
                    continue

                rc = resolve_inverse_constraints_for_property(
                    concrete_target_cls,
                    prop_name,
                    op,
                    ontology_bp,
                    parents_map,
                )

                source_classes_for_ref: List[str] = []
                for source_domain in source_domains:
                    source_classes_for_ref.extend(concrete_descendants(source_domain, classes, children_map))
                source_classes_for_ref = sorted(set(source_classes_for_ref))

                target_collection = None
                if len(source_classes_for_ref) == 1:
                    target_collection = class_to_collection.get(source_classes_for_ref[0])

                is_array = not (rc.max_items == 1)
                ref_name = f"{logical_name}_ids" if is_array else f"{logical_name}_id"

                specs[target_col_name].refs[ref_name] = RefField(
                    name=ref_name,
                    is_array=is_array,
                    required=rc.required,
                    min_items=rc.min_items if is_array else None,
                    max_items=rc.max_items if is_array else None,
                    target_collection=target_collection,
                    source_property=prop_name,
                    notes=["reverse reference materialized from inverse restriction/cardinality"],
                )

    return specs


def collection_spec_to_validator(
        spec: CollectionSpec,
        *,
        allow_additional_properties: bool = True,
) -> Dict[str, Any]:
    root: Dict[str, Any] = {
        "bsonType": "object",
        "properties": {},
        "additionalProperties": allow_additional_properties,
    }

    # scalar fields
    for field in spec.fields.values():
        set_schema_at_path(
            root,
            field.name,
            {"bsonType": field.bson_type},
            required=field.required,
        )

    # refs
    for ref in spec.refs.values():
        if ref.is_array:
            schema: Dict[str, Any] = {
                "bsonType": "array",
                "items": {"bsonType": "objectId"},
            }
            if ref.min_items is not None:
                schema["minItems"] = ref.min_items
            if ref.max_items is not None:
                schema["maxItems"] = ref.max_items
        else:
            schema = {"bsonType": "objectId"}

        set_schema_at_path(root, ref.name, schema, required=ref.required)

    prune_empty_required(root)
    return {"$jsonSchema": root}


def build_mongodb_validators_from_ontology_blueprint(
        ontology_bp: Dict[str, Any],
        *,
        allow_additional_properties: bool = True,
) -> Dict[str, Dict[str, Any]]:
    specs = build_collection_specs(ontology_bp)
    out: Dict[str, Dict[str, Any]] = {}
    for col_name in sorted(specs.keys()):
        out[col_name] = collection_spec_to_validator(
            specs[col_name],
            allow_additional_properties=allow_additional_properties,
        )
    return out


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build MongoDB validators directly from ontology_blueprint.json"
    )
    parser.add_argument(
        "ontology_blueprint",
        nargs="?",
        default="ontology_blueprint.json",
        help="Path to ontology_blueprint.json",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="mongo_validators.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Set additionalProperties=false in generated validators",
    )
    args = parser.parse_args()

    src = Path(args.ontology_blueprint)
    if not src.exists():
        raise FileNotFoundError(f"Ontology blueprint not found: {src}")

    ontology_bp = json.loads(src.read_text(encoding="utf-8"))
    validators = build_mongodb_validators_from_ontology_blueprint(
        ontology_bp,
        allow_additional_properties=not args.strict,
    )

    out_path = Path(args.out)
    out_path.write_text(json.dumps(validators, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote validators to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
