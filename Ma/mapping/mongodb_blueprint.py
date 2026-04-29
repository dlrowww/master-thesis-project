from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# =========================
# Dataclasses
# =========================

@dataclass
class MongoFieldBlueprint:
    name: str
    source_property: str
    bson_type: str
    required: bool = False
    is_array: bool = False
    source_domain: Optional[str] = None
    description: Optional[str] = None
    iri: Optional[str] = None
    inherited_from: Optional[str] = None
    validation_layer: str = "validator"


@dataclass
class MongoReferenceBlueprint:
    name: str
    logical_name: str
    source_property: str
    target_collection: Optional[str]
    target_class: Optional[str]
    target_classes: List[str] = field(default_factory=list)
    is_array: bool = False
    required: bool = False
    min_items: Optional[int] = None
    max_items: Optional[int] = None
    inverse_of: Optional[str] = None
    description: Optional[str] = None
    iri: Optional[str] = None
    inherited_from: Optional[str] = None
    validation_layer: str = "service"
    service_rules: List[str] = field(default_factory=list)


@dataclass
class CollectionSeedBlueprint:
    source_individual: str
    values: Dict[str, Any]
    description: Optional[str] = None


@dataclass
class CollectionConstraintBlueprint:
    kind: str
    applies_to: str
    details: Dict[str, Any] = field(default_factory=dict)
    validation_layer: str = "service"


@dataclass
class MongoCollectionBlueprint:
    name: str
    source_class: str
    materialization: str  # collection | vocabulary_collection
    description: Optional[str] = None
    iri: Optional[str] = None
    is_abstract: bool = False
    parents: List[str] = field(default_factory=list)
    subclasses: List[str] = field(default_factory=list)
    subtype_field: Optional[str] = "_ontology_type"
    subtype_values: List[str] = field(default_factory=list)
    fields: List[MongoFieldBlueprint] = field(default_factory=list)
    references: List[MongoReferenceBlueprint] = field(default_factory=list)
    collection_constraints: List[CollectionConstraintBlueprint] = field(default_factory=list)
    seed_documents: List[CollectionSeedBlueprint] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class MongoDBBlueprint:
    source_ontology_iri: Optional[str] = None
    source_blueprint_path: Optional[str] = None
    collections: Dict[str, MongoCollectionBlueprint] = field(default_factory=dict)
    abstract_classes: List[str] = field(default_factory=list)
    disjoint_groups: List[List[str]] = field(default_factory=list)
    subclass_axioms: List[Dict[str, str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# =========================
# Naming helpers
# =========================


def _to_snake(name: str) -> str:
    out: List[str] = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0 and (
                name[i - 1].islower() or (i + 1 < len(name) and name[i + 1].islower())
        ):
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


SPECIAL_NORMALIZATION = {
    "Apperance": "appearance",
}


def normalize_name(name: str) -> str:
    return SPECIAL_NORMALIZATION.get(name, _to_snake(name))


VOCAB_COLLECTION_OVERRIDES = {
    "Sex": "sexes",
    "Channel": "channels",
    "LifeActivity": "life_activities",
    "Modality": "modalities",
}


def pluralize(name: str) -> str:
    if name.endswith("s"):
        return name + "es"
    if name.endswith("y") and len(name) > 1 and name[-2] not in "aeiou":
        return name[:-1] + "ies"
    return name + "s"


# =========================
# JSON helpers
# =========================


def load_ontology_blueprint(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# =========================
# Annotation helpers
# =========================


def _annotation_values(obj: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    anns = obj.get("annotations", {}) or {}
    return anns.get(key, []) or []


def _first_annotation_text(obj: Dict[str, Any], key: str) -> Optional[str]:
    vals = _annotation_values(obj, key)
    if not vals:
        return None
    first = vals[0]
    return first.get("value")


def _is_abstract_class(cls_obj: Dict[str, Any]) -> bool:
    for ann in _annotation_values(cls_obj, "conceptType"):
        if str(ann.get("value", "")).lower() == "abstract":
            return True
    return False


# =========================
# Ontology helpers
# =========================


def _build_parent_child_maps(axioms: Dict[str, Any]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
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


def _transitive_closure(seed: str, relation: Dict[str, Set[str]]) -> List[str]:
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


def _effective_classes_for_domain(domain: str, classes: Dict[str, Any], children: Dict[str, Set[str]]) -> List[str]:
    descendants = _transitive_closure(domain, children)
    concrete_descendants = [c for c in descendants if c in classes and not _is_abstract_class(classes[c])]
    if domain in classes and not _is_abstract_class(classes[domain]):
        concrete_descendants.insert(0, domain)
    # unique preserve order
    out: List[str] = []
    seen: Set[str] = set()
    for c in concrete_descendants:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _all_ancestors(cls_name: str, parents: Dict[str, Set[str]]) -> List[str]:
    return _transitive_closure(cls_name, parents)


def _class_description(cls_obj: Dict[str, Any]) -> Optional[str]:
    return _first_annotation_text(cls_obj, "rdfs:comment")


def _property_description(prop_obj: Dict[str, Any]) -> Optional[str]:
    return _first_annotation_text(prop_obj, "rdfs:comment")


# =========================
# Restriction resolution
# =========================


@dataclass
class ResolvedConstraint:
    min_items: Optional[int] = None
    max_items: Optional[int] = None
    required: bool = False
    allowed_target_classes: Set[str] = field(default_factory=set)
    service_rules: List[str] = field(default_factory=list)


def _merge_min(current: Optional[int], new: Optional[int]) -> Optional[int]:
    if new is None:
        return current
    if current is None:
        return new
    return max(current, new)


def _merge_max(current: Optional[int], new: Optional[int]) -> Optional[int]:
    if new is None:
        return current
    if current is None:
        return new
    return min(current, new)


def _resolve_constraints_for_property(
        class_name: str,
        prop_name: str,
        prop_obj: Dict[str, Any],
        ontology_bp: Dict[str, Any],
        parents_map: Dict[str, Set[str]],
) -> ResolvedConstraint:
    rc = ResolvedConstraint()
    if prop_obj.get("is_functional"):
        rc.max_items = 1

    # explicit forward cardinality already extracted on the property
    cards = prop_obj.get("forward_cardinality_by_domain", {}) or {}
    for ancestor in [class_name] + _all_ancestors(class_name, parents_map):
        card = cards.get(ancestor)
        if card:
            rc.min_items = _merge_min(rc.min_items, card.get("min"))
            rc.max_items = _merge_max(rc.max_items, card.get("max"))

    # restrictions from axioms (including inherited ones)
    all_subjects = [class_name] + _all_ancestors(class_name, parents_map)
    for r in ontology_bp.get("axioms", {}).get("restriction_axioms", []):
        if r.get("subject_class") not in all_subjects:
            continue
        if r.get("property_name") != prop_name:
            continue
        if r.get("inverse"):
            continue

        rtype = r.get("restriction_type")
        if rtype in {"object_some_values_from", "data_some_values_from"}:
            rc.required = True
            rc.min_items = _merge_min(rc.min_items, 1)
            filler = r.get("filler")
            if filler:
                rc.allowed_target_classes.add(filler)
        elif rtype == "object_exact_cardinality":
            n = r.get("cardinality")
            rc.required = True if n and n >= 1 else rc.required
            rc.min_items = _merge_min(rc.min_items, n)
            rc.max_items = _merge_max(rc.max_items, n)
            filler = r.get("filler")
            if filler:
                rc.allowed_target_classes.add(filler)
        elif rtype == "object_min_cardinality":
            n = r.get("cardinality")
            if n is not None and n >= 1:
                rc.required = True
            rc.min_items = _merge_min(rc.min_items, n)
            filler = r.get("filler")
            if filler:
                rc.allowed_target_classes.add(filler)
        elif rtype == "object_max_cardinality":
            rc.max_items = _merge_max(rc.max_items, r.get("cardinality"))
            filler = r.get("filler")
            if filler:
                rc.allowed_target_classes.add(filler)
        elif rtype == "object_all_values_from":
            filler = r.get("filler")
            if filler:
                rc.allowed_target_classes.add(filler)
                rc.service_rules.append(f"ONLY target type(s) allowed: {filler}")

    # if max_items is 1, relation is single-valued
    return rc


def _resolve_inverse_rules(
        target_class: str,
        prop_name: str,
        prop_obj: Dict[str, Any],
        ontology_bp: Dict[str, Any],
        parents_map: Dict[str, Set[str]],
) -> List[str]:
    rules: List[str] = []
    inv_cards = prop_obj.get("inverse_cardinality_by_domain", {}) or {}
    all_subjects = [target_class] + _all_ancestors(target_class, parents_map)
    for cls in all_subjects:
        if cls in inv_cards:
            card = inv_cards[cls]
            rules.append(
                f"INVERSE cardinality on target class {cls}: min={card.get('min')} max={card.get('max')}"
            )

    for r in ontology_bp.get("axioms", {}).get("restriction_axioms", []):
        if r.get("subject_class") not in all_subjects:
            continue
        if r.get("property_name") != prop_name:
            continue
        if not r.get("inverse"):
            continue
        rtype = r.get("restriction_type")
        if rtype == "object_all_values_from" and r.get("filler"):
            rules.append(f"INVERSE ONLY allowed source type: {r.get('filler')}")
        if rtype in {"object_min_cardinality", "object_exact_cardinality", "object_max_cardinality"}:
            rules.append(
                f"INVERSE {rtype}: cardinality={r.get('cardinality')} on target class {r.get('subject_class')}"
            )
    return rules


# =========================
# Core builder
# =========================


def build_mongodb_blueprint(
        ontology_bp: Dict[str, Any],
        *,
        source_blueprint_path: Optional[str] = None,
) -> MongoDBBlueprint:
    classes: Dict[str, Any] = ontology_bp.get("classes", {}) or {}
    data_properties: Dict[str, Any] = ontology_bp.get("data_properties", {}) or {}
    object_properties: Dict[str, Any] = ontology_bp.get("object_properties", {}) or {}
    named_individuals: Dict[str, Any] = ontology_bp.get("named_individuals", {}) or {}
    axioms: Dict[str, Any] = ontology_bp.get("axioms", {}) or {}

    parents_map, children_map = _build_parent_child_maps(axioms)

    abstract_classes = sorted([name for name, obj in classes.items() if _is_abstract_class(obj)])

    named_individuals_by_type: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for ind_name, ind in named_individuals.items():
        for asserted_type in ind.get("asserted_types", []):
            named_individuals_by_type.setdefault(asserted_type, []).append((ind_name, ind))

    def materialization_for_class(class_name: str, cls_obj: Dict[str, Any]) -> Optional[str]:
        if _is_abstract_class(cls_obj):
            return None
        if class_name in named_individuals_by_type:
            return "vocabulary_collection"
        return "collection"

    class_to_collection: Dict[str, str] = {}
    collections: Dict[str, MongoCollectionBlueprint] = {}

    # 1) materialize classes
    for class_name, cls_obj in sorted(classes.items()):
        materialization = materialization_for_class(class_name, cls_obj)
        if materialization is None:
            continue

        if materialization == "vocabulary_collection":
            col_name = VOCAB_COLLECTION_OVERRIDES.get(class_name, pluralize(normalize_name(class_name)))
        else:
            col_name = pluralize(normalize_name(class_name))

        class_to_collection[class_name] = col_name
        collections[col_name] = MongoCollectionBlueprint(
            name=col_name,
            source_class=class_name,
            materialization=materialization,
            description=_class_description(cls_obj),
            iri=cls_obj.get("iri"),
            is_abstract=False,
            parents=sorted(parents_map.get(class_name, set())),
            subclasses=sorted(children_map.get(class_name, set())),
            subtype_values=sorted(children_map.get(class_name, set())),
        )

    # 2) seed documents for controlled individuals
    for class_name, individuals in sorted(named_individuals_by_type.items()):
        col_name = class_to_collection.get(class_name)
        if not col_name:
            continue
        col = collections[col_name]
        for ind_name, ind in sorted(individuals, key=lambda x: x[0]):
            col.seed_documents.append(
                CollectionSeedBlueprint(
                    source_individual=ind_name,
                    values={
                        "name": ind_name,
                        "iri": ind.get("iri"),
                        "label": _first_annotation_text(ind, "rdfs:comment") or ind_name,
                    },
                    description=_first_annotation_text(ind, "rdfs:comment"),
                )
            )
        col.notes.append("Controlled vocabulary collection seeded from named individuals.")

    # helper for adding fields/refs without duplicates
    def add_field(col: MongoCollectionBlueprint, field_bp: MongoFieldBlueprint) -> None:
        if all(f.name != field_bp.name for f in col.fields):
            col.fields.append(field_bp)

    def add_ref(col: MongoCollectionBlueprint, ref_bp: MongoReferenceBlueprint) -> None:
        if all(r.name != ref_bp.name for r in col.references):
            col.references.append(ref_bp)

    # 3) data properties -> fields (inherit to concrete subclasses)
    for prop_name, dp in sorted(data_properties.items()):
        domains = dp.get("domains", []) or []
        ranges = dp.get("ranges", []) or []
        bson_type = _bson_type_for_range(ranges[0] if ranges else None)
        description = _property_description(dp)
        field_name = normalize_name(prop_name)
        for domain in domains:
            effective_classes = _effective_classes_for_domain(domain, classes, children_map)
            for concrete_cls in effective_classes:
                col_name = class_to_collection.get(concrete_cls)
                if not col_name:
                    continue
                inherited_from = None if concrete_cls == domain else domain
                required = _data_property_is_required(concrete_cls, prop_name, ontology_bp, parents_map)
                add_field(
                    collections[col_name],
                    MongoFieldBlueprint(
                        name=field_name,
                        source_property=prop_name,
                        bson_type=bson_type,
                        required=required,
                        is_array=not bool(dp.get("is_functional", False)) and False,
                        source_domain=domain,
                        description=description,
                        iri=dp.get("iri"),
                        inherited_from=inherited_from,
                        validation_layer="validator",
                    ),
                )

    # global owl:Thing data properties -> all concrete collections
    for prop_name, dp in sorted(data_properties.items()):
        domains = dp.get("domains", []) or []
        if "owl:Thing" not in domains:
            continue
        ranges = dp.get("ranges", []) or []
        bson_type = _bson_type_for_range(ranges[0] if ranges else None)
        description = _property_description(dp)
        field_name = normalize_name(prop_name)
        for class_name, col_name in class_to_collection.items():
            add_field(
                collections[col_name],
                MongoFieldBlueprint(
                    name=field_name,
                    source_property=prop_name,
                    bson_type=bson_type,
                    required=_data_property_is_required(class_name, prop_name, ontology_bp, parents_map),
                    is_array=False,
                    source_domain="owl:Thing",
                    description=description,
                    iri=dp.get("iri"),
                    inherited_from="owl:Thing",
                    validation_layer="validator",
                ),
            )

    # 4) object properties -> references (inherit to concrete subclasses)
    for prop_name, op in sorted(object_properties.items()):
        domains = op.get("domains", []) or []
        ranges = op.get("ranges", []) or []
        if not domains:
            continue
        logical_name = normalize_name(prop_name)
        description = _property_description(op)
        for domain in domains:
            effective_classes = _effective_classes_for_domain(domain, classes, children_map)
            for concrete_cls in effective_classes:
                source_col = collections.get(class_to_collection.get(concrete_cls, ""))
                if not source_col:
                    continue
                inherited_from = None if concrete_cls == domain else domain
                resolved = _resolve_constraints_for_property(concrete_cls, prop_name, op, ontology_bp, parents_map)
                is_array = not (resolved.max_items == 1 or op.get("is_functional"))

                effective_target_classes = _resolve_target_classes(ranges, resolved.allowed_target_classes, classes,
                                                                   children_map)
                # pick a primary collection target when possible
                target_collection = None
                target_class = None
                if len(effective_target_classes) == 1:
                    target_class = effective_target_classes[0]
                    target_collection = class_to_collection.get(target_class)
                elif ranges:
                    target_class = ranges[0]
                    if target_class in class_to_collection:
                        target_collection = class_to_collection[target_class]

                service_rules = list(resolved.service_rules)
                for tc in effective_target_classes:
                    service_rules.extend(_resolve_inverse_rules(tc, prop_name, op, ontology_bp, parents_map))
                # de-duplicate while preserving order
                dedup_rules: List[str] = []
                seen: Set[str] = set()
                for rule in service_rules:
                    if rule not in seen:
                        seen.add(rule)
                        dedup_rules.append(rule)

                validation_layer = "validator"
                if dedup_rules or op.get("inverse_cardinality_by_domain"):
                    validation_layer = "service"
                if resolved.max_items is not None and resolved.min_items is not None and not is_array and not dedup_rules:
                    validation_layer = "validator"

                add_ref(
                    source_col,
                    MongoReferenceBlueprint(
                        name=f"{logical_name}_ids" if is_array else f"{logical_name}_id",
                        logical_name=logical_name,
                        source_property=prop_name,
                        target_collection=target_collection,
                        target_class=target_class,
                        target_classes=effective_target_classes,
                        is_array=is_array,
                        required=resolved.required,
                        min_items=resolved.min_items,
                        max_items=resolved.max_items,
                        inverse_of=op.get("inverse_of"),
                        description=description,
                        iri=op.get("iri"),
                        inherited_from=inherited_from,
                        validation_layer=validation_layer,
                        service_rules=dedup_rules,
                    ),
                )

    # 5) collection constraints from subclass/disjoint/equivalent restrictions
    disjoint_groups = [ax.get("class_names", []) for ax in axioms.get("disjoint_class_axioms", [])]
    subclass_axioms = list(axioms.get("subclass_axioms", []))

    # add subtype / disjoint notes
    for col in collections.values():
        if col.parents:
            col.notes.append(f"Subclass of: {', '.join(col.parents)}")
        if col.subclasses:
            col.notes.append(f"Direct subclasses: {', '.join(col.subclasses)}")
        for group in disjoint_groups:
            if col.source_class in group:
                col.collection_constraints.append(
                    CollectionConstraintBlueprint(
                        kind="disjoint_membership",
                        applies_to=col.source_class,
                        details={"group": group},
                        validation_layer="service",
                    )
                )

    # 6) add abstract notes for references targeting abstract concepts only
    for col in collections.values():
        for ref in col.references:
            if ref.target_class in abstract_classes:
                ref.service_rules.append(
                    f"Target class {ref.target_class} is abstract; bind to a concrete subclass or controlled vocabulary."
                )
                ref.validation_layer = "service"

    bp = MongoDBBlueprint(
        source_ontology_iri=(ontology_bp.get("ontology_info", {}) or {}).get("ontology_iri"),
        source_blueprint_path=str(source_blueprint_path) if source_blueprint_path else None,
        collections=collections,
        abstract_classes=abstract_classes,
        disjoint_groups=disjoint_groups,
        subclass_axioms=subclass_axioms,
        notes=[
            "Object properties are mapped as references (reference-only strategy).",
            "Class/data-property annotations are preserved as schema metadata, not business fields.",
            "Cross-document cardinality, inverse constraints, disjointness, and value restrictions are marked for service-layer validation.",
            "Named individuals are materialized as seed data for controlled-vocabulary collections when their asserted type is concrete.",
            "Abstract concepts are kept in blueprint metadata and do not materialize as collections by default.",
        ],
    )
    return bp


# =========================
# Utility functions used by builder
# =========================


def _bson_type_for_range(dt: Optional[str]) -> str:
    if not dt:
        return "string"
    lower = dt.lower()
    if "int" in lower:
        return "int"
    if "float" in lower or "double" in lower or "decimal" in lower:
        return "double"
    if "bool" in lower:
        return "bool"
    if "date" in lower or "time" in lower:
        return "date"
    return "string"


def _data_property_is_required(
        concrete_cls: str,
        prop_name: str,
        ontology_bp: Dict[str, Any],
        parents_map: Dict[str, Set[str]],
) -> bool:
    all_subjects = [concrete_cls] + _all_ancestors(concrete_cls, parents_map)
    for r in ontology_bp.get("axioms", {}).get("restriction_axioms", []):
        if r.get("subject_class") in all_subjects and r.get("property_name") == prop_name:
            if r.get("restriction_type") == "data_some_values_from":
                return True
    return False


def _resolve_target_classes(
        declared_ranges: List[str],
        allowed_target_classes: Set[str],
        classes: Dict[str, Any],
        children_map: Dict[str, Set[str]],
) -> List[str]:
    targets = set(allowed_target_classes) if allowed_target_classes else set(declared_ranges)
    expanded: List[str] = []
    for t in sorted(targets):
        if t in classes and _is_abstract_class(classes[t]):
            expanded.extend(_effective_classes_for_domain(t, classes, children_map))
        else:
            expanded.append(t)
    # unique preserve order
    out: List[str] = []
    seen: Set[str] = set()
    for t in expanded:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# =========================
# Serialization / pretty print
# =========================


def mongodb_blueprint_to_dict(bp: MongoDBBlueprint) -> Dict[str, Any]:
    return asdict(bp)


def mongodb_blueprint_to_json(bp: MongoDBBlueprint) -> str:
    return json.dumps(mongodb_blueprint_to_dict(bp), indent=2, ensure_ascii=False)


def pretty_print_mongodb_blueprint(bp: MongoDBBlueprint) -> str:
    lines: List[str] = []
    lines.append("=== MongoDB Blueprint Summary ===")
    lines.append(f"Source ontology IRI: {bp.source_ontology_iri}")
    lines.append(f"Collections: {len(bp.collections)}")
    lines.append(f"Abstract classes: {len(bp.abstract_classes)}")
    lines.append("")
    for col_name in sorted(bp.collections.keys()):
        col = bp.collections[col_name]
        lines.append(f"- Collection: {col.name}  (from Class: {col.source_class})")
        lines.append(f"  Materialization: {col.materialization}")
        if col.description:
            lines.append(f"  Description: {col.description}")
        if col.parents:
            lines.append(f"  Parents: {', '.join(col.parents)}")
        if col.subclasses:
            lines.append(f"  Subclasses: {', '.join(col.subclasses)}")
        lines.append("  Fields:")
        if col.fields:
            for f in col.fields:
                inherited = f" [inherited from {f.inherited_from}]" if f.inherited_from else ""
                req = " required" if f.required else ""
                lines.append(f"    - {f.name}: {f.bson_type}{req}{inherited}")
        else:
            lines.append("    (none)")
        lines.append("  References:")
        if col.references:
            for r in col.references:
                inherited = f" [inherited from {r.inherited_from}]" if r.inherited_from else ""
                card = f" min={r.min_items} max={r.max_items}" if (
                            r.min_items is not None or r.max_items is not None) else ""
                lines.append(
                    f"    - {r.name} -> {r.target_collection or r.target_class}  "
                    f"[array={r.is_array}; required={r.required}; layer={r.validation_layer}{card}]{inherited}"
                )
                for rule in r.service_rules:
                    lines.append(f"        rule: {rule}")
        else:
            lines.append("    (none)")
        if col.seed_documents:
            lines.append("  Seed documents:")
            for s in col.seed_documents[:5]:
                lines.append(f"    - {s.source_individual}: {s.values}")
            if len(col.seed_documents) > 5:
                lines.append(f"    ... ({len(col.seed_documents) - 5} more)")
        if col.collection_constraints:
            lines.append("  Collection constraints:")
            for c in col.collection_constraints:
                lines.append(f"    - {c.kind}: {c.details} [{c.validation_layer}]")
        if col.notes:
            lines.append("  Notes:")
            for note in col.notes:
                lines.append(f"    - {note}")
        lines.append("")
    if bp.notes:
        lines.append("Blueprint notes:")
        for note in bp.notes:
            lines.append(f"- {note}")
    return "\n".join(lines)
