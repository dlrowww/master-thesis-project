from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# -----------------------------
# Generic helpers
# -----------------------------

VOCAB_COLLECTION_OVERRIDES: Dict[str, str] = {
    "Sex": "sexes",
}


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def camel_to_snake(name: str) -> str:
    if not name:
        return name
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.replace("-", "_").lower()


def normalize_name(name: str) -> str:
    return camel_to_snake(name)


def pluralize(name: str) -> str:
    if name.endswith("y") and len(name) > 1 and name[-2] not in "aeiou":
        return name[:-1] + "ies"
    if name.endswith(("s", "x", "z", "ch", "sh")):
        return name + "es"
    return name + "s"


def collection_name_for_class(class_name: str, vocabulary_types: Set[str]) -> str:
    if class_name in vocabulary_types:
        return VOCAB_COLLECTION_OVERRIDES.get(class_name, pluralize(normalize_name(class_name)))
    return pluralize(normalize_name(class_name))


def local_name(iri_or_name: str) -> str:
    if not iri_or_name:
        return iri_or_name
    if "#" in iri_or_name:
        return iri_or_name.rsplit("#", 1)[-1]
    if "/" in iri_or_name:
        return iri_or_name.rstrip("/").rsplit("/", 1)[-1]
    return iri_or_name


# -----------------------------
# Registry-facing specs
# -----------------------------

@dataclass(frozen=True)
class ScalarFieldSpec:
    name: str
    bson_type: str
    required: bool = False
    is_array: bool = False
    min_items: Optional[int] = None
    max_items: Optional[int] = None
    source_property: Optional[str] = None
    source_classes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ReferenceFieldSpec:
    name: str
    source_property: Optional[str]
    target_classes: Tuple[str, ...]
    target_collection: Optional[str]
    target_collections: Tuple[str, ...] = ()
    required: bool = False
    is_array: bool = False
    min_items: Optional[int] = None
    max_items: Optional[int] = None
    inverse_of: Optional[str] = None
    source_classes: Tuple[str, ...] = ()
    on_delete: str = "restrict"


@dataclass
class CollectionSpec:
    name: str
    source_classes: Set[str] = field(default_factory=set)
    required_fields: Set[str] = field(default_factory=set)
    scalar_fields: Dict[str, ScalarFieldSpec] = field(default_factory=dict)
    reference_fields: Dict[str, ReferenceFieldSpec] = field(default_factory=dict)
    metadata_fields: Set[str] = field(default_factory=set)


class BlueprintRegistry:
    """
    Unified registry over ontology_blueprint(.json) and mongo_validators.json.

    Supported blueprint variants:
      - schema-only blueprint
      - schema+data blueprint produced from a mixed OWL file
      - enriched blueprint produced by augmenting a base schema blueprint

    Design principles:
      - schema-level access is always available
      - instance assertions are indexed only if present
      - validators are optional, but recommended for field-level DAL checks
    """

    def __init__(self, ontology_blueprint: Dict[str, Any], validators: Optional[Dict[str, Any]] = None):
        self.ontology_blueprint = ontology_blueprint or {}
        self.validators = validators or {}

        # raw sections
        self.classes_by_name: Dict[str, Dict[str, Any]] = dict(self.ontology_blueprint.get("classes") or {})
        self.data_properties_by_name: Dict[str, Dict[str, Any]] = dict(
            self.ontology_blueprint.get("data_properties") or {})
        self.object_properties_by_name: Dict[str, Dict[str, Any]] = dict(
            self.ontology_blueprint.get("object_properties") or {})
        self.named_individuals_by_name: Dict[str, Dict[str, Any]] = dict(
            self.ontology_blueprint.get("named_individuals") or {})
        self.axioms: Dict[str, Any] = dict(self.ontology_blueprint.get("axioms") or {})

        # hierarchy / abstractness
        self.parents_map: Dict[str, Set[str]] = {}
        self.children_map: Dict[str, Set[str]] = {}
        self.abstract_classes: Set[str] = set()
        self.disjoint_groups: List[Set[str]] = []
        self.union_parent_to_members: Dict[str, Set[str]] = {}

        # class / collection mapping
        self.named_individuals_by_type: Dict[str, List[str]] = {}
        self.vocabulary_types: Set[str] = set()
        self.class_to_collection: Dict[str, str] = {}
        self.collection_to_classes: Dict[str, Set[str]] = {}

        # property / field mapping
        self.collection_specs: Dict[str, CollectionSpec] = {}
        self.property_to_field: Dict[Tuple[str, str], str] = {}
        self.field_to_source_property: Dict[Tuple[str, str], Optional[str]] = {}
        self.object_property_target_classes: Dict[Tuple[str, str], Tuple[str, ...]] = {}
        self.data_property_domains: Dict[str, Tuple[str, ...]] = {}
        self.object_property_domains: Dict[str, Tuple[str, ...]] = {}
        self.collection_field_origin: Dict[Tuple[str, str], Dict[str, Any]] = {}

        # reverse reference support for delete validation
        self.incoming_reference_index: Dict[str, List[Dict[str, Any]]] = {}

        # individual indexes (optional enhancement)
        self.individual_type_map: Dict[str, Tuple[str, ...]] = {}
        self.individual_data_assertions: Dict[str, Dict[str, Any]] = {}
        self.individual_object_assertions: Dict[str, Dict[str, Any]] = {}

        self._build_indexes()

    # -----------------------------
    # Build pipeline
    # -----------------------------

    def _build_indexes(self) -> None:
        self._build_hierarchy_indexes()
        self._build_named_individual_indexes()
        self._build_class_collection_indexes()
        self._build_property_indexes()
        self._build_collection_specs_from_validators()
        self._supplement_field_property_mapping()
        self._build_incoming_reference_index()

    def _build_hierarchy_indexes(self) -> None:
        for cls in self.classes_by_name:
            self.parents_map.setdefault(cls, set())
            self.children_map.setdefault(cls, set())

        for ax in self.axioms.get("subclass_axioms", []) or []:
            sub = ax.get("subclass")
            sup = ax.get("superclass")
            if not sub or not sup:
                continue
            self.parents_map.setdefault(sub, set()).add(sup)
            self.children_map.setdefault(sup, set()).add(sub)
            self.parents_map.setdefault(sup, set())
            self.children_map.setdefault(sub, set())

        for cls_name, cls_obj in self.classes_by_name.items():
            anns = cls_obj.get("annotations") or {}
            concept_type_values = anns.get("conceptType") or []
            if any(self._annotation_text(v).strip().lower() == "abstract" for v in concept_type_values):
                self.abstract_classes.add(cls_name)

        for ax in self.axioms.get("disjoint_class_axioms", []) or []:
            names = ax.get("class_names") or []
            if names:
                self.disjoint_groups.append(set(names))

        for ax in self.axioms.get("equivalent_class_axioms", []) or []:
            lhs = ax.get("class_name") or ax.get("class") or ax.get("lhs")
            union_members = ax.get("union_of") or ax.get("members") or ax.get("class_names")
            if lhs and union_members:
                self.union_parent_to_members.setdefault(lhs, set()).update([m for m in union_members if m])

    def _build_named_individual_indexes(self) -> None:
        for ind_name, ind in self.named_individuals_by_name.items():
            asserted = tuple(ind.get("asserted_types") or ())
            self.individual_type_map[ind_name] = asserted
            self.individual_data_assertions[ind_name] = dict(ind.get("data_assertions") or {})
            self.individual_object_assertions[ind_name] = dict(ind.get("object_assertions") or {})
            for t in asserted:
                self.named_individuals_by_type.setdefault(t, []).append(ind_name)

        self.vocabulary_types = {
            cls_name
            for cls_name, inds in self.named_individuals_by_type.items()
            if inds and cls_name in self.classes_by_name and not self.is_abstract_class(cls_name)
        }

    def _build_class_collection_indexes(self) -> None:
        for class_name in sorted(self.classes_by_name.keys()):
            if self.is_abstract_class(class_name):
                continue
            col_name = collection_name_for_class(class_name, self.vocabulary_types)
            self.class_to_collection[class_name] = col_name
            self.collection_to_classes.setdefault(col_name, set()).add(class_name)
            self.collection_specs.setdefault(col_name, CollectionSpec(name=col_name, source_classes={class_name}))

    def _build_property_indexes(self) -> None:
        for prop_name, dp in self.data_properties_by_name.items():
            domains = tuple(dp.get("domains") or ())
            self.data_property_domains[prop_name] = domains

        for prop_name, op in self.object_properties_by_name.items():
            domains = tuple(op.get("domains") or ())
            self.object_property_domains[prop_name] = domains
            self.object_property_target_classes[("__global__", prop_name)] = tuple(op.get("ranges") or ())

    def _build_collection_specs_from_validators(self) -> None:
        if not self.validators:
            return

        for coll_name, validator_doc in self.validators.items():
            root = (validator_doc or {}).get("$jsonSchema") or {}
            properties = dict(root.get("properties") or {})
            required = set(root.get("required") or [])

            spec = self.collection_specs.setdefault(
                coll_name,
                CollectionSpec(name=coll_name, source_classes=set(self.collection_to_classes.get(coll_name, set()))),
            )
            spec.required_fields.update(required)

            for field_name, schema in properties.items():
                if field_name == "metadata":
                    nested = (schema.get("properties") or {})
                    for child_name in nested.keys():
                        spec.metadata_fields.add(f"metadata.{child_name}")
                    continue

                field_spec = self._schema_to_field_spec(
                    coll_name=coll_name,
                    field_name=field_name,
                    schema=schema,
                    required=(field_name in required),
                )
                if isinstance(field_spec, ReferenceFieldSpec):
                    spec.reference_fields[field_name] = field_spec
                else:
                    spec.scalar_fields[field_name] = field_spec

    def _schema_to_field_spec(
            self,
            *,
            coll_name: str,
            field_name: str,
            schema: Dict[str, Any],
            required: bool,
    ) -> ScalarFieldSpec | ReferenceFieldSpec:
        bson_type = schema.get("bsonType")

        if bson_type == "objectId":
            src_prop = self._infer_source_property_from_field_name(field_name)
            targets = self._infer_target_classes_for_reference(coll_name, field_name, src_prop)
            target_collections = tuple(self._choose_target_collections(targets))
            target_collection = target_collections[0] if target_collections else None
            return ReferenceFieldSpec(
                name=field_name,
                source_property=src_prop,
                target_classes=tuple(targets),
                target_collection=target_collection,
                target_collections=target_collections,
                required=required,
                is_array=False,
                min_items=None,
                max_items=None,
                inverse_of=self._inverse_of(src_prop),
                source_classes=tuple(sorted(self.collection_to_classes.get(coll_name, set()))),
                on_delete=self._resolve_reference_delete_policy(
                    source_property=src_prop,
                    required=required,
                    is_array=False,
                    min_items=None,
                ),
            )

        if bson_type == "array" and ((schema.get("items") or {}).get("bsonType") == "objectId"):
            src_prop = self._infer_source_property_from_field_name(field_name)
            targets = self._infer_target_classes_for_reference(coll_name, field_name, src_prop)
            target_collections = tuple(self._choose_target_collections(targets))
            target_collection = target_collections[0] if target_collections else None
            min_items = schema.get("minItems")
            max_items = schema.get("maxItems")
            return ReferenceFieldSpec(
                name=field_name,
                source_property=src_prop,
                target_classes=tuple(targets),
                target_collection=target_collection,
                target_collections=target_collections,
                required=required,
                is_array=True,
                min_items=min_items,
                max_items=max_items,
                inverse_of=self._inverse_of(src_prop),
                source_classes=tuple(sorted(self.collection_to_classes.get(coll_name, set()))),
                on_delete=self._resolve_reference_delete_policy(
                    source_property=src_prop,
                    required=required,
                    is_array=True,
                    min_items=min_items,
                ),
            )

        src_prop = self._infer_source_property_from_field_name(field_name)
        return ScalarFieldSpec(
            name=field_name,
            bson_type=bson_type or "object",
            required=required,
            is_array=(bson_type == "array"),
            min_items=schema.get("minItems"),
            max_items=schema.get("maxItems"),
            source_property=src_prop,
            source_classes=tuple(sorted(self.collection_to_classes.get(coll_name, set()))),
        )

    def _supplement_field_property_mapping(self) -> None:
        for coll_name, spec in self.collection_specs.items():
            source_classes = sorted(self.collection_to_classes.get(coll_name, set()))
            if not source_classes:
                source_classes = sorted(spec.source_classes)

            for src_class in source_classes:
                for prop_name, dp in self.data_properties_by_name.items():
                    if self._property_applies_to_class(src_class, dp.get("domains") or []):
                        candidate = normalize_name(prop_name)
                        if candidate in spec.scalar_fields:
                            self.property_to_field[(src_class, prop_name)] = candidate
                            self.field_to_source_property[(coll_name, candidate)] = prop_name
                            self.collection_field_origin[(coll_name, candidate)] = {
                                "mongo_field": candidate,
                                "source_property": prop_name,
                                "field_kind": "scalar",
                                "declared_on": tuple(dp.get("domains") or ()),
                                "available_on": src_class,
                            }

                for prop_name, op in self.object_properties_by_name.items():
                    if self._property_applies_to_class(src_class, op.get("domains") or []):
                        candidate_single = f"{normalize_name(prop_name)}_id"
                        candidate_multi = f"{normalize_name(prop_name)}_ids"
                        if candidate_single in spec.reference_fields:
                            self.property_to_field[(src_class, prop_name)] = candidate_single
                            self.field_to_source_property[(coll_name, candidate_single)] = prop_name
                            self.collection_field_origin[(coll_name, candidate_single)] = {
                                "mongo_field": candidate_single,
                                "source_property": prop_name,
                                "field_kind": "reference",
                                "declared_on": tuple(op.get("domains") or ()),
                                "available_on": src_class,
                            }
                        if candidate_multi in spec.reference_fields:
                            self.property_to_field[(src_class, prop_name)] = candidate_multi
                            self.field_to_source_property[(coll_name, candidate_multi)] = prop_name
                            self.collection_field_origin[(coll_name, candidate_multi)] = {
                                "mongo_field": candidate_multi,
                                "source_property": prop_name,
                                "field_kind": "reference",
                                "declared_on": tuple(op.get("domains") or ()),
                                "available_on": src_class,
                            }

    def _build_incoming_reference_index(self) -> None:
        self.incoming_reference_index = {}
        for source_collection, spec in self.collection_specs.items():
            for field_name, ref_spec in spec.reference_fields.items():
                target_collections = list(ref_spec.target_collections or (
                    () if not ref_spec.target_collection else (ref_spec.target_collection,)))
                if not target_collections:
                    continue
                for target_collection in target_collections:
                    self.incoming_reference_index.setdefault(target_collection, []).append(
                        {
                            "source_collection": source_collection,
                            "source_field": field_name,
                            "source_property": ref_spec.source_property,
                            "is_array": ref_spec.is_array,
                            "required": ref_spec.required,
                            "target_classes": list(ref_spec.target_classes),
                            "on_delete": ref_spec.on_delete,
                        }
                    )

    # -----------------------------
    # Internal inference helpers
    # -----------------------------

    @staticmethod
    def _annotation_text(value: Any) -> str:
        if isinstance(value, dict):
            return str(value.get("value") or "")
        return str(value or "")

    def _all_ancestors(self, class_name: str) -> Set[str]:
        out: Set[str] = set()
        stack = list(self.parents_map.get(class_name, set()))
        while stack:
            cur = stack.pop()
            if cur in out:
                continue
            out.add(cur)
            stack.extend(self.parents_map.get(cur, set()))
        return out

    def _all_descendants(self, class_name: str) -> Set[str]:
        out: Set[str] = set()
        stack = list(self.children_map.get(class_name, set()))
        while stack:
            cur = stack.pop()
            if cur in out:
                continue
            out.add(cur)
            stack.extend(self.children_map.get(cur, set()))
        return out

    def _effective_concrete_classes_for_domain(self, domain: str) -> List[str]:
        if domain not in self.classes_by_name:
            return []
        if not self.is_abstract_class(domain):
            return [domain]
        descendants = [d for d in sorted(self._all_descendants(domain)) if not self.is_abstract_class(d)]
        return descendants

    def _property_applies_to_class(self, class_name: str, domains: Iterable[str]) -> bool:
        domain_set = set(domains or [])
        if not domain_set:
            return False
        if "owl:Thing" in domain_set:
            return True
        if class_name in domain_set:
            return True
        ancestors = self._all_ancestors(class_name)
        return any(d in ancestors for d in domain_set)

    def _infer_source_property_from_field_name(self, field_name: str) -> Optional[str]:
        base = field_name
        if base.endswith("_ids"):
            base = base[:-4]
        elif base.endswith("_id"):
            base = base[:-3]
        for prop_name in list(self.data_properties_by_name.keys()) + list(self.object_properties_by_name.keys()):
            if normalize_name(prop_name) == base:
                return prop_name
        return None

    def _infer_target_classes_for_reference(
            self, coll_name: str, field_name: str, src_prop: Optional[str]
    ) -> List[str]:
        source_classes = sorted(self.collection_to_classes.get(coll_name, set()))
        if src_prop and src_prop in self.object_properties_by_name:
            op = self.object_properties_by_name[src_prop]
            declared = list(op.get("ranges") or [])
            restriction_fillers = self._restriction_fillers_for_reference(source_classes, src_prop)
            targets: List[str] = []
            for t in declared + restriction_fillers:
                if not t:
                    continue
                if t in self.classes_by_name and self.is_abstract_class(t):
                    targets.extend(self._effective_concrete_classes_for_domain(t))
                else:
                    targets.append(t)
            out: List[str] = []
            seen: Set[str] = set()
            for t in targets:
                if t not in seen:
                    seen.add(t)
                    out.append(t)
            if out:
                return out

        if field_name.endswith("_id") or field_name.endswith("_ids"):
            base = field_name[:-3] if field_name.endswith("_id") else field_name[:-4]
            guessed_class = self._guess_class_from_collectionish_token(base)
            if guessed_class:
                return [guessed_class]
        return []

    def _restriction_fillers_for_reference(self, source_classes: List[str], prop_name: str) -> List[str]:
        fillers: List[str] = []
        subjects: Set[str] = set()
        for cls in source_classes:
            subjects.add(cls)
            subjects.update(self._all_ancestors(cls))

        for ax in self.axioms.get("restriction_axioms", []) or []:
            if ax.get("subject_class") not in subjects:
                continue
            if ax.get("property_name") != prop_name:
                continue
            if not str(ax.get("restriction_type") or "").startswith("object_"):
                continue
            filler = ax.get("filler")
            if filler:
                fillers.append(filler)
        return fillers

    def _choose_target_collections(self, target_classes: Iterable[str]) -> List[str]:
        out: List[str] = []
        seen: Set[str] = set()
        for target_class in target_classes:
            coll = self.class_to_collection.get(target_class)
            if coll and coll not in seen:
                seen.add(coll)
                out.append(coll)
        return out

    def _resolve_reference_delete_policy(
            self,
            *,
            source_property: Optional[str],
            required: bool,
            is_array: bool,
            min_items: Optional[int],
    ) -> str:
        explicit = self._explicit_delete_policy_for_property(source_property)
        if explicit:
            return explicit
        if required or (min_items is not None and min_items > 0):
            return "restrict"
        return "pull" if is_array else "unset"

    def _explicit_delete_policy_for_property(self, prop_name: Optional[str]) -> Optional[str]:
        if not prop_name:
            return None
        op = self.object_properties_by_name.get(prop_name) or {}
        for key in ("on_delete", "delete_policy"):
            raw = op.get(key)
            if isinstance(raw, str) and raw.strip().lower() in {"restrict", "unset", "pull", "cascade"}:
                return raw.strip().lower()

        anns = op.get("annotations") or {}
        for ann_key in ("onDelete", "deletePolicy", "delete_policy"):
            vals = anns.get(ann_key) or []
            for val in vals:
                text_val = self._annotation_text(val).strip().lower()
                if text_val in {"restrict", "unset", "pull", "cascade"}:
                    return text_val
        return None

    def _guess_class_from_collectionish_token(self, token: str) -> Optional[str]:
        for class_name, coll_name in self.class_to_collection.items():
            if coll_name == pluralize(token):
                return class_name
            if normalize_name(class_name) == token:
                return class_name
        return None

    def _inverse_of(self, prop_name: Optional[str]) -> Optional[str]:
        if not prop_name:
            return None
        return (self.object_properties_by_name.get(prop_name) or {}).get("inverse_of")

    # -----------------------------
    # Public query API
    # -----------------------------

    def class_exists(self, class_name: str) -> bool:
        return class_name in self.classes_by_name

    def individual_exists(self, individual_name: str) -> bool:
        return individual_name in self.named_individuals_by_name

    def get_individual_asserted_types(self, individual_name: str) -> List[str]:
        return list(self.individual_type_map.get(individual_name, ()))

    def get_individual_data_assertions(self, individual_name: str) -> Dict[str, Any]:
        return dict(self.individual_data_assertions.get(individual_name, {}))

    def get_individual_object_assertions(self, individual_name: str) -> Dict[str, Any]:
        return dict(self.individual_object_assertions.get(individual_name, {}))

    def is_abstract_class(self, class_name: str) -> bool:
        return class_name in self.abstract_classes

    def get_parent_classes(self, class_name: str, transitive: bool = False) -> List[str]:
        return sorted(self._all_ancestors(class_name) if transitive else self.parents_map.get(class_name, set()))

    def get_child_classes(self, class_name: str, transitive: bool = False) -> List[str]:
        return sorted(self._all_descendants(class_name) if transitive else self.children_map.get(class_name, set()))

    def is_subclass_of(self, child: str, parent: str) -> bool:
        if child == parent:
            return True
        return parent in self._all_ancestors(child)

    def is_type_compatible(self, actual_type: str, allowed_types: Iterable[str]) -> bool:
        for allowed in allowed_types or []:
            if actual_type == allowed or self.is_subclass_of(actual_type, allowed):
                return True
        return False

    def get_union_members(self, class_name: str) -> List[str]:
        return sorted(self.union_parent_to_members.get(class_name, set()))

    def is_union_parent(self, class_name: str) -> bool:
        return class_name in self.union_parent_to_members and bool(self.union_parent_to_members[class_name])

    def get_disjoint_groups_for_class(self, class_name: str) -> List[List[str]]:
        return [sorted(group) for group in self.disjoint_groups if class_name in group]

    def get_collection_for_class(self, class_name: str) -> Optional[str]:
        return self.class_to_collection.get(class_name)

    def get_classes_for_collection(self, collection_name: str) -> List[str]:
        return sorted(self.collection_to_classes.get(collection_name, set()))

    def get_primary_class_for_collection(self, collection_name: str) -> Optional[str]:
        classes = self.get_classes_for_collection(collection_name)
        return classes[0] if classes else None

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collection_specs

    def get_required_fields(self, collection_name: str) -> Set[str]:
        spec = self.collection_specs.get(collection_name)
        return set(spec.required_fields) if spec else set()

    def get_scalar_fields(self, collection_name: str) -> Dict[str, ScalarFieldSpec]:
        spec = self.collection_specs.get(collection_name)
        return dict(spec.scalar_fields) if spec else {}

    def get_reference_fields(self, collection_name: str) -> Dict[str, ReferenceFieldSpec]:
        spec = self.collection_specs.get(collection_name)
        return dict(spec.reference_fields) if spec else {}

    def get_metadata_fields(self, collection_name: str) -> Set[str]:
        spec = self.collection_specs.get(collection_name)
        return set(spec.metadata_fields) if spec else set()

    def field_exists(self, collection_name: str, field_name: str) -> bool:
        spec = self.collection_specs.get(collection_name)
        if not spec:
            return False
        return field_name in spec.scalar_fields or field_name in spec.reference_fields or field_name in spec.metadata_fields

    def is_field_allowed_for_class(self, class_name: str, field_name: str) -> bool:
        coll = self.get_collection_for_class(class_name)
        if coll and self.field_exists(coll, field_name):
            return True
        for ancestor in self.get_parent_classes(class_name, transitive=True):
            coll = self.get_collection_for_class(ancestor)
            if coll and self.field_exists(coll, field_name):
                return True
        return False

    def get_field_origin_for_class(self, class_name: str, field_name: str) -> Optional[Dict[str, Any]]:
        candidate_classes = [class_name] + self.get_parent_classes(class_name, transitive=True)
        for candidate in candidate_classes:
            coll = self.get_collection_for_class(candidate)
            if not coll:
                continue
            origin = self.collection_field_origin.get((coll, field_name))
            if origin:
                return dict(origin)
        return None

    def is_reference_field(self, collection_name: str, field_name: str) -> bool:
        spec = self.collection_specs.get(collection_name)
        return bool(spec and field_name in spec.reference_fields)

    def get_reference_spec(self, collection_name: str, field_name: str) -> Optional[ReferenceFieldSpec]:
        spec = self.collection_specs.get(collection_name)
        if not spec:
            return None
        return spec.reference_fields.get(field_name)

    def get_scalar_spec(self, collection_name: str, field_name: str) -> Optional[ScalarFieldSpec]:
        spec = self.collection_specs.get(collection_name)
        if not spec:
            return None
        return spec.scalar_fields.get(field_name)

    def get_target_classes(self, collection_name: str, field_name: str) -> List[str]:
        ref = self.get_reference_spec(collection_name, field_name)
        return list(ref.target_classes) if ref else []

    def get_target_collection(self, collection_name: str, field_name: str) -> Optional[str]:
        ref = self.get_reference_spec(collection_name, field_name)
        return ref.target_collection if ref else None

    def get_target_collections(self, collection_name: str, field_name: str) -> List[str]:
        ref = self.get_reference_spec(collection_name, field_name)
        return list(ref.target_collections) if ref else []

    def is_multi_reference(self, collection_name: str, field_name: str) -> bool:
        ref = self.get_reference_spec(collection_name, field_name)
        return bool(ref and ref.is_array)

    def get_delete_policy(self, collection_name: str, field_name: str) -> Optional[str]:
        ref = self.get_reference_spec(collection_name, field_name)
        return ref.on_delete if ref else None

    def get_mongo_field_for_property(self, class_name: str, property_name: str) -> Optional[str]:
        if (class_name, property_name) in self.property_to_field:
            return self.property_to_field[(class_name, property_name)]

        for ancestor in self.get_parent_classes(class_name, transitive=True):
            if (ancestor, property_name) in self.property_to_field:
                return self.property_to_field[(ancestor, property_name)]

        for child in self.get_child_classes(class_name, transitive=True):
            if (child, property_name) in self.property_to_field:
                return self.property_to_field[(child, property_name)]

        return None

    def get_allowed_data_properties_for_class(self, class_name: str) -> List[str]:
        return sorted(
            prop_name
            for prop_name, dp in self.data_properties_by_name.items()
            if self._property_applies_to_class(class_name, dp.get("domains") or [])
        )

    def get_allowed_object_properties_for_class(self, class_name: str) -> List[str]:
        return sorted(
            prop_name
            for prop_name, op in self.object_properties_by_name.items()
            if self._property_applies_to_class(class_name, op.get("domains") or [])
        )

    def get_allowed_fields_for_class(self, class_name: str) -> Set[str]:
        coll = self.get_collection_for_class(class_name)
        if not coll:
            return set()
        spec = self.collection_specs.get(coll)
        if not spec:
            return set()
        return set(spec.scalar_fields.keys()) | set(spec.reference_fields.keys())

    def is_vocabulary_type(self, class_name: str) -> bool:
        return class_name in self.vocabulary_types

    def get_vocabulary_individuals(self, class_name: str) -> List[str]:
        return list(self.named_individuals_by_type.get(class_name, []))

    def is_controlled_individual(self, individual_name: str) -> bool:
        asserted = self.get_individual_asserted_types(individual_name)
        return any(self.is_vocabulary_type(t) for t in asserted)

    def choose_primary_type(self, asserted_types: Iterable[str]) -> Optional[str]:
        types = [t for t in (asserted_types or []) if t in self.classes_by_name]
        concrete = [t for t in types if not self.is_abstract_class(t) and t in self.class_to_collection]
        if concrete:
            return concrete[0]
        for t in types:
            members = self.get_union_members(t)
            if members:
                concrete_members = [m for m in members if
                                    not self.is_abstract_class(m) and m in self.class_to_collection]
                if concrete_members:
                    return concrete_members[0]
        return types[0] if types else None

    def get_required_properties_for_class(self, class_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Returns ontology-level requiredness derived from restriction axioms.
        Useful for DAL validation beyond validator-required fields.
        """
        subjects = {class_name} | set(self.get_parent_classes(class_name, transitive=True))
        out: Dict[str, Dict[str, Any]] = {}
        for ax in self.axioms.get("restriction_axioms", []) or []:
            if ax.get("subject_class") not in subjects:
                continue
            prop = ax.get("property_name")
            rtype = ax.get("restriction_type")
            if not prop or not rtype:
                continue
            if rtype in {"data_some_values_from", "object_some_values_from"}:
                out[prop] = {"kind": "some", "cardinality": None, "inverse": bool(ax.get("inverse"))}
            elif rtype in {"object_min_cardinality", "data_min_cardinality"}:
                out[prop] = {"kind": "min", "cardinality": ax.get("cardinality"), "inverse": bool(ax.get("inverse"))}
            elif rtype in {"object_exact_cardinality", "data_exact_cardinality"}:
                out[prop] = {"kind": "exact", "cardinality": ax.get("cardinality"), "inverse": bool(ax.get("inverse"))}
        return out

    def get_cardinality_rules_for_class(self, class_name: str) -> Dict[str, Dict[str, Any]]:
        subjects = {class_name} | set(self.get_parent_classes(class_name, transitive=True))
        out: Dict[str, Dict[str, Any]] = {}
        for ax in self.axioms.get("restriction_axioms", []) or []:
            if ax.get("subject_class") not in subjects:
                continue
            prop = ax.get("property_name")
            rtype = ax.get("restriction_type")
            if not prop or not rtype:
                continue
            if rtype in {
                "object_min_cardinality",
                "object_max_cardinality",
                "object_exact_cardinality",
                "data_min_cardinality",
                "data_max_cardinality",
                "data_exact_cardinality",
            }:
                out[prop] = {
                    "restriction_type": rtype,
                    "cardinality": ax.get("cardinality"),
                    "inverse": bool(ax.get("inverse")),
                    "filler": ax.get("filler"),
                    "value_datatype": ax.get("value_datatype"),
                }
        return out

    def get_required_rule_map_for_class(self, class_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Merge ontology-level required rules and validator-level required fields.

        Keyed by mongo field name when possible, otherwise by ontology property name.
        """
        result: Dict[str, Dict[str, Any]] = {}
        coll = self.get_collection_for_class(class_name)
        validator_required_fields = self.get_required_fields(coll) if coll else set()
        ont_required = self.get_required_properties_for_class(class_name)

        for field_name in sorted(validator_required_fields):
            source_property = None
            if coll:
                source_property = self.field_to_source_property.get((coll, field_name))
            result[field_name] = {
                "mongo_field": field_name,
                "source_property": source_property,
                "required_by_validator": True,
                "required_by_ontology": False,
                "ontology_rule_kind": None,
            }

        for prop_name, rule in ont_required.items():
            field_name = self.get_mongo_field_for_property(class_name, prop_name)
            key = field_name or prop_name
            existing = result.setdefault(
                key,
                {
                    "mongo_field": field_name,
                    "source_property": prop_name,
                    "required_by_validator": False,
                    "required_by_ontology": False,
                    "ontology_rule_kind": None,
                },
            )
            existing["source_property"] = prop_name
            existing["required_by_ontology"] = True
            existing["ontology_rule_kind"] = rule.get("kind")

        return result

    def get_incoming_references(self, target_collection: str) -> List[Dict[str, Any]]:
        return list(self.incoming_reference_index.get(target_collection, []))

    @classmethod
    def from_files(
            cls,
            ontology_blueprint_path: str | Path,
            validators_path: Optional[str | Path] = None
    ) -> "BlueprintRegistry":
        ontology_bp = load_json(ontology_blueprint_path)
        validators = load_json(validators_path) if validators_path else None
        return cls(ontology_bp, validators)


if __name__ == "__main__":
    import argparse
    import pprint

    parser = argparse.ArgumentParser(description="Inspect ontology blueprint registry")
    parser.add_argument("ontology_blueprint",
                        help="Path to ontology_blueprint.json or ontology_blueprint_enriched.json")
    parser.add_argument("--validators", default=None, help="Optional path to mongo_validators.json")
    parser.add_argument("--class_name", default=None, help="Optional class to inspect")
    parser.add_argument("--collection", default=None, help="Optional collection to inspect")
    args = parser.parse_args()

    registry = BlueprintRegistry.from_files(args.ontology_blueprint, args.validators)

    print("=== BlueprintRegistry Summary ===")
    print(f"Classes: {len(registry.classes_by_name)}")
    print(f"Data properties: {len(registry.data_properties_by_name)}")
    print(f"Object properties: {len(registry.object_properties_by_name)}")
    print(f"Named individuals: {len(registry.named_individuals_by_name)}")
    print(f"Abstract classes: {len(registry.abstract_classes)}")
    print(f"Collections: {len(registry.collection_specs)}")
    print(f"Vocabulary types: {len(registry.vocabulary_types)}")

    if args.class_name:
        print(f"\n--- Class: {args.class_name} ---")
        print("exists:", registry.class_exists(args.class_name))
        print("abstract:", registry.is_abstract_class(args.class_name))
        print("parents:", registry.get_parent_classes(args.class_name, transitive=True))
        print("children:", registry.get_child_classes(args.class_name, transitive=True))
        print("collection:", registry.get_collection_for_class(args.class_name))
        print("data properties:", registry.get_allowed_data_properties_for_class(args.class_name))
        print("object properties:", registry.get_allowed_object_properties_for_class(args.class_name))
        print("required properties:")
        pprint.pp(registry.get_required_properties_for_class(args.class_name))
        print("cardinality rules:")
        pprint.pp(registry.get_cardinality_rules_for_class(args.class_name))
        print("required rule map:")
        pprint.pp(registry.get_required_rule_map_for_class(args.class_name))

    if args.collection:
        print(f"\n--- Collection: {args.collection} ---")
        print("classes:", registry.get_classes_for_collection(args.collection))
        print("required fields:", sorted(registry.get_required_fields(args.collection)))
        print("scalar fields:", sorted(registry.get_scalar_fields(args.collection).keys()))
        print("reference fields:", sorted(registry.get_reference_fields(args.collection).keys()))
        print("incoming references:")
        pprint.pp(registry.get_incoming_references(args.collection))
