from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from xml.etree import ElementTree as ET

RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"
OWL_NS = "http://www.w3.org/2002/07/owl#"
XML_NS = "http://www.w3.org/XML/1998/namespace"
XSD_NS = "http://www.w3.org/2001/XMLSchema#"

RESERVED_SCHEMA_TAGS = {
    (OWL_NS, "Ontology"),
    (OWL_NS, "Class"),
    (OWL_NS, "ObjectProperty"),
    (OWL_NS, "DatatypeProperty"),
    (OWL_NS, "AnnotationProperty"),
    (OWL_NS, "NamedIndividual"),
}

ANNOTATION_TAGS = {
    (RDFS_NS, "label"): "rdfs:label",
    (RDFS_NS, "comment"): "rdfs:comment",
    (OWL_NS, "versionInfo"): "owl:versionInfo",
}

APPEARANCE_OCCLUSION_BASE = "http://www.semanticweb.org/GRISERA/contextualOntology/models/appearanceOcclusion#"
IMPORTED_APPEARANCE_VALUE_CLASS = "ImportedAppearanceValue"


def _local_name(tag: str) -> str:
    if not tag:
        return ""
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _namespace_uri(tag: str) -> str:
    if tag.startswith("{") and "}" in tag:
        return tag[1:].split("}", 1)[0]
    return ""


def _text(elem: Optional[ET.Element]) -> str:
    return (elem.text or "").strip() if elem is not None else ""


def _get_resource_ref(elem: ET.Element) -> Optional[str]:
    for key in (
            f"{{{RDF_NS}}}resource",
            f"{{{RDF_NS}}}about",
            f"{{{RDF_NS}}}ID",
            "resource",
            "about",
            "ID",
    ):
        if key in elem.attrib:
            val = elem.attrib[key]
            if key.endswith("ID") and not val.startswith("#"):
                return f"#{val}"
            return val
    return None


def _resource_to_name(ref: Optional[str]) -> str:
    if not ref:
        return ""
    if ref.startswith("#"):
        return ref[1:]
    if ":" in ref and not ref.startswith("http://") and not ref.startswith("https://"):
        return ref
    return ref.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def _subject_identity(elem: ET.Element) -> Tuple[Optional[str], Optional[str]]:
    ref = _get_resource_ref(elem)
    if not ref:
        return None, None
    return _resource_to_name(ref), ref


def _make_literal_annotation(elem: ET.Element) -> Dict[str, Any]:
    return {
        "value": _text(elem),
        "datatype": elem.attrib.get(f"{{{RDF_NS}}}datatype"),
        "lang": elem.attrib.get(f"{{{XML_NS}}}lang"),
        "value_type": "literal",
    }


def _append_unique(lst: List[Any], item: Any) -> None:
    if item not in lst:
        lst.append(item)


def _is_subject_element(elem: ET.Element) -> bool:
    ns = _namespace_uri(elem.tag)
    ln = _local_name(elem.tag)
    if (ns, ln) in RESERVED_SCHEMA_TAGS:
        return False
    return _subject_identity(elem)[0] is not None


def _guess_xsd_range_from_text(value: str, datatype_hint: Optional[str]) -> str:
    if datatype_hint:
        short = _resource_to_name(datatype_hint)
        if short.startswith("xsd:"):
            return short
        if datatype_hint.startswith(XSD_NS):
            return "xsd:" + datatype_hint.rsplit("#", 1)[-1]
        return datatype_hint

    val = value.strip().lower()
    if val in {"true", "false"}:
        return "xsd:boolean"
    try:
        int(val)
        return "xsd:integer"
    except Exception:
        pass
    try:
        float(val)
        return "xsd:double"
    except Exception:
        pass
    return "xsd:string"


class UnifiedAugmenter:
    """
    Recursive RDF/XML instance augmenter.

    This version fixes a key downstream issue:
    imported object/data properties discovered only from instance OWL are not left
    merely in unresolved_imports; they are promoted into object_properties /
    data_properties with minimally inferred domains and ranges so validator and
    Mongo layer generation can materialize them.
    """

    def __init__(self, base_bp: Dict[str, Any], instance_path: str) -> None:
        self.bp = deepcopy(base_bp)
        self.instance_path = instance_path

        self.classes: Dict[str, Dict[str, Any]] = self.bp.setdefault("classes", {}) or {}
        self.data_properties: Dict[str, Dict[str, Any]] = self.bp.setdefault("data_properties", {}) or {}
        self.object_properties: Dict[str, Dict[str, Any]] = self.bp.setdefault("object_properties", {}) or {}
        self.named_individuals: Dict[str, Dict[str, Any]] = self.bp.setdefault("named_individuals", {}) or {}
        self.annotation_properties: Dict[str, Dict[str, Any]] = self.bp.setdefault("annotation_properties", {}) or {}
        self.axioms: Dict[str, Any] = self.bp.setdefault("axioms", {}) or {}
        self.axioms.setdefault("subclass_axioms", [])

        self.ordinary_individuals: Dict[str, Dict[str, Any]] = {}
        self.imported_individuals: Dict[str, Dict[str, Any]] = {}
        self.unresolved = {
            "classes": [],
            "data_properties": [],
            "object_properties": [],
            "resources": [],
        }
        self._seen_unresolved: Dict[str, Set[str]] = {k: set() for k in self.unresolved}
        self._visiting: Set[str] = set()

    # ---------- schema inference ----------

    def add_unresolved(self, bucket: str, value: str) -> None:
        if value and value not in self._seen_unresolved[bucket]:
            self._seen_unresolved[bucket].add(value)
            self.unresolved[bucket].append(value)

    def is_abstract_class(self, class_name: str) -> bool:
        cls = self.classes.get(class_name, {}) or {}
        anns = cls.get("annotations", {}) or {}
        for ann in anns.get("conceptType", []) or []:
            if str((ann or {}).get("value", "")).strip().lower() == "abstract":
                return True
        return False

    def ensure_class(self, class_name: str, iri: Optional[str] = None) -> None:
        if not class_name:
            return
        if class_name not in self.classes:
            self.classes[class_name] = {
                "name": class_name,
                "iri": iri or f"#{class_name}",
                "annotations": {},
            }

    def ensure_subclass_axiom(self, sub_class: str, super_class: str) -> None:
        if not sub_class or not super_class or sub_class == super_class:
            return
        payload = {"sub_class": sub_class, "super_class": super_class}
        if payload not in self.axioms["subclass_axioms"]:
            self.axioms["subclass_axioms"].append(payload)

    def ensure_data_property(self, prop_name: str, domain: Optional[str], xsd_range: Optional[str]) -> None:
        if not prop_name:
            return
        entry = self.data_properties.setdefault(
            prop_name,
            {
                "name": prop_name,
                "iri": f"#{prop_name}",
                "domains": [],
                "ranges": [],
                "is_functional": False,
                "cardinality_by_domain": {},
                "annotations": {},
            },
        )
        if domain:
            self.ensure_class(domain)
            _append_unique(entry["domains"], domain)
        if xsd_range:
            _append_unique(entry["ranges"], xsd_range)

    def ensure_object_property(self, prop_name: str, domain: Optional[str], target_class: Optional[str]) -> None:
        if not prop_name:
            return
        entry = self.object_properties.setdefault(
            prop_name,
            {
                "name": prop_name,
                "iri": f"#{prop_name}",
                "domains": [],
                "ranges": [],
                "inverse_of": None,
                "is_functional": False,
                "forward_cardinality_by_domain": {},
                "inverse_cardinality_by_domain": {},
                "annotations": {},
            },
        )
        if domain:
            self.ensure_class(domain)
            _append_unique(entry["domains"], domain)
        if target_class:
            self.ensure_class(target_class)
            _append_unique(entry["ranges"], target_class)

    def _declared_or_inferred_property_ranges(self, prop_name: str) -> List[str]:
        op = self.object_properties.get(prop_name, {}) or {}
        return list(op.get("ranges", []) or [])

    def _ensure_concrete_child_for_abstract_range(self, abstract_range: str) -> str:
        child_name = f"Imported{abstract_range}"
        self.ensure_class(child_name, f"#{child_name}")
        self.ensure_subclass_axiom(child_name, abstract_range)
        return child_name

    def target_type_candidates(self, prop_name: str, explicit_tag_class: Optional[str]) -> List[str]:
        if explicit_tag_class:
            return [explicit_tag_class]
        ranges = self._declared_or_inferred_property_ranges(prop_name)
        if not ranges:
            return []
        out: List[str] = []
        for r in ranges:
            if self.is_abstract_class(r):
                out.append(self._ensure_concrete_child_for_abstract_range(r))
            else:
                out.append(r)
        seen: Set[str] = set()
        uniq: List[str] = []
        for r in out:
            if r not in seen:
                seen.add(r)
                uniq.append(r)
        return uniq

    def _imported_target_type_overrides(self, prop_name: str, target_ref: str) -> List[str]:
        if not isinstance(target_ref, str):
            return []
        if target_ref.startswith(APPEARANCE_OCCLUSION_BASE) and prop_name in {"hasMoustacheValue", "hasBeardValue"}:
            self.ensure_class(IMPORTED_APPEARANCE_VALUE_CLASS, f"#{IMPORTED_APPEARANCE_VALUE_CLASS}")
            self.ensure_object_property(prop_name, None, IMPORTED_APPEARANCE_VALUE_CLASS)
            return [IMPORTED_APPEARANCE_VALUE_CLASS]
        return []

    def _candidate_types_for_resource_target(self, prop_name: str, target_ref: str) -> List[str]:
        inferred_targets = self.target_type_candidates(prop_name, None)
        if inferred_targets:
            return inferred_targets
        overrides = self._imported_target_type_overrides(prop_name, target_ref)
        if overrides:
            return overrides
        return []

    def mirror_annotations_for_seeding(self, ind_doc: Dict[str, Any]) -> None:
        anns = ind_doc.setdefault("annotations", {})
        labels = anns.get("rdfs:label", []) or []
        comments = anns.get("rdfs:comment", []) or []
        if labels and not comments:
            anns["rdfs:comment"] = deepcopy(labels)

    # ---------- individual storage ----------

    def _get_store(self, asserted_types: List[str]) -> Dict[str, Dict[str, Any]]:
        if any(t in self.classes for t in asserted_types):
            return self.ordinary_individuals
        return self.imported_individuals

    def _ensure_individual_doc(self, name: str, iri: str) -> Dict[str, Any]:
        doc = self.named_individuals.setdefault(
            name,
            {
                "name": name,
                "iri": iri,
                "asserted_types": [],
                "annotations": {},
                "data_assertions": {},
                "object_assertions": {},
            },
        )
        if not doc.get("iri"):
            doc["iri"] = iri
        doc.setdefault("annotations", {})
        doc.setdefault("data_assertions", {})
        doc.setdefault("object_assertions", {})
        return doc

    def _ensure_aux_individual_doc(self, store: Dict[str, Dict[str, Any]], name: str, iri: str) -> Dict[str, Any]:
        doc = store.setdefault(
            name,
            {
                "name": name,
                "iri": iri,
                "asserted_types": [],
                "annotations": {},
                "data_assertions": {},
                "object_assertions": {},
                "source_file": self.instance_path,
            },
        )
        if not doc.get("iri"):
            doc["iri"] = iri
        return doc

    def _append_annotation(self, ind_doc: Dict[str, Any], prop_name: str, value: Dict[str, Any]) -> None:
        arr = ind_doc.setdefault("annotations", {}).setdefault(prop_name, [])
        if value not in arr:
            arr.append(value)

    def _append_data_assertion(self, doc: Dict[str, Any], prop_name: str, value: Any) -> None:
        arr = doc.setdefault("data_assertions", {}).setdefault(prop_name, [])
        if value not in arr:
            arr.append(value)

    def _append_object_assertion(self, doc: Dict[str, Any], prop_name: str, target_ref: str) -> None:
        arr = doc.setdefault("object_assertions", {}).setdefault(prop_name, [])
        if target_ref not in arr:
            arr.append(target_ref)

    def _ensure_placeholder_individual(self, target_name: str, target_iri: str, candidate_types: List[str]) -> None:
        ind_doc = self._ensure_individual_doc(target_name, target_iri)
        for t in candidate_types:
            self.ensure_class(t)
            _append_unique(ind_doc["asserted_types"], t)
        self.mirror_annotations_for_seeding(ind_doc)

    # ---------- merge ----------

    def _merge_asserted_types(self, target: Dict[str, Any], asserted_types: List[str]) -> None:
        target.setdefault("asserted_types", [])
        for t in asserted_types:
            if t and t != "NamedIndividual":
                _append_unique(target["asserted_types"], t)

    # ---------- parse ----------

    def _record_literal_property(
            self,
            primary_doc: Dict[str, Any],
            aux_doc: Dict[str, Any],
            subject_domain: Optional[str],
            prop_name: str,
            child: ET.Element,
    ) -> None:
        literal_text = _text(child)
        datatype_hint = child.attrib.get(f"{{{RDF_NS}}}datatype")
        xsd_range = _guess_xsd_range_from_text(literal_text, datatype_hint)

        # Promote inferred imported data property into official blueprint.
        self.ensure_data_property(prop_name, subject_domain, xsd_range)

        self._append_data_assertion(primary_doc, prop_name, literal_text)
        self._append_data_assertion(aux_doc, prop_name, literal_text)

    def _record_resource_object(
            self,
            primary_doc: Dict[str, Any],
            aux_doc: Dict[str, Any],
            subject_domain: Optional[str],
            prop_name: str,
            target_ref: str,
    ) -> None:
        target_name = _resource_to_name(target_ref)
        inferred_targets = self._candidate_types_for_resource_target(prop_name, target_ref)

        # Promote inferred imported object property into official blueprint.
        if inferred_targets:
            for target_t in inferred_targets:
                self.ensure_object_property(prop_name, subject_domain, target_t)
        else:
            # Still promote property even if range is unknown for now.
            self.ensure_object_property(prop_name, subject_domain, None)

        if target_ref.startswith("http://") or target_ref.startswith("https://"):
            self.add_unresolved("resources", target_ref)
        self._ensure_placeholder_individual(target_name, target_ref, inferred_targets)
        self._append_object_assertion(primary_doc, prop_name, target_ref)
        self._append_object_assertion(aux_doc, prop_name, target_ref)

    def _record_nested_subjects(
            self,
            primary_doc: Dict[str, Any],
            aux_doc: Dict[str, Any],
            subject_domain: Optional[str],
            prop_name: str,
            nested_subjects: List[ET.Element],
    ) -> None:
        for nested in nested_subjects:
            nested_name, nested_iri = _subject_identity(nested)
            nested_tag_class = _local_name(nested.tag)
            inferred_targets = self.target_type_candidates(prop_name, nested_tag_class)

            # Promote inferred imported object property into official blueprint.
            if inferred_targets:
                for target_t in inferred_targets:
                    self.ensure_object_property(prop_name, subject_domain, target_t)
                    if nested_tag_class and target_t != nested_tag_class:
                        self.ensure_subclass_axiom(nested_tag_class, target_t)
            elif nested_tag_class:
                self.ensure_object_property(prop_name, subject_domain, nested_tag_class)
            else:
                self.ensure_object_property(prop_name, subject_domain, None)

            self.process_subject(nested, inferred_targets)

            if nested_iri:
                self._append_object_assertion(primary_doc, prop_name, nested_iri)
                self._append_object_assertion(aux_doc, prop_name, nested_iri)

    def process_subject(self, elem: ET.Element, implied_types: Optional[List[str]] = None) -> Optional[str]:
        name, iri = _subject_identity(elem)
        if not name or not iri:
            return None
        if iri in self._visiting:
            return name

        ns = _namespace_uri(elem.tag)
        ln = _local_name(elem.tag)

        tag_class = None
        if (ns, ln) not in RESERVED_SCHEMA_TAGS and ln not in {"type", "Description"}:
            tag_class = ln

        explicit_rdf_types = [
            _resource_to_name(ch.attrib.get(f"{{{RDF_NS}}}resource"))
            for ch in elem
            if (_namespace_uri(ch.tag), _local_name(ch.tag)) == (RDF_NS, "type")
               and ch.attrib.get(f"{{{RDF_NS}}}resource")
        ]

        asserted_types: List[str] = []
        if tag_class:
            _append_unique(asserted_types, tag_class)
        for t in explicit_rdf_types:
            if t and t != "NamedIndividual":
                _append_unique(asserted_types, t)
        for t in implied_types or []:
            if t and t != "NamedIndividual":
                _append_unique(asserted_types, t)

        for t in asserted_types:
            self.ensure_class(t)

        primary_doc = self._ensure_individual_doc(name, iri)
        self._merge_asserted_types(primary_doc, asserted_types)

        store = self._get_store(asserted_types)
        aux_doc = self._ensure_aux_individual_doc(store, name, iri)
        self._merge_asserted_types(aux_doc, asserted_types)

        self._visiting.add(iri)
        try:
            for child in list(elem):
                if not isinstance(child.tag, str):
                    continue

                c_ns = _namespace_uri(child.tag)
                c_ln = _local_name(child.tag)

                if (c_ns, c_ln) == (RDF_NS, "type"):
                    continue

                ann_key = ANNOTATION_TAGS.get((c_ns, c_ln))
                if ann_key:
                    val = _make_literal_annotation(child)
                    self._append_annotation(primary_doc, ann_key, val)
                    self._append_annotation(aux_doc, ann_key, val)
                    continue

                subject_domain = asserted_types[0] if asserted_types else None

                nested_subjects = [ch for ch in list(child) if isinstance(ch.tag, str) and _is_subject_element(ch)]
                if nested_subjects:
                    self._record_nested_subjects(primary_doc, aux_doc, subject_domain, c_ln, nested_subjects)
                    continue

                target_ref = child.attrib.get(f"{{{RDF_NS}}}resource")
                if target_ref:
                    self._record_resource_object(primary_doc, aux_doc, subject_domain, c_ln, target_ref)
                    continue

                literal = _text(child)
                if literal:
                    self._record_literal_property(primary_doc, aux_doc, subject_domain, c_ln, child)
                    continue

            self.mirror_annotations_for_seeding(primary_doc)
            self.mirror_annotations_for_seeding(aux_doc)
            return name
        finally:
            self._visiting.remove(iri)

    def collect(self) -> None:
        tree = ET.parse(self.instance_path)
        root = tree.getroot()

        for child in list(root):
            if isinstance(child.tag, str) and _is_subject_element(child):
                self.process_subject(child)

        self.bp["ordinary_individuals"] = self.ordinary_individuals
        self.bp["imported_individuals"] = self.imported_individuals
        self.bp["unresolved_imports"] = self.unresolved

        self.bp.setdefault("instance_notes", [])
        self.bp["instance_notes"].extend(
            [
                "ordinary_individuals mirrors recursively extracted instance assertions from the second OWL file.",
                "imported_individuals tracks subjects whose types were not originally in the base schema.",
                "named_individuals now also carries recursively extracted data_assertions and object_assertions for downstream reuse.",
                "Object assertions are stored in a downstream-compatible flat format (resource / iri strings).",
                "Data assertions are stored as downstream-compatible scalar values.",
                "Imported properties discovered from instance OWL are promoted into official object_properties/data_properties so validator and Mongo generation can materialize them.",
            ]
        )

        self.bp.setdefault("notes", [])
        self.bp["notes"].append(
            "This file was enriched in a second stage using a recursive RDF/XML instance expansion pass; extracted assertions were synchronized back into named_individuals and imported properties were promoted into official blueprint property maps."
        )


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def enrich_blueprint(base_bp: Dict[str, Any], instance_owl_path: str) -> Dict[str, Any]:
    augmenter = UnifiedAugmenter(base_bp, instance_owl_path)
    augmenter.collect()
    return augmenter.bp


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Augment a schema-derived ontology blueprint with an instance/import OWL using recursive RDF/XML expansion, "
            "synchronize extracted assertions back into named_individuals, and promote imported properties into the official blueprint maps."
        )
    )
    parser.add_argument("base_blueprint", type=str, help="Path to base ontology_blueprint.json built from schema OWL")
    parser.add_argument("instance_owl", type=str, help="Path to instance/import-rich OWL file")
    parser.add_argument(
        "--out",
        type=str,
        default="ontology_blueprint_enriched.json",
        help="Output path for enriched blueprint JSON",
    )
    args = parser.parse_args()

    base_path = Path(args.base_blueprint)
    instance_path = Path(args.instance_owl)
    if not base_path.exists():
        raise FileNotFoundError(f"Base blueprint not found: {base_path}")
    if not instance_path.exists():
        raise FileNotFoundError(f"Instance OWL not found: {instance_path}")

    base_bp = load_json(base_path)
    enriched = enrich_blueprint(base_bp, str(instance_path))

    out_path = Path(args.out)
    out_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")

    print("########################################")
    print("### Unified Enriched Blueprint       ###")
    print("########################################")
    print(f"Base blueprint : {base_path.resolve()}")
    print(f"Instance OWL   : {instance_path.resolve()}")
    print(f"Classes total  : {len(enriched.get('classes', {}))}")
    print(f"Data properties total : {len(enriched.get('data_properties', {}))}")
    print(f"Object properties total: {len(enriched.get('object_properties', {}))}")
    print(f"Named individuals total: {len(enriched.get('named_individuals', {}))}")
    print(f"Ordinary individuals view: {len(enriched.get('ordinary_individuals', {}))}")
    print(f"Imported individuals view: {len(enriched.get('imported_individuals', {}))}")
    print(f"Wrote enriched blueprint to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
