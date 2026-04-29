# ontology/owl_parser.py
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Dict, Optional

from ontology.ontology_model import (
    AnnotationPropertyDef,
    ClassDef,
    DataPropertyDef,
    ObjectPropertyDef,
    OntologyModel,
)

RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"
OWL_NS = "http://www.w3.org/2002/07/owl#"
XSD_NS = "http://www.w3.org/2001/XMLSchema#"


def _local_name(tag: str) -> str:
    if tag.startswith("{") and "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _ns_uri(tag: str) -> Optional[str]:
    if tag.startswith("{") and "}" in tag:
        return tag[1:].split("}", 1)[0]
    return None


def _read_literal_text(el: ET.Element) -> Optional[str]:
    if el.text is None:
        return None
    t = el.text.strip()
    return t if t else None


def _iri_to_name(iri: str) -> str:
    if "#" in iri:
        return iri.split("#")[-1]
    return iri.rstrip("/").split("/")[-1]


def _collect_xmlns_map(root: ET.Element) -> Dict[str, str]:
    nsmap: Dict[str, str] = {}
    for k, v in root.attrib.items():
        if k.startswith("xmlns:"):
            nsmap[k.split(":", 1)[1]] = v
        elif k == "xmlns":
            nsmap[""] = v
    return nsmap


def _ensure_model_extras(model: OntologyModel) -> None:
    """
    Make parser robust even if OntologyModel was edited manually and is missing fields.

    This fixes the situation where ontology_model.py contains multiple OntologyModel
    definitions or missing attributes: we add them dynamically so parsing never crashes.
    """
    # base fields
    if not hasattr(model, "namespace_map"):
        model.namespace_map = {}
    if not hasattr(model, "base_iri"):
        model.base_iri = None
    if not hasattr(model, "source_path"):
        model.source_path = None

    # Step 3.5 helpers
    if not hasattr(model, "functional_object_properties"):
        model.functional_object_properties = set()
    if not hasattr(model, "functional_data_properties"):
        model.functional_data_properties = set()

    # Cardinality axioms: (domain_class, property_name, kind, n, is_inverse)
    if not hasattr(model, "object_property_cardinality_axioms"):
        model.object_property_cardinality_axioms = []


# ---------------- OWL/XML PARSER ----------------

def _is_owlxml(root: ET.Element) -> bool:
    return (
            (_ns_uri(root.tag) == OWL_NS)
            and (_local_name(root.tag) == "Ontology")
            and (root.find(f"./{{{OWL_NS}}}Declaration") is not None)
    )


def _owlxml_get_iri(el: ET.Element) -> Optional[str]:
    # OWL/XML uses IRI="#X" or abbreviatedIRI="xsd:string"
    return el.attrib.get("IRI") or el.attrib.get("abbreviatedIRI")


def _ensure_class(model: OntologyModel, iri: str) -> ClassDef:
    name = _iri_to_name(iri)
    if name not in model.classes:
        model.classes[name] = ClassDef(name=name, iri=iri, annotations={})
    return model.classes[name]


def _ensure_dp(model: OntologyModel, iri: str) -> DataPropertyDef:
    name = _iri_to_name(iri)
    if name not in model.data_properties:
        model.data_properties[name] = DataPropertyDef(
            name=name,
            iri=iri,
            domains=set(),
            range_datatype=None,
            cardinality_by_domain={},
            annotations={},
        )
    return model.data_properties[name]


def _ensure_op(model: OntologyModel, iri: str) -> ObjectPropertyDef:
    name = _iri_to_name(iri)
    if name not in model.object_properties:
        model.object_properties[name] = ObjectPropertyDef(
            name=name,
            iri=iri,
            domains=set(),
            ranges=set(),
            inverse_of=None,
            cardinality_by_domain={},
            annotations={},
        )
    return model.object_properties[name]


def _parse_owlxml(model: OntologyModel, root: ET.Element) -> OntologyModel:
    _ensure_model_extras(model)

    model.namespace_map = _collect_xmlns_map(root)
    model.base_iri = root.attrib.get("xml:base") or root.attrib.get("ontologyIRI")

    # --- Declarations ---
    for decl in root.findall(f"./{{{OWL_NS}}}Declaration"):
        children = list(decl)
        if not children:
            continue
        item = children[0]
        kind = _local_name(item.tag)
        iri = _owlxml_get_iri(item)
        if not iri:
            continue

        if kind == "Class":
            _ensure_class(model, iri)
        elif kind == "DataProperty":
            _ensure_dp(model, iri)
        elif kind == "ObjectProperty":
            _ensure_op(model, iri)
        elif kind == "AnnotationProperty":
            name = _iri_to_name(iri)
            model.annotation_properties[name] = AnnotationPropertyDef(name=name, iri=iri)

    # --- ObjectPropertyDomain / Range ---
    for opd in root.findall(f"./{{{OWL_NS}}}ObjectPropertyDomain"):
        op_el = opd.find(f"./{{{OWL_NS}}}ObjectProperty")
        cls_el = opd.find(f"./{{{OWL_NS}}}Class")
        if op_el is None or cls_el is None:
            continue
        op_iri = _owlxml_get_iri(op_el)
        cls_iri = _owlxml_get_iri(cls_el)
        if not op_iri or not cls_iri:
            continue
        op = _ensure_op(model, op_iri)
        cls = _ensure_class(model, cls_iri)
        op.domains.add(cls.name)

    for opr in root.findall(f"./{{{OWL_NS}}}ObjectPropertyRange"):
        op_el = opr.find(f"./{{{OWL_NS}}}ObjectProperty")
        cls_el = opr.find(f"./{{{OWL_NS}}}Class")
        if op_el is None or cls_el is None:
            continue
        op_iri = _owlxml_get_iri(op_el)
        cls_iri = _owlxml_get_iri(cls_el)
        if not op_iri or not cls_iri:
            continue
        op = _ensure_op(model, op_iri)
        cls = _ensure_class(model, cls_iri)
        op.ranges.add(cls.name)

    # --- DataPropertyDomain / Range ---
    for dpd in root.findall(f"./{{{OWL_NS}}}DataPropertyDomain"):
        dp_el = dpd.find(f"./{{{OWL_NS}}}DataProperty")
        cls_el = dpd.find(f"./{{{OWL_NS}}}Class")
        if dp_el is None or cls_el is None:
            continue
        dp_iri = _owlxml_get_iri(dp_el)
        cls_iri = _owlxml_get_iri(cls_el)
        if not dp_iri or not cls_iri:
            continue
        dp = _ensure_dp(model, dp_iri)
        cls = _ensure_class(model, cls_iri)
        dp.domains.add(cls.name)

    for dpr in root.findall(f"./{{{OWL_NS}}}DataPropertyRange"):
        dp_el = dpr.find(f"./{{{OWL_NS}}}DataProperty")
        dt_el = dpr.find(f"./{{{OWL_NS}}}Datatype")
        if dp_el is None or dt_el is None:
            continue
        dp_iri = _owlxml_get_iri(dp_el)
        if not dp_iri:
            continue
        dp = _ensure_dp(model, dp_iri)
        dt = dt_el.attrib.get("abbreviatedIRI") or dt_el.attrib.get("IRI")
        dp.range_datatype = dt

    # --- InverseObjectProperties ---
    for inv in root.findall(f"./{{{OWL_NS}}}InverseObjectProperties"):
        ops = inv.findall(f"./{{{OWL_NS}}}ObjectProperty")
        if len(ops) != 2:
            continue
        a = _owlxml_get_iri(ops[0])
        b = _owlxml_get_iri(ops[1])
        if not a or not b:
            continue
        op_a = _ensure_op(model, a)
        op_b = _ensure_op(model, b)
        op_a.inverse_of = op_b.name
        op_b.inverse_of = op_a.name

    # --- FunctionalObjectProperty / FunctionalDataProperty ---
    # These drive Step 3.5 "single value field" decision.
    for fop in root.findall(f"./{{{OWL_NS}}}FunctionalObjectProperty"):
        op_el = fop.find(f"./{{{OWL_NS}}}ObjectProperty")
        if op_el is None:
            continue
        op_iri = _owlxml_get_iri(op_el)
        if not op_iri:
            continue
        op = _ensure_op(model, op_iri)
        # safe even if model lacks field (ensured above)
        model.functional_object_properties.add(op.name)

    for fdp in root.findall(f"./{{{OWL_NS}}}FunctionalDataProperty"):
        dp_el = fdp.find(f"./{{{OWL_NS}}}DataProperty")
        if dp_el is None:
            continue
        dp_iri = _owlxml_get_iri(dp_el)
        if not dp_iri:
            continue
        dp = _ensure_dp(model, dp_iri)
        model.functional_data_properties.add(dp.name)

    # --- AnnotationAssertion ---
    for aa in root.findall(f"./{{{OWL_NS}}}AnnotationAssertion"):
        ap_el = aa.find(f"./{{{OWL_NS}}}AnnotationProperty")
        iri_el = aa.find(f"./{{{OWL_NS}}}IRI")
        lit_el = aa.find(f"./{{{OWL_NS}}}Literal")
        if ap_el is None or iri_el is None or lit_el is None:
            continue
        ap_iri = _owlxml_get_iri(ap_el)
        target_iri = _read_literal_text(iri_el)
        lit = _read_literal_text(lit_el)
        if not ap_iri or not target_iri or not lit:
            continue

        ap_name = _iri_to_name(ap_iri)
        target_name = _iri_to_name(target_iri)

        if target_name in model.classes:
            model.classes[target_name].annotations.setdefault(ap_name, []).append(lit)
        elif target_name in model.data_properties:
            model.data_properties[target_name].annotations.setdefault(ap_name, []).append(lit)
        elif target_name in model.object_properties:
            model.object_properties[target_name].annotations.setdefault(ap_name, []).append(lit)

        if ap_name not in model.annotation_properties:
            model.annotation_properties[ap_name] = AnnotationPropertyDef(name=ap_name, iri=ap_iri)

    # --- Cardinality in SubClassOf ---
    # Store axioms instead of inventing "inv:*" synthetic properties.
    for sc in root.findall(f"./{{{OWL_NS}}}SubClassOf"):
        c_el = sc.find(f"./{{{OWL_NS}}}Class")
        if c_el is None:
            continue
        class_iri = _owlxml_get_iri(c_el)
        if not class_iri:
            continue
        cls = _ensure_class(model, class_iri)

        for card_tag, kind in (
                ("ObjectExactCardinality", "exact"),
                ("ObjectMinCardinality", "min"),
                ("ObjectMaxCardinality", "max"),
        ):
            # priority: exact > min/max (schema builder can decide further)
            card_el = sc.find(f"./{{{OWL_NS}}}{card_tag}")
            if card_el is None:
                continue

            try:
                n = int(card_el.attrib.get("cardinality", ""))
            except ValueError:
                continue

            prop_iri = None
            is_inverse = False

            inv_of = card_el.find(f"./{{{OWL_NS}}}ObjectInverseOf")
            if inv_of is not None:
                op_el = inv_of.find(f"./{{{OWL_NS}}}ObjectProperty")
                if op_el is not None:
                    prop_iri = _owlxml_get_iri(op_el)
                    is_inverse = True
            else:
                op_el = card_el.find(f"./{{{OWL_NS}}}ObjectProperty")
                if op_el is not None:
                    prop_iri = _owlxml_get_iri(op_el)

            if not prop_iri:
                continue

            prop_name = _iri_to_name(prop_iri)
            _ensure_op(model, prop_iri)

            model.object_property_cardinality_axioms.append(
                (cls.name, prop_name, kind, n, is_inverse)
            )

    return model


# ---------------- RDF/XML PARSER (fallback; minimal) ----------------

def _rdfxml_get_attr(el: ET.Element, local: str) -> Optional[str]:
    for k, v in el.attrib.items():
        if _local_name(k) == local:
            return v
    return None


def _parse_rdfxml(model: OntologyModel, root: ET.Element) -> OntologyModel:
    _ensure_model_extras(model)

    model.namespace_map = _collect_xmlns_map(root)

    for c in root.findall(f".//{{{OWL_NS}}}Class"):
        iri = _rdfxml_get_attr(c, "about") or _rdfxml_get_attr(c, "ID")
        if iri:
            _ensure_class(model, iri)

    for dp in root.findall(f".//{{{OWL_NS}}}DatatypeProperty") + root.findall(f".//{{{OWL_NS}}}DataProperty"):
        iri = _rdfxml_get_attr(dp, "about") or _rdfxml_get_attr(dp, "ID")
        if iri:
            _ensure_dp(model, iri)

    for op in root.findall(f".//{{{OWL_NS}}}ObjectProperty"):
        iri = _rdfxml_get_attr(op, "about") or _rdfxml_get_attr(op, "ID")
        if iri:
            _ensure_op(model, iri)

    for ap in root.findall(f".//{{{OWL_NS}}}AnnotationProperty"):
        iri = _rdfxml_get_attr(ap, "about") or _rdfxml_get_attr(ap, "ID")
        if iri:
            name = _iri_to_name(iri)
            model.annotation_properties[name] = AnnotationPropertyDef(name=name, iri=iri)

    return model


def parse_owl(owl_path: str) -> OntologyModel:
    tree = ET.parse(owl_path)
    root = tree.getroot()

    model = OntologyModel()
    _ensure_model_extras(model)
    model.source_path = owl_path

    if _is_owlxml(root):
        return _parse_owlxml(model, root)

    return _parse_rdfxml(model, root)
