from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from xml.etree import ElementTree as ET

from mapping.mapping_rules import MappingConfig, MappingOverrides

XML_NS = "http://www.w3.org/XML/1998/namespace"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"
OWL_NS = "http://www.w3.org/2002/07/owl#"
XSD_NS = "http://www.w3.org/2001/XMLSchema#"


@dataclass
class AnnotationValueBlueprint:
    value: str
    lang: Optional[str] = None
    datatype: Optional[str] = None
    value_type: str = "literal"


@dataclass
class AnnotationPropertyBlueprint:
    name: str
    iri: str = ""
    annotations: Dict[str, List["AnnotationValueBlueprint"]] = field(default_factory=dict)


@dataclass
class NamedIndividualBlueprint:
    name: str
    iri: str = ""
    asserted_types: List[str] = field(default_factory=list)
    annotations: Dict[str, List["AnnotationValueBlueprint"]] = field(default_factory=dict)


@dataclass
class ClassBlueprint:
    name: str
    iri: str = ""
    annotations: Dict[str, List["AnnotationValueBlueprint"]] = field(default_factory=dict)


@dataclass
class DataPropertyBlueprint:
    name: str
    iri: str = ""
    domains: List[str] = field(default_factory=list)
    ranges: List[str] = field(default_factory=list)
    is_functional: bool = False
    cardinality_by_domain: Dict[str, Dict[str, Optional[int]]] = field(default_factory=dict)
    annotations: Dict[str, List["AnnotationValueBlueprint"]] = field(default_factory=dict)


@dataclass
class ObjectPropertyBlueprint:
    name: str
    iri: str = ""
    domains: List[str] = field(default_factory=list)
    ranges: List[str] = field(default_factory=list)
    inverse_of: Optional[str] = None
    is_functional: bool = False
    forward_cardinality_by_domain: Dict[str, Dict[str, Optional[int]]] = field(default_factory=dict)
    inverse_cardinality_by_domain: Dict[str, Dict[str, Optional[int]]] = field(default_factory=dict)
    annotations: Dict[str, List["AnnotationValueBlueprint"]] = field(default_factory=dict)


@dataclass
class OntologyInfoBlueprint:
    ontology_iri: Optional[str] = None
    xml_base: Optional[str] = None
    prefixes: Dict[str, str] = field(default_factory=dict)
    namespaces: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, List["AnnotationValueBlueprint"]] = field(default_factory=dict)


@dataclass(frozen=True)
class SubclassAxiomBlueprint:
    sub_class: str
    super_class: str


@dataclass(frozen=True)
class EquivalentClassAxiomBlueprint:
    class_name: str
    expression_type: str
    operands: Tuple[str, ...] = ()


@dataclass(frozen=True)
class DisjointClassAxiomBlueprint:
    class_names: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RestrictionAxiomBlueprint:
    subject_class: str
    restriction_type: str
    property_name: str
    inverse: bool = False
    filler: Optional[str] = None
    cardinality: Optional[int] = None
    value_datatype: Optional[str] = None


@dataclass
class AxiomsBlueprint:
    subclass_axioms: List[SubclassAxiomBlueprint] = field(default_factory=list)
    equivalent_class_axioms: List[EquivalentClassAxiomBlueprint] = field(default_factory=list)
    disjoint_class_axioms: List[DisjointClassAxiomBlueprint] = field(default_factory=list)
    restriction_axioms: List[RestrictionAxiomBlueprint] = field(default_factory=list)


@dataclass
class OntologyBlueprint:
    ontology_info: OntologyInfoBlueprint = field(default_factory=OntologyInfoBlueprint)
    classes: Dict[str, ClassBlueprint] = field(default_factory=dict)
    data_properties: Dict[str, DataPropertyBlueprint] = field(default_factory=dict)
    object_properties: Dict[str, ObjectPropertyBlueprint] = field(default_factory=dict)
    annotation_properties: Dict[str, AnnotationPropertyBlueprint] = field(default_factory=dict)
    named_individuals: Dict[str, NamedIndividualBlueprint] = field(default_factory=dict)
    axioms: AxiomsBlueprint = field(default_factory=AxiomsBlueprint)
    notes: List[str] = field(default_factory=list)


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


def _normalize_prefix_name(name: str) -> str:
    return name[:-1] if name.endswith(":") else name


def _entity_ref_from_elem(elem: Optional[ET.Element]) -> Optional[str]:
    if elem is None:
        return None
    for key in ("IRI", "abbreviatedIRI", "iri", "abbreviatediri"):
        if key in elem.attrib:
            return elem.attrib[key]
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
    txt = _text(elem)
    return txt or None


def _name_from_ref(ref: Optional[str]) -> str:
    if not ref:
        return ""
    if ref.startswith("#"):
        return ref[1:]
    if ":" in ref and not ref.startswith("http://") and not ref.startswith("https://"):
        return ref
    return ref.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def _class_iri(name: str) -> str:
    return f"#{name}"


def _make_ann_value(value: str, lang: Optional[str], datatype: Optional[str],
                    value_type: str) -> AnnotationValueBlueprint:
    return AnnotationValueBlueprint(value=value, lang=lang, datatype=datatype, value_type=value_type)


def _ann_rank(v: AnnotationValueBlueprint) -> Tuple[int, int, int]:
    return (1 if v.lang else 0, 1 if v.datatype else 0, 1 if v.value_type != "literal" else 0)


def _append_annotation(bucket: Dict[str, List[AnnotationValueBlueprint]], prop_name: str,
                       value: AnnotationValueBlueprint) -> None:
    arr = bucket.setdefault(prop_name, [])
    for i, existing in enumerate(arr):
        if existing.value == value.value and existing.value_type == value.value_type:
            if existing.lang == value.lang and existing.datatype == value.datatype:
                return
            if _ann_rank(value) > _ann_rank(existing):
                arr[i] = value
            return
    arr.append(value)


def _append_unique(lst: List[Any], item: Any) -> None:
    if item not in lst:
        lst.append(item)


def _to_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value]
    return [str(value)]


def _normalize_min_max(min_c: Optional[int], max_c: Optional[int]) -> Dict[str, Optional[int]]:
    return {"min": min_c, "max": max_c}


def _parse_property_expression(elem: ET.Element) -> Tuple[Optional[str], bool]:
    lname = _local_name(elem.tag)
    if lname == "ObjectProperty":
        return _name_from_ref(_entity_ref_from_elem(elem)), False
    if lname == "ObjectInverseOf":
        for ch in elem:
            if _local_name(ch.tag) == "ObjectProperty":
                return _name_from_ref(_entity_ref_from_elem(ch)), True
    return None, False


def _parse_class_expression_name(elem: Optional[ET.Element]) -> Optional[str]:
    if elem is None:
        return None
    if _local_name(elem.tag) == "Class":
        return _name_from_ref(_entity_ref_from_elem(elem))
    return None


def _is_annotation_tag(tag: str) -> bool:
    ns = _namespace_uri(tag)
    lname = _local_name(tag)
    if ns == RDFS_NS and lname in {"label", "comment", "seeAlso", "isDefinedBy"}:
        return True
    return ns == OWL_NS and lname in {"versionInfo", "deprecated"}


def _parse_rdf_literal(elem: ET.Element) -> AnnotationValueBlueprint:
    datatype = elem.attrib.get(f"{{{RDF_NS}}}datatype")
    if datatype:
        return _make_ann_value(_text(elem), None, datatype, "literal")
    return _make_ann_value(_text(elem), elem.attrib.get(f"{{{XML_NS}}}lang"), None, "literal")


def _iter_non_comment_children(elem: ET.Element) -> List[ET.Element]:
    return [ch for ch in list(elem) if isinstance(ch.tag, str)]


@dataclass
class _XmlExtraction:
    ontology_info: OntologyInfoBlueprint
    annotation_properties: Dict[str, AnnotationPropertyBlueprint]
    target_annotations: Dict[str, Dict[str, List[AnnotationValueBlueprint]]]
    subclass_axioms: List[SubclassAxiomBlueprint]
    equivalent_class_axioms: List[EquivalentClassAxiomBlueprint]
    disjoint_class_axioms: List[DisjointClassAxiomBlueprint]
    class_assertions: Dict[str, List[str]]
    restriction_axioms: List[RestrictionAxiomBlueprint]
    xml_classes: Dict[str, Dict[str, Any]]
    xml_data_properties: Dict[str, Dict[str, Any]]
    xml_object_properties: Dict[str, Dict[str, Any]]


def _extract_xml_metadata(owl_path: str) -> _XmlExtraction:
    namespaces: Dict[str, str] = {}
    for _event, item in ET.iterparse(owl_path, events=("start-ns",)):
        prefix, uri = item
        namespaces[prefix or ""] = uri

    tree = ET.parse(owl_path)
    root = tree.getroot()

    ontology_info = OntologyInfoBlueprint(
        ontology_iri=root.attrib.get("ontologyIRI") or None,
        xml_base=root.attrib.get(f"{{{XML_NS}}}base") or None,
        prefixes={},
        namespaces=dict(namespaces),
        annotations={},
    )

    annotation_properties: Dict[str, AnnotationPropertyBlueprint] = {}
    target_annotations: Dict[str, Dict[str, List[AnnotationValueBlueprint]]] = {}
    subclass_axioms: List[SubclassAxiomBlueprint] = []
    equivalent_class_axioms: List[EquivalentClassAxiomBlueprint] = []
    disjoint_class_axioms: List[DisjointClassAxiomBlueprint] = []
    class_assertions: Dict[str, List[str]] = {}
    restriction_axioms: List[RestrictionAxiomBlueprint] = []
    xml_classes: Dict[str, Dict[str, Any]] = {}
    xml_data_properties: Dict[str, Dict[str, Any]] = {}
    xml_object_properties: Dict[str, Dict[str, Any]] = {}

    def ensure_ann_prop(prop_ref: str) -> None:
        name = _name_from_ref(prop_ref)
        iri = prop_ref
        if prop_ref.startswith("rdfs:"):
            iri = namespaces.get("rdfs", "") + prop_ref.split(":", 1)[1]
        elif prop_ref.startswith("rdf:"):
            iri = namespaces.get("rdf", "") + prop_ref.split(":", 1)[1]
        elif prop_ref.startswith("owl:"):
            iri = namespaces.get("owl", "") + prop_ref.split(":", 1)[1]
        elif prop_ref.startswith("xsd:"):
            iri = namespaces.get("xsd", "") + prop_ref.split(":", 1)[1]
        annotation_properties.setdefault(name, AnnotationPropertyBlueprint(name=name, iri=iri))

    def parse_annotation_value(elem: ET.Element) -> AnnotationValueBlueprint:
        lname = _local_name(elem.tag)
        if lname == "Literal":
            return _make_ann_value(
                value=_text(elem),
                lang=elem.attrib.get(f"{{{XML_NS}}}lang"),
                datatype=elem.attrib.get("datatypeIRI") or elem.attrib.get("datatypeiri"),
                value_type="literal",
            )
        ref = _entity_ref_from_elem(elem) or _text(elem)
        return _make_ann_value(value=ref, lang=None, datatype=None, value_type="iri")

    def parse_annotation_container(annotation_elem: ET.Element):
        prop_name = None
        value = None
        for ch in _iter_non_comment_children(annotation_elem):
            lname = _local_name(ch.tag)
            if lname == "AnnotationProperty":
                prop_name = _entity_ref_from_elem(ch)
            elif lname in {"Literal", "IRI", "AbbreviatedIRI", "AnonymousIndividual"}:
                value = parse_annotation_value(ch)
        return prop_name, value

    for child in _iter_non_comment_children(root):
        if _local_name(child.tag) == "Prefix":
            raw_name = child.attrib.get("name", "")
            iri = child.attrib.get("IRI", "")
            ontology_info.prefixes[_normalize_prefix_name(raw_name)] = iri
    if not ontology_info.prefixes:
        ontology_info.prefixes = dict(namespaces)

    for child in _iter_non_comment_children(root):
        if _local_name(child.tag) == "Annotation":
            prop_ref, ann_value = parse_annotation_container(child)
            if prop_ref and ann_value:
                ensure_ann_prop(prop_ref)
                _append_annotation(ontology_info.annotations, _name_from_ref(prop_ref), ann_value)

    for elem in root.iter():
        if not isinstance(elem.tag, str):
            continue
        if _local_name(elem.tag) != "AnnotationAssertion":
            continue
        children = _iter_non_comment_children(elem)
        if len(children) < 3:
            continue
        prop_ref = _entity_ref_from_elem(children[0])
        target_ref = _entity_ref_from_elem(children[1])
        if not prop_ref or not target_ref:
            continue
        ensure_ann_prop(prop_ref)
        ann_value = parse_annotation_value(children[2])
        bucket = target_annotations.setdefault(target_ref, {})
        _append_annotation(bucket, _name_from_ref(prop_ref), ann_value)

    # -------- OWL/XML extraction --------
    for elem in _iter_non_comment_children(root):
        lname = _local_name(elem.tag)

        if lname == "SubClassOf":
            children = _iter_non_comment_children(elem)
            if len(children) != 2:
                continue
            left, right = children
            left_name = _parse_class_expression_name(left)
            if not left_name:
                continue
            right_lname = _local_name(right.tag)
            if right_lname == "Class":
                right_name = _parse_class_expression_name(right)
                if right_name:
                    _append_unique(subclass_axioms, SubclassAxiomBlueprint(left_name, right_name))
            elif right_lname in {
                "ObjectSomeValuesFrom",
                "ObjectAllValuesFrom",
                "ObjectMinCardinality",
                "ObjectMaxCardinality",
                "ObjectExactCardinality",
                "DataSomeValuesFrom",
            }:
                if right_lname == "ObjectSomeValuesFrom":
                    ch = _iter_non_comment_children(right)
                    if len(ch) >= 2:
                        prop_name, inverse = _parse_property_expression(ch[0])
                        filler = _parse_class_expression_name(ch[1])
                        if prop_name:
                            _append_unique(
                                restriction_axioms,
                                RestrictionAxiomBlueprint(
                                    subject_class=left_name,
                                    restriction_type="object_some_values_from",
                                    property_name=prop_name,
                                    inverse=inverse,
                                    filler=filler,
                                ),
                            )
                elif right_lname == "ObjectAllValuesFrom":
                    ch = _iter_non_comment_children(right)
                    if len(ch) >= 2:
                        prop_name, inverse = _parse_property_expression(ch[0])
                        filler = _parse_class_expression_name(ch[1])
                        if prop_name:
                            _append_unique(
                                restriction_axioms,
                                RestrictionAxiomBlueprint(
                                    subject_class=left_name,
                                    restriction_type="object_all_values_from",
                                    property_name=prop_name,
                                    inverse=inverse,
                                    filler=filler,
                                ),
                            )
                elif right_lname in {"ObjectMinCardinality", "ObjectMaxCardinality", "ObjectExactCardinality"}:
                    card = right.attrib.get("cardinality")
                    try:
                        card_n = int(card) if card is not None else None
                    except ValueError:
                        card_n = None
                    ch = _iter_non_comment_children(right)
                    if ch:
                        prop_name, inverse = _parse_property_expression(ch[0])
                        filler = _parse_class_expression_name(ch[1]) if len(ch) > 1 else None
                        if prop_name:
                            rtype = {
                                "ObjectMinCardinality": "object_min_cardinality",
                                "ObjectMaxCardinality": "object_max_cardinality",
                                "ObjectExactCardinality": "object_exact_cardinality",
                            }[right_lname]
                            _append_unique(
                                restriction_axioms,
                                RestrictionAxiomBlueprint(
                                    subject_class=left_name,
                                    restriction_type=rtype,
                                    property_name=prop_name,
                                    inverse=inverse,
                                    filler=filler,
                                    cardinality=card_n,
                                ),
                            )
                elif right_lname == "DataSomeValuesFrom":
                    ch = _iter_non_comment_children(right)
                    if len(ch) >= 2 and _local_name(ch[0].tag) == "DataProperty":
                        prop_name = _name_from_ref(_entity_ref_from_elem(ch[0]))
                        dtype = _entity_ref_from_elem(ch[1]) or _text(ch[1])
                        if prop_name:
                            _append_unique(
                                restriction_axioms,
                                RestrictionAxiomBlueprint(
                                    subject_class=left_name,
                                    restriction_type="data_some_values_from",
                                    property_name=prop_name,
                                    inverse=False,
                                    value_datatype=dtype,
                                ),
                            )

        elif lname == "EquivalentClasses":
            children = _iter_non_comment_children(elem)
            if len(children) != 2:
                continue
            left, right = children
            left_name = _parse_class_expression_name(left)
            if not left_name:
                continue
            if _local_name(right.tag) == "ObjectUnionOf":
                ops = tuple(_parse_class_expression_name(ch) for ch in _iter_non_comment_children(right) if
                            _parse_class_expression_name(ch))
                if ops:
                    _append_unique(equivalent_class_axioms,
                                   EquivalentClassAxiomBlueprint(left_name, "object_union_of", ops))
            elif _local_name(right.tag) == "Class":
                right_name = _parse_class_expression_name(right)
                if right_name:
                    _append_unique(equivalent_class_axioms,
                                   EquivalentClassAxiomBlueprint(left_name, "named_class_list", (right_name,)))

        elif lname == "DisjointClasses":
            names = tuple(sorted({_parse_class_expression_name(ch) for ch in _iter_non_comment_children(elem) if
                                  _parse_class_expression_name(ch)}))
            if len(names) >= 2:
                _append_unique(disjoint_class_axioms, DisjointClassAxiomBlueprint(names))

        elif lname == "ClassAssertion":
            children = _iter_non_comment_children(elem)
            if len(children) != 2:
                continue
            cls_elem, ind_elem = children
            cls_name = _parse_class_expression_name(cls_elem)
            ind_ref = _entity_ref_from_elem(ind_elem)
            ind_name = _name_from_ref(ind_ref)
            if cls_name and ind_name:
                class_assertions.setdefault(ind_name, [])
                if cls_name not in class_assertions[ind_name]:
                    class_assertions[ind_name].append(cls_name)

    # -------- RDF/XML extraction --------
    def ensure_xml_class(name: str) -> None:
        if not name:
            return
        xml_classes.setdefault(name, {"iri": _class_iri(name), "annotations": {}})

    def ensure_xml_dp(name: str) -> None:
        if not name:
            return
        xml_data_properties.setdefault(
            name,
            {
                "iri": _class_iri(name),
                "domains": set(),
                "ranges": set(),
                "is_functional": False,
                "annotations": {},
            },
        )

    def ensure_xml_op(name: str) -> None:
        if not name:
            return
        xml_object_properties.setdefault(
            name,
            {
                "iri": _class_iri(name),
                "domains": set(),
                "ranges": set(),
                "is_functional": False,
                "inverse_of": None,
                "annotations": {},
            },
        )

    def add_target_annotation(target_ref: str, prop_name: str, value: AnnotationValueBlueprint) -> None:
        bucket = target_annotations.setdefault(target_ref, {})
        _append_annotation(bucket, prop_name, value)

    def add_rdf_annotations(elem: ET.Element, target_ref: str) -> None:
        for ch in _iter_non_comment_children(elem):
            if _is_annotation_tag(ch.tag):
                ns = _namespace_uri(ch.tag)
                lname = _local_name(ch.tag)
                if ns == RDFS_NS:
                    prop = f"rdfs:{lname}"
                elif ns == OWL_NS:
                    prop = f"owl:{lname}"
                else:
                    prop = lname
                ensure_ann_prop(prop)
                add_target_annotation(target_ref, _name_from_ref(prop), _parse_rdf_literal(ch))

    def rdf_ref(attr_key: str, elem: ET.Element) -> Optional[str]:
        val = elem.attrib.get(f"{{{RDF_NS}}}{attr_key}")
        if not val:
            return None
        if attr_key == "ID" and not val.startswith("#"):
            return f"#{val}"
        return val

    # ontology header annotations in RDF/XML
    for elem in root.iter():
        if not isinstance(elem.tag, str):
            continue
        if _namespace_uri(elem.tag) == OWL_NS and _local_name(elem.tag) == "Ontology":
            ont_ref = rdf_ref("about", elem) or rdf_ref("ID", elem) or ""
            if ont_ref and not ontology_info.ontology_iri:
                ontology_info.ontology_iri = ont_ref
            add_rdf_annotations(elem, ont_ref or "")
            for ch in _iter_non_comment_children(elem):
                if _is_annotation_tag(ch.tag):
                    ns = _namespace_uri(ch.tag)
                    lname = _local_name(ch.tag)
                    prop = f"rdfs:{lname}" if ns == RDFS_NS else f"owl:{lname}"
                    _append_annotation(ontology_info.annotations, _name_from_ref(prop), _parse_rdf_literal(ch))

    schema_type_by_ref = {
        f"{OWL_NS}Class": "class",
        f"{OWL_NS}DatatypeProperty": "data_property",
        f"{OWL_NS}ObjectProperty": "object_property",
        f"{OWL_NS}FunctionalProperty": "functional_property",
        f"{OWL_NS}AnnotationProperty": "annotation_property",
    }

    # first pass: explicit schema resources
    for elem in _iter_non_comment_children(root):
        if not isinstance(elem.tag, str):
            continue
        ns = _namespace_uri(elem.tag)
        lname = _local_name(elem.tag)

        if ns == OWL_NS and lname == "Class":
            ref = rdf_ref("about", elem) or rdf_ref("ID", elem)
            name = _name_from_ref(ref)
            if not name:
                continue
            ensure_xml_class(name)
            add_rdf_annotations(elem, ref or _class_iri(name))
            for ch in _iter_non_comment_children(elem):
                if _namespace_uri(ch.tag) == RDFS_NS and _local_name(ch.tag) == "subClassOf":
                    super_ref = rdf_ref("resource", ch) or _text(ch)
                    super_name = _name_from_ref(super_ref)
                    if super_name:
                        ensure_xml_class(super_name)
                        _append_unique(subclass_axioms, SubclassAxiomBlueprint(name, super_name))

        elif ns == OWL_NS and lname == "DatatypeProperty":
            ref = rdf_ref("about", elem) or rdf_ref("ID", elem)
            name = _name_from_ref(ref)
            if not name:
                continue
            ensure_xml_dp(name)
            add_rdf_annotations(elem, ref or _class_iri(name))
            for ch in _iter_non_comment_children(elem):
                ch_ns = _namespace_uri(ch.tag)
                ch_ln = _local_name(ch.tag)
                if ch_ns == RDFS_NS and ch_ln == "domain":
                    dom_name = _name_from_ref(rdf_ref("resource", ch) or _text(ch))
                    if dom_name:
                        ensure_xml_class(dom_name)
                        xml_data_properties[name]["domains"].add(dom_name)
                elif ch_ns == RDFS_NS and ch_ln == "range":
                    rng = rdf_ref("resource", ch) or _text(ch)
                    if rng:
                        xml_data_properties[name]["ranges"].add(rng)

        elif ns == OWL_NS and lname == "ObjectProperty":
            ref = rdf_ref("about", elem) or rdf_ref("ID", elem)
            name = _name_from_ref(ref)
            if not name:
                continue
            ensure_xml_op(name)
            add_rdf_annotations(elem, ref or _class_iri(name))
            for ch in _iter_non_comment_children(elem):
                ch_ns = _namespace_uri(ch.tag)
                ch_ln = _local_name(ch.tag)
                if ch_ns == RDFS_NS and ch_ln == "domain":
                    dom_name = _name_from_ref(rdf_ref("resource", ch) or _text(ch))
                    if dom_name:
                        ensure_xml_class(dom_name)
                        xml_object_properties[name]["domains"].add(dom_name)
                elif ch_ns == RDFS_NS and ch_ln == "range":
                    rng_name = _name_from_ref(rdf_ref("resource", ch) or _text(ch))
                    if rng_name:
                        ensure_xml_class(rng_name)
                        xml_object_properties[name]["ranges"].add(rng_name)
                elif ch_ns == OWL_NS and ch_ln == "inverseOf":
                    inv_name = _name_from_ref(rdf_ref("resource", ch) or _text(ch))
                    if inv_name:
                        xml_object_properties[name]["inverse_of"] = inv_name

    # second pass: typed resources / individuals / rdf:Description based schema
    for elem in _iter_non_comment_children(root):
        if not isinstance(elem.tag, str):
            continue
        ns = _namespace_uri(elem.tag)
        lname = _local_name(elem.tag)
        ref = rdf_ref("about", elem) or rdf_ref("ID", elem)

        if ns not in {RDF_NS, RDFS_NS, OWL_NS}:
            class_name = lname
            ind_name = _name_from_ref(ref)
            if class_name and ind_name:
                ensure_xml_class(class_name)
                class_assertions.setdefault(ind_name, [])
                if class_name not in class_assertions[ind_name]:
                    class_assertions[ind_name].append(class_name)
                add_rdf_annotations(elem, ref or _class_iri(ind_name))

        if ns == RDF_NS and lname == "Description":
            subject_ref = ref
            subject_name = _name_from_ref(subject_ref)
            typed_classes: List[str] = []
            for ch in _iter_non_comment_children(elem):
                ch_ns = _namespace_uri(ch.tag)
                ch_ln = _local_name(ch.tag)
                if ch_ns == RDF_NS and ch_ln == "type":
                    type_ref = rdf_ref("resource", ch) or _text(ch)
                    schema_kind = schema_type_by_ref.get(type_ref)
                    if schema_kind == "class":
                        ensure_xml_class(subject_name)
                    elif schema_kind == "data_property":
                        ensure_xml_dp(subject_name)
                    elif schema_kind == "object_property":
                        ensure_xml_op(subject_name)
                    elif schema_kind == "functional_property":
                        if subject_name in xml_data_properties:
                            xml_data_properties[subject_name]["is_functional"] = True
                        if subject_name in xml_object_properties:
                            xml_object_properties[subject_name]["is_functional"] = True
                    elif schema_kind == "annotation_property":
                        ensure_ann_prop(subject_ref or subject_name)
                    else:
                        cls_name = _name_from_ref(type_ref)
                        if cls_name:
                            ensure_xml_class(cls_name)
                            typed_classes.append(cls_name)
                elif _is_annotation_tag(ch.tag):
                    continue
                elif subject_name:
                    # schema details on rdf:Description for properties/classes
                    if subject_name in xml_data_properties:
                        if ch_ns == RDFS_NS and ch_ln == "domain":
                            dom_name = _name_from_ref(rdf_ref("resource", ch) or _text(ch))
                            if dom_name:
                                ensure_xml_class(dom_name)
                                xml_data_properties[subject_name]["domains"].add(dom_name)
                        elif ch_ns == RDFS_NS and ch_ln == "range":
                            rng = rdf_ref("resource", ch) or _text(ch)
                            if rng:
                                xml_data_properties[subject_name]["ranges"].add(rng)
                    if subject_name in xml_object_properties:
                        if ch_ns == RDFS_NS and ch_ln == "domain":
                            dom_name = _name_from_ref(rdf_ref("resource", ch) or _text(ch))
                            if dom_name:
                                ensure_xml_class(dom_name)
                                xml_object_properties[subject_name]["domains"].add(dom_name)
                        elif ch_ns == RDFS_NS and ch_ln == "range":
                            rng_name = _name_from_ref(rdf_ref("resource", ch) or _text(ch))
                            if rng_name:
                                ensure_xml_class(rng_name)
                                xml_object_properties[subject_name]["ranges"].add(rng_name)
                        elif ch_ns == OWL_NS and ch_ln == "inverseOf":
                            inv_name = _name_from_ref(rdf_ref("resource", ch) or _text(ch))
                            if inv_name:
                                xml_object_properties[subject_name]["inverse_of"] = inv_name
                    if subject_name in xml_classes and ch_ns == RDFS_NS and ch_ln == "subClassOf":
                        super_name = _name_from_ref(rdf_ref("resource", ch) or _text(ch))
                        if super_name:
                            ensure_xml_class(super_name)
                            _append_unique(subclass_axioms, SubclassAxiomBlueprint(subject_name, super_name))

            if typed_classes and subject_name:
                class_assertions.setdefault(subject_name, [])
                for cls_name in typed_classes:
                    if cls_name not in class_assertions[subject_name]:
                        class_assertions[subject_name].append(cls_name)
                add_rdf_annotations(elem, subject_ref or _class_iri(subject_name))

    # infer classes from property declarations and class assertions
    for dp in xml_data_properties.values():
        for dom in list(dp["domains"]):
            ensure_xml_class(dom)
    for op in xml_object_properties.values():
        for dom in list(op["domains"]):
            ensure_xml_class(dom)
        for rng in list(op["ranges"]):
            ensure_xml_class(rng)
    for ind_name, asserted in class_assertions.items():
        for cls_name in asserted:
            ensure_xml_class(cls_name)
            ind_ref = f"#{ind_name}" if not ind_name.startswith("#") else ind_name
            add_rdf_annotations(root.find("."), ind_ref)  # no-op, keeps target_annotations structure untouched

    return _XmlExtraction(
        ontology_info=ontology_info,
        annotation_properties=annotation_properties,
        target_annotations=target_annotations,
        subclass_axioms=subclass_axioms,
        equivalent_class_axioms=equivalent_class_axioms,
        disjoint_class_axioms=disjoint_class_axioms,
        class_assertions=class_assertions,
        restriction_axioms=restriction_axioms,
        xml_classes=xml_classes,
        xml_data_properties=xml_data_properties,
        xml_object_properties=xml_object_properties,
    )


def _collect_cardinality_maps(model: Any):
    axioms = getattr(model, "object_property_cardinality_axioms", None) or []
    forward = {}
    inverse = {}
    for domain, prop, kind, n, is_inverse in axioms:
        target = inverse if is_inverse else forward
        key = (str(domain), str(prop))
        mn, mx = target.get(key, (None, None))
        if kind == "min":
            mn = n
        elif kind == "max":
            mx = n
        elif kind == "exact":
            mn, mx = n, n
        target[key] = (mn, mx)
    return forward, inverse


def build_ontology_blueprint(model: Any, overrides: MappingOverrides, cfg: MappingConfig, *,
                             owl_path: Optional[str] = None) -> OntologyBlueprint:
    bp = OntologyBlueprint()

    xml = _extract_xml_metadata(owl_path) if owl_path else _XmlExtraction(
        ontology_info=OntologyInfoBlueprint(),
        annotation_properties={},
        target_annotations={},
        subclass_axioms=[],
        equivalent_class_axioms=[],
        disjoint_class_axioms=[],
        class_assertions={},
        restriction_axioms=[],
        xml_classes={},
        xml_data_properties={},
        xml_object_properties={},
    )
    bp.ontology_info = xml.ontology_info
    bp.annotation_properties = xml.annotation_properties
    bp.axioms.subclass_axioms = list(xml.subclass_axioms)
    bp.axioms.equivalent_class_axioms = list(xml.equivalent_class_axioms)
    bp.axioms.disjoint_class_axioms = list(xml.disjoint_class_axioms)
    bp.axioms.restriction_axioms = list(xml.restriction_axioms)

    classes: Dict[str, Any] = getattr(model, "classes", {}) or {}
    data_props: Dict[str, Any] = getattr(model, "data_properties", {}) or {}
    object_props: Dict[str, Any] = getattr(model, "object_properties", {}) or {}

    forward_card_map, inverse_card_map = _collect_cardinality_maps(model)
    functional_dps = set(getattr(model, "functional_data_properties", []) or [])
    functional_ops = set(getattr(model, "functional_object_properties", []) or [])

    # classes: union of model + XML classes
    class_names: Set[str] = set(classes.keys()) | set(xml.xml_classes.keys())
    for class_name in sorted(class_names):
        if cfg.ignore_owl_thing and class_name in overrides.ignore_classes:
            continue
        cls_obj = classes.get(class_name)
        xml_cls = xml.xml_classes.get(class_name, {})
        iri = getattr(cls_obj, "iri", xml_cls.get("iri") or _class_iri(class_name))
        anns = xml.target_annotations.get(iri, {}) or xml.target_annotations.get(_class_iri(class_name), {})
        bp.classes[class_name] = ClassBlueprint(
            name=class_name,
            iri=iri,
            annotations=anns,
        )

    # named individuals from explicit class assertions (preferred source)
    for ind_name in sorted(xml.class_assertions.keys()):
        ind_iri = f"#{ind_name}" if not ind_name.startswith("#") else ind_name
        bp.named_individuals[ind_name] = NamedIndividualBlueprint(
            name=ind_name,
            iri=ind_iri,
            asserted_types=sorted(xml.class_assertions[ind_name]),
            annotations=xml.target_annotations.get(ind_iri, {}),
        )

    # fallback: if model attached individuals to classes but XML assertions somehow absent
    for class_name in sorted(classes.keys()):
        if cfg.ignore_owl_thing and class_name in overrides.ignore_classes:
            continue
        cls_obj = classes[class_name]
        for ind_name in sorted(_to_str_list(getattr(cls_obj, "individuals", []))):
            ind_iri = f"#{ind_name}"
            if ind_name not in bp.named_individuals:
                bp.named_individuals[ind_name] = NamedIndividualBlueprint(
                    name=ind_name,
                    iri=ind_iri,
                    asserted_types=[class_name],
                    annotations=xml.target_annotations.get(ind_iri, {}),
                )
            else:
                _append_unique(bp.named_individuals[ind_name].asserted_types, class_name)

    # data properties: merge model + XML
    data_prop_names: Set[str] = set(data_props.keys()) | set(xml.xml_data_properties.keys())
    for prop_name in sorted(data_prop_names):
        if prop_name in overrides.ignore_properties:
            continue
        dp = data_props.get(prop_name)
        xdp = xml.xml_data_properties.get(prop_name, {})
        domains = sorted(set(_to_str_list(getattr(dp, "domains", []))) | set(xdp.get("domains", set())))
        ranges = sorted(set(_to_str_list(getattr(dp, "range_datatype", None))) | set(xdp.get("ranges", set())))
        card_by_domain: Dict[str, Dict[str, Optional[int]]] = {}
        raw_cards = getattr(dp, "cardinality_by_domain", None) or {}
        for dom, value in raw_cards.items():
            if isinstance(value, dict):
                card_by_domain[str(dom)] = {"min": value.get("min"), "max": value.get("max")}
            elif isinstance(value, tuple) and len(value) == 2:
                card_by_domain[str(dom)] = {"min": value[0], "max": value[1]}
        iri = getattr(dp, "iri", xdp.get("iri") or f"#{prop_name}")
        bp.data_properties[prop_name] = DataPropertyBlueprint(
            name=prop_name,
            iri=iri,
            domains=domains,
            ranges=ranges,
            is_functional=(prop_name in functional_dps) or bool(xdp.get("is_functional")),
            cardinality_by_domain=card_by_domain,
            annotations=xml.target_annotations.get(iri, {}),
        )

    # object properties: merge model + XML
    object_prop_names: Set[str] = set(object_props.keys()) | set(xml.xml_object_properties.keys())
    for prop_name in sorted(object_prop_names):
        if prop_name in overrides.ignore_properties:
            continue
        op = object_props.get(prop_name)
        xop = xml.xml_object_properties.get(prop_name, {})
        domains = sorted(set(_to_str_list(getattr(op, "domains", []))) | set(xop.get("domains", set())))
        ranges = sorted(set(_to_str_list(getattr(op, "ranges", []))) | set(xop.get("ranges", set())))
        inverse_of = getattr(op, "inverse_of", None) or xop.get("inverse_of")
        forward_cardinality_by_domain: Dict[str, Dict[str, Optional[int]]] = {}
        inverse_cardinality_by_domain: Dict[str, Dict[str, Optional[int]]] = {}
        for dom in domains:
            mn, mx = forward_card_map.get((dom, prop_name), (None, None))
            if mn is not None or mx is not None:
                forward_cardinality_by_domain[dom] = _normalize_min_max(mn, mx)
        for dom in (set(classes.keys()) | set(xml.xml_classes.keys())):
            mn, mx = inverse_card_map.get((dom, prop_name), (None, None))
            if mn is not None or mx is not None:
                inverse_cardinality_by_domain[dom] = _normalize_min_max(mn, mx)
        iri = getattr(op, "iri", xop.get("iri") or f"#{prop_name}")
        bp.object_properties[prop_name] = ObjectPropertyBlueprint(
            name=prop_name,
            iri=iri,
            domains=domains,
            ranges=ranges,
            inverse_of=inverse_of,
            is_functional=(prop_name in functional_ops) or bool(xop.get("is_functional")),
            forward_cardinality_by_domain=forward_cardinality_by_domain,
            inverse_cardinality_by_domain=inverse_cardinality_by_domain,
            annotations=xml.target_annotations.get(iri, {}),
        )

    bp.notes.extend([
        "Pure ontology-oriented blueprint extracted from OWL.",
        "No Mongo realization hints or derived implementation statuses are included.",
        "Ontology header information and annotation literal metadata (lang/datatype) are preserved when available.",
        "Subclass / equivalent / disjoint axioms are parsed directly from OWL XML.",
        "Named individuals store asserted types directly; no separate class_assertion_axioms section is used.",
        "RDF/XML schema extraction is merged with parser model output so schema-only and schema+individual files are both supported.",
    ])
    return bp


def ontology_blueprint_to_dict(bp: OntologyBlueprint) -> Dict[str, Any]:
    return asdict(bp)


def ontology_blueprint_to_json(bp: OntologyBlueprint) -> str:
    import json
    return json.dumps(ontology_blueprint_to_dict(bp), indent=2, ensure_ascii=False)


def pretty_print_ontology_blueprint(bp: OntologyBlueprint) -> str:
    lines: List[str] = []
    lines.append("=== Pure Ontology Blueprint Summary ===")
    lines.append(f"Ontology IRI: {bp.ontology_info.ontology_iri}")
    lines.append(f"XML Base: {bp.ontology_info.xml_base}")
    lines.append(f"Prefixes: {len(bp.ontology_info.prefixes)}")
    lines.append(f"Namespaces: {len(bp.ontology_info.namespaces)}")
    lines.append(f"Classes: {len(bp.classes)}")
    lines.append(f"Data properties: {len(bp.data_properties)}")
    lines.append(f"Object properties: {len(bp.object_properties)}")
    lines.append(f"Annotation properties: {len(bp.annotation_properties)}")
    lines.append(f"Named individuals: {len(bp.named_individuals)}")
    lines.append(f"Subclass axioms: {len(bp.axioms.subclass_axioms)}")
    lines.append(f"Equivalent class axioms: {len(bp.axioms.equivalent_class_axioms)}")
    lines.append(f"Disjoint class axioms: {len(bp.axioms.disjoint_class_axioms)}")
    lines.append(f"Restriction axioms: {len(bp.axioms.restriction_axioms)}")
    lines.append("")
    lines.append("Example classes:")
    for class_name in sorted(bp.classes.keys())[:10]:
        c = bp.classes[class_name]
        lines.append(f"- {class_name}: annotations={list(c.annotations.keys())}")
    lines.append("")
    lines.append("Example named individuals:")
    for ind_name in sorted(bp.named_individuals.keys())[:10]:
        ind = bp.named_individuals[ind_name]
        lines.append(f"- {ind_name}: asserted_types={ind.asserted_types}, annotations={list(ind.annotations.keys())}")
    lines.append("")
    lines.append("Example object properties:")
    for prop_name in sorted(bp.object_properties.keys())[:10]:
        p = bp.object_properties[prop_name]
        lines.append(
            f"- {prop_name}: domains={p.domains}, ranges={p.ranges}, inverse_of={p.inverse_of}, functional={p.is_functional}")
    lines.append("")
    lines.append("Annotation properties:")
    for ap_name in sorted(bp.annotation_properties.keys()):
        ap = bp.annotation_properties[ap_name]
        lines.append(f"- {ap_name}: iri={ap.iri}")
    lines.append("")
    lines.append("Notes:")
    for note in bp.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)
