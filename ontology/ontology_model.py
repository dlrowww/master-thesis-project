# ontology/ontology_model.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class Cardinality:
    """
    Cardinality constraint used for (Data/Object)Property restrictions.

    Fields:
      - min_c: minimum cardinality (None means not specified)
      - max_c: maximum cardinality (None means unbounded/not specified)
    """
    min_c: Optional[int] = None
    max_c: Optional[int] = None

    def is_single(self) -> bool:
        """Return True if max cardinality is exactly 1."""
        return self.max_c == 1

    def is_many(self) -> bool:
        """Return True if max is None (unbounded) or > 1."""
        return self.max_c is None or self.max_c > 1


@dataclass
class ClassDef:
    """
    Represents an OWL Class.

    Fields:
      - name: local name of the class (e.g., 'Student')
      - iri: full IRI if available (optional)
      - annotations: annotations attached to this class (ap -> list of values)
    """
    name: str
    iri: Optional[str] = None
    annotations: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class DataPropertyDef:
    """
    Represents an OWL DataProperty.

    Fields:
      - name: local name of the data property (e.g., 'hasValue')
      - iri: full IRI if available (optional)
      - domains: set of domain class names
      - range_datatype: XSD datatype short name or IRI (e.g., 'xsd:string')
      - cardinality_by_domain: optional cardinality constraints keyed by domain class name
      - annotations: annotations attached to this property (ap -> list of values)
    """
    name: str
    iri: Optional[str] = None
    domains: Set[str] = field(default_factory=set)
    range_datatype: Optional[str] = None
    cardinality_by_domain: Dict[str, Cardinality] = field(default_factory=dict)
    annotations: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class ObjectPropertyDef:
    """
    Represents an OWL ObjectProperty.

    Fields:
      - name: local name of the object property (e.g., 'hasCourse')
      - iri: full IRI if available (optional)
      - domains: set of domain class names
      - ranges: set of range class names
      - inverse_of: inverse property name if specified (optional)
      - cardinality_by_domain: optional cardinality constraints keyed by domain class name
      - annotations: annotations attached to this property (ap -> list of values)
    """
    name: str
    iri: Optional[str] = None
    domains: Set[str] = field(default_factory=set)
    ranges: Set[str] = field(default_factory=set)
    inverse_of: Optional[str] = None
    cardinality_by_domain: Dict[str, Cardinality] = field(default_factory=dict)
    annotations: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class AnnotationPropertyDef:
    """
    Represents an OWL AnnotationProperty.

    Fields:
      - name: local name of the annotation property (e.g., 'label', 'comment')
      - iri: full IRI if available (optional)
    """
    name: str
    iri: Optional[str] = None


@dataclass
class OntologyModel:
    """
    Unified in-memory representation of the ontology, produced by the parser and consumed by mapping.

    Core fields:
      - classes: class name -> ClassDef
      - data_properties: property name -> DataPropertyDef
      - object_properties: property name -> ObjectPropertyDef
      - annotation_properties: property name -> AnnotationPropertyDef
      - namespace_map: prefix -> IRI map
      - base_iri: optional xml:base / ontologyIRI
      - source_path: OWL file path

    Step 3.5 fields (used by schema builder / mapping):
      - functional_object_properties: set of object property names that are FunctionalObjectProperty
      - functional_data_properties: set of data property names that are FunctionalDataProperty
      - object_property_cardinality_axioms:
            list of tuples (domain_class, property_name, kind, n, is_inverse)
            kind: "min" | "max" | "exact"
            is_inverse: True if restriction uses ObjectInverseOf(...)
    """
    classes: Dict[str, ClassDef] = field(default_factory=dict)
    data_properties: Dict[str, DataPropertyDef] = field(default_factory=dict)
    object_properties: Dict[str, ObjectPropertyDef] = field(default_factory=dict)
    annotation_properties: Dict[str, AnnotationPropertyDef] = field(default_factory=dict)

    namespace_map: Dict[str, str] = field(default_factory=dict)
    base_iri: Optional[str] = None
    source_path: Optional[str] = None

    functional_object_properties: Set[str] = field(default_factory=set)
    functional_data_properties: Set[str] = field(default_factory=set)

    object_property_cardinality_axioms: List[Tuple[str, str, str, int, bool]] = field(default_factory=list)

    # Convenience helpers
    def list_class_names(self) -> List[str]:
        return sorted(self.classes.keys())

    def list_data_property_names(self) -> List[str]:
        return sorted(self.data_properties.keys())

    def list_object_property_names(self) -> List[str]:
        return sorted(self.object_properties.keys())

    def list_annotation_property_names(self) -> List[str]:
        return sorted(self.annotation_properties.keys())
