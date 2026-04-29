from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, Tuple


@dataclass
class MappingOverrides:
    """
    User overrides for mapping decisions.

    Keys are (domain_class_name, object_property_name).
    """

    # legacy physical overrides (still kept for compatibility)
    force_embed: Dict[Tuple[str, str], bool] = field(default_factory=dict)
    force_reference: Dict[Tuple[str, str], bool] = field(default_factory=dict)
    force_association: Dict[Tuple[str, str], bool] = field(default_factory=dict)

    # new semantic overrides
    force_strong: Dict[Tuple[str, str], bool] = field(default_factory=dict)
    force_weak: Dict[Tuple[str, str], bool] = field(default_factory=dict)
    force_required: Dict[Tuple[str, str], bool] = field(default_factory=dict)
    force_optional: Dict[Tuple[str, str], bool] = field(default_factory=dict)
    force_default_expand: Dict[Tuple[str, str], bool] = field(default_factory=dict)
    force_cascade_delete: Dict[Tuple[str, str], bool] = field(default_factory=dict)
    force_ownership: Dict[Tuple[str, str], bool] = field(default_factory=dict)

    ignore_classes: Set[str] = field(default_factory=set)
    ignore_properties: Set[str] = field(default_factory=set)

    collection_name_overrides: Dict[str, str] = field(default_factory=dict)


@dataclass
class MappingConfig:
    # naming / normalization
    improved_naming: bool = True
    canonicalize_inverse: bool = True
    ignore_owl_thing: bool = True

    # keep for compatibility, but in the new strategy all OWL object relations become references
    default_relation: str = "reference"

    # if True: FunctionalObjectProperty implies max=1 instead of array
    functional_implies_single: bool = True

    # -------- new storage policy --------
    # all OWL object properties are stored as references
    object_relation_mode: str = "reference_only"

    # -------- semantic defaults --------
    default_strength: str = "weak"  # "strong" | "weak"
    default_required: bool = False
    default_ownership: bool = False
    default_allow_orphan: bool = True
    default_cascade_delete: bool = False
    default_expand: bool = False

    # -------- heuristic knobs (semantic only, not storage) --------
    # if a target class is referenced by >= this many domain occurrences,
    # we consider the relation shared -> weak
    shared_threshold: int = 2

    # self reference is weak by default
    self_reference_is_weak: bool = True

    # legacy field retained only so old caller code does not break;
    # it no longer decides embed/reference
    max_embed_depth: int | None = None
