from __future__ import annotations

import argparse
from pathlib import Path

from mapping.mapping_rules import MappingConfig, MappingOverrides
from mapping.ontology_blueprint import (
    build_ontology_blueprint,
    ontology_blueprint_to_json,
    pretty_print_ontology_blueprint,
)
from ontology.owl_parser import parse_owl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pure ontology-oriented blueprint from OWL.")
    parser.add_argument("owl_path", type=str, help="Path to OWL file")
    parser.add_argument("--ontology_out", type=str, default="ontology_blueprint.json",
                        help="Output path for pure ontology blueprint")
    args = parser.parse_args()

    owl_path = Path(args.owl_path)
    if not owl_path.exists():
        raise FileNotFoundError(f"OWL file not found: {owl_path}")

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

    model = parse_owl(str(owl_path))
    ontology_bp = build_ontology_blueprint(model, overrides=overrides, cfg=cfg, owl_path=str(owl_path))

    print("########################################")
    print("### Pure Ontology Blueprint          ###")
    print("########################################")
    print(pretty_print_ontology_blueprint(ontology_bp))

    Path(args.ontology_out).write_text(ontology_blueprint_to_json(ontology_bp), encoding="utf-8")
    print(f"\nWrote ontology blueprint to: {Path(args.ontology_out).resolve()}")


if __name__ == "__main__":
    main()
