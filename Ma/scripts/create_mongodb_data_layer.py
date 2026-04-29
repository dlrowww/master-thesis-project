from __future__ import annotations

"""
Unified MongoDB data-layer builder with iterative dependency-aware import
+ skip diagnostics
+ imported-reference placeholders.

Key improvements:
- instance import runs in multiple rounds until no more documents can be
  inserted or updated with newly resolved objectId references
- unresolved references that point to imported ontologies loaded only by
  owl:imports are preserved as placeholders instead of being silently lost
- an _import_diagnostics block is written into affected documents

Notes:
- placeholders are stored only for unresolved *external imported* object
  references; they do not satisfy required MongoDB validator fields
- therefore required structural references still need real ObjectIds
"""

import argparse
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import xml.etree.ElementTree as ET

from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import CollectionInvalid, OperationFailure


# --------------------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------------------

def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def camel_to_snake(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.replace("__", "_").lower()


def pluralize_snake(snake: str) -> str:
    if snake.endswith("y") and len(snake) > 1 and snake[-2] not in "aeiou":
        return snake[:-1] + "ies"
    if snake.endswith("sis"):
        return snake[:-2] + "es"
    if snake.endswith(("s", "x", "z", "ch", "sh")):
        return snake + "es"
    return snake + "s"


def class_to_collection_candidates(class_name: str) -> List[str]:
    snake = camel_to_snake(class_name)
    candidates = [pluralize_snake(snake)]
    if snake.endswith("_series"):
        candidates.append(snake + "es")
    if snake.endswith("data"):
        candidates.append(snake + "s")
    if snake.endswith("name"):
        candidates.append(pluralize_snake(snake))
    out: List[str] = []
    seen = set()
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def iri_to_local_name(iri: str) -> str:
    if "#" in iri:
        return iri.rsplit("#", 1)[-1]
    if "/" in iri:
        return iri.rstrip("/").rsplit("/", 1)[-1]
    return iri


def iri_without_fragment(iri: str) -> str:
    if "#" in iri:
        return iri.split("#", 1)[0]
    return iri.rstrip("/")


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def ensure_list_dict(d: Dict[str, List[Any]], key: str) -> List[Any]:
    if key not in d:
        d[key] = []
    return d[key]


def parse_bool(text: str) -> Optional[bool]:
    v = text.strip().lower()
    if v in {"true", "1"}:
        return True
    if v in {"false", "0"}:
        return False
    return None


def parse_datetime_value(text: str) -> Optional[datetime]:
    v = text.strip()
    if not v:
        return None
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(v)
    except Exception:
        pass
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(v, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return None


def parse_scalar_by_bson(value: Any, bson_type: Optional[str]) -> Any:
    if bson_type is None or value is None:
        return value

    if bson_type == "string":
        return str(value)

    if bson_type == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            parsed = parse_bool(value)
            return parsed if parsed is not None else value
        return bool(value)

    if bson_type in {"double", "decimal"}:
        try:
            return float(value)
        except Exception:
            return value

    if bson_type in {"int", "long"}:
        try:
            return int(float(value))
        except Exception:
            return value

    if bson_type == "date":
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            parsed = parse_datetime_value(value)
            return parsed if parsed is not None else value
        return value

    return value


def dedupe_dict_list(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for item in items:
        key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


# --------------------------------------------------------------------------------------
# Validator/schema inspection
# --------------------------------------------------------------------------------------

@dataclass
class FieldSpec:
    name: str
    bson_type: Optional[str]
    is_array: bool
    is_reference: bool
    required: bool


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_collection_field_specs(validators: Dict[str, Any]) -> Dict[str, Dict[str, FieldSpec]]:
    result: Dict[str, Dict[str, FieldSpec]] = {}
    for coll_name, validator in validators.items():
        schema = (validator or {}).get("$jsonSchema", {})
        props = schema.get("properties", {}) or {}
        required = set(schema.get("required", []) or [])
        specs: Dict[str, FieldSpec] = {}
        for field_name, spec in props.items():
            bson_type = spec.get("bsonType")
            is_array = bson_type == "array"
            item_bson = (spec.get("items") or {}).get("bsonType") if is_array else None
            is_reference = bson_type == "objectId" or item_bson == "objectId"
            specs[field_name] = FieldSpec(
                name=field_name,
                bson_type=item_bson if is_array and not is_reference else bson_type,
                is_array=is_array,
                is_reference=is_reference,
                required=field_name in required,
            )
        result[coll_name] = specs
    return result


# --------------------------------------------------------------------------------------
# Ontology helpers
# --------------------------------------------------------------------------------------

def class_is_abstract(ontology_bp: Dict[str, Any], class_name: str) -> bool:
    c = (ontology_bp.get("classes") or {}).get(class_name) or {}
    anns = c.get("annotations") or {}
    ct = anns.get("conceptType") or []
    for entry in ct:
        if str((entry or {}).get("value", "")).strip().lower() == "abstract":
            return True
    return False


def build_class_to_collection(validators: Dict[str, Any], ontology_bp: Dict[str, Any]) -> Dict[str, str]:
    available = set(validators.keys())
    mapping: Dict[str, str] = {}
    for class_name in (ontology_bp.get("classes") or {}).keys():
        for cand in class_to_collection_candidates(class_name):
            if cand in available:
                mapping[class_name] = cand
                break
    return mapping


def _build_parent_child_maps(ontology_bp: Dict[str, Any]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Build subclass hierarchy maps from ontology blueprint axioms.

    Supports both possible key styles seen in local project variants:
      - {"subclass": "...", "superclass": "..."}
      - {"sub_class": "...", "super_class": "..."}
    """
    parents: Dict[str, Set[str]] = {}
    children: Dict[str, Set[str]] = {}

    for cls_name in (ontology_bp.get("classes") or {}).keys():
        parents.setdefault(cls_name, set())
        children.setdefault(cls_name, set())

    axioms = (ontology_bp.get("axioms") or {}).get("subclass_axioms", []) or []
    for ax in axioms:
        sub = ax.get("subclass") or ax.get("sub_class")
        sup = ax.get("superclass") or ax.get("super_class")
        if not sub or not sup:
            continue

        parents.setdefault(sub, set()).add(sup)
        children.setdefault(sup, set()).add(sub)
        parents.setdefault(sup, set())
        children.setdefault(sub, set())

    return parents, children


def _collect_domain_and_descendants(
        domain: str,
        ontology_bp: Dict[str, Any],
        children_map: Dict[str, Set[str]],
        class_to_collection: Dict[str, str],
) -> List[str]:
    """
    Return the given domain plus all descendant classes that are routable
    to Mongo collections in the current blueprint/validator setup.
    """
    out: List[str] = []
    seen: Set[str] = set()
    stack: List[str] = [domain]

    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)

        if current in class_to_collection:
            out.append(current)

        for child in children_map.get(current, set()):
            if child not in seen:
                stack.append(child)

    return out


def build_prop_maps(
        validators: Dict[str, Any],
        ontology_bp: Dict[str, Any],
        class_to_collection: Dict[str, str],
) -> Tuple[Dict[Tuple[str, str], str], Dict[Tuple[str, str], str]]:
    """
    Build property-to-field maps for instance import.

    IMPORTANT:
    We must propagate properties declared on a parent class domain
    to all concrete/routable descendant classes as well.

    Example:
      - forCompany domain = Job
      - ITJob subclassOf Job
      => both (Job, forCompany) and (ITJob, forCompany) should map
         to for_company_id when the field exists in the target collection.
    """
    field_specs_by_coll = get_collection_field_specs(validators)
    data_map: Dict[Tuple[str, str], str] = {}
    obj_map: Dict[Tuple[str, str], str] = {}

    _, children_map = _build_parent_child_maps(ontology_bp)

    # -------------------------
    # Data properties
    # -------------------------
    for prop_name, prop in (ontology_bp.get("data_properties") or {}).items():
        prop = prop or {}
        prop_field = camel_to_snake(prop_name)

        for domain in prop.get("domains") or []:
            effective_domains = _collect_domain_and_descendants(
                domain=domain,
                ontology_bp=ontology_bp,
                children_map=children_map,
                class_to_collection=class_to_collection,
            )

            for eff_domain in effective_domains:
                coll = class_to_collection.get(eff_domain)
                if not coll:
                    continue

                specs = field_specs_by_coll.get(coll, {})
                if prop_field in specs:
                    data_map[(eff_domain, prop_name)] = prop_field

    # -------------------------
    # Object properties
    # -------------------------
    for prop_name, prop in (ontology_bp.get("object_properties") or {}).items():
        prop = prop or {}
        base = camel_to_snake(prop_name)

        for domain in prop.get("domains") or []:
            effective_domains = _collect_domain_and_descendants(
                domain=domain,
                ontology_bp=ontology_bp,
                children_map=children_map,
                class_to_collection=class_to_collection,
            )

            for eff_domain in effective_domains:
                coll = class_to_collection.get(eff_domain)
                if not coll:
                    continue

                specs = field_specs_by_coll.get(coll, {})
                singular = f"{base}_id"
                plural = f"{base}_ids"

                if singular in specs:
                    obj_map[(eff_domain, prop_name)] = singular
                elif plural in specs:
                    obj_map[(eff_domain, prop_name)] = plural

    return data_map, obj_map


def choose_concrete_type(
        ontology_bp: Dict[str, Any],
        asserted_types: List[str],
        class_to_collection: Dict[str, str],
) -> Optional[str]:
    if not asserted_types:
        return None

    concrete = [t for t in asserted_types if t in class_to_collection and not class_is_abstract(ontology_bp, t)]
    if concrete:
        concrete.sort(key=lambda x: (x.count("_"), len(x)), reverse=True)
        return concrete[0]

    routed = [t for t in asserted_types if t in class_to_collection]
    return routed[0] if routed else None


def get_current_ontology_bases(ontology_bp: Dict[str, Any]) -> Set[str]:
    info = ontology_bp.get("ontology_info") or {}
    bases: Set[str] = set()
    for raw in [info.get("ontology_iri"), info.get("xml_base")]:
        if raw:
            bases.add(iri_without_fragment(str(raw)))
    for raw in (info.get("prefixes") or {}).values():
        if raw:
            bases.add(iri_without_fragment(str(raw)))
    return {b for b in bases if b}


# --------------------------------------------------------------------------------------
# Optional OWL instance parser (shallow fallback only)
# --------------------------------------------------------------------------------------

RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"
OWL_NS = "http://www.w3.org/2002/07/owl#"


def split_tag(tag: str) -> Tuple[Optional[str], str]:
    if tag.startswith("{"):
        ns, local = tag[1:].split("}", 1)
        return ns, local
    return None, tag


def normalize_resource_value(raw: str) -> str:
    return raw.strip()


def parse_owl_imports(path: str | Path) -> List[str]:
    tree = ET.parse(str(path))
    root = tree.getroot()
    imports: List[str] = []
    for elem in root.iter():
        ns, local = split_tag(elem.tag)
        if ns == OWL_NS and local == "imports":
            raw = elem.attrib.get(f"{{{RDF_NS}}}resource") or elem.attrib.get("resource")
            if raw:
                raw = raw.strip()
                if raw not in imports:
                    imports.append(raw)
    return imports


def parse_mixed_owl_instances(path: str | Path) -> Dict[str, Dict[str, Any]]:
    tree = ET.parse(str(path))
    root = tree.getroot()

    instances: Dict[str, Dict[str, Any]] = {}

    for elem in root:
        ns, local = split_tag(elem.tag)

        if ns in {OWL_NS, RDFS_NS} and local in {
            "Ontology",
            "Class",
            "ObjectProperty",
            "DatatypeProperty",
            "AnnotationProperty",
            "NamedIndividual",
        }:
            continue

        iri = elem.attrib.get(f"{{{RDF_NS}}}about") or elem.attrib.get(f"{{{RDF_NS}}}ID")
        if not iri:
            continue
        iri = normalize_resource_value(iri)
        name = iri[1:] if iri.startswith("#") else iri_to_local_name(iri)

        entry = instances.setdefault(
            name,
            {
                "name": name,
                "iri": iri if iri.startswith("http") or iri.startswith("#") else f"#{iri}",
                "asserted_types": [],
                "annotations": {},
                "data_assertions": {},
                "object_assertions": {},
            },
        )

        if not (ns == RDF_NS and local == "Description") and ns not in {OWL_NS, RDFS_NS, RDF_NS}:
            if local not in entry["asserted_types"]:
                entry["asserted_types"].append(local)

        for child in elem:
            cns, clocal = split_tag(child.tag)
            text = (child.text or "").strip()
            resource = child.attrib.get(f"{{{RDF_NS}}}resource")

            if cns == RDF_NS and clocal == "type" and resource:
                t = iri_to_local_name(resource)
                if t not in entry["asserted_types"]:
                    entry["asserted_types"].append(t)
                continue

            if cns == RDFS_NS and clocal in {"label", "comment"}:
                ann_key = f"rdfs:{clocal}"
                ensure_list_dict(entry["annotations"], ann_key).append(
                    {
                        "value": text,
                        "lang": child.attrib.get("{http://www.w3.org/XML/1998/namespace}lang"),
                        "datatype": None,
                        "value_type": "literal",
                    }
                )
                continue

            if resource:
                ensure_list_dict(entry["object_assertions"], clocal).append(normalize_resource_value(resource))
            elif text:
                ensure_list_dict(entry["data_assertions"], clocal).append(text)

    return instances


def merge_owl_assertions_into_blueprint(
        ontology_bp: Dict[str, Any], owl_instances: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    bp = deepcopy(ontology_bp)
    named = bp.setdefault("named_individuals", {})

    for name, inst in owl_instances.items():
        if name not in named:
            named[name] = inst
            continue

        tgt = named[name]
        tgt.setdefault("name", inst.get("name", name))
        tgt.setdefault("iri", inst.get("iri", f"#{name}"))
        tgt.setdefault("asserted_types", [])
        tgt.setdefault("annotations", {})
        tgt.setdefault("data_assertions", {})
        tgt.setdefault("object_assertions", {})

        for t in inst.get("asserted_types", []):
            if t not in tgt["asserted_types"]:
                tgt["asserted_types"].append(t)

        for k, values in (inst.get("annotations") or {}).items():
            ensure_list_dict(tgt["annotations"], k).extend(values)

        for k, values in (inst.get("data_assertions") or {}).items():
            ensure_list_dict(tgt["data_assertions"], k).extend(values)

        for k, values in (inst.get("object_assertions") or {}).items():
            ensure_list_dict(tgt["object_assertions"], k).extend(values)

    return bp


def match_import_source(raw_target: str, imported_iris: List[str], current_bases: Set[str]) -> Optional[str]:
    if not isinstance(raw_target, str) or not raw_target.startswith(("http://", "https://")):
        return None

    normalized_target = iri_without_fragment(raw_target)
    if any(normalized_target.startswith(base) for base in current_bases):
        return None

    for imp in imported_iris:
        normalized_import = iri_without_fragment(imp)
        if normalized_target.startswith(normalized_import):
            return imp

    # If we know it's external but not matched by an explicit import line,
    # preserve the normalized target namespace as a best-effort import source.
    return normalized_target


def unresolved_field_name_for_reference(field_name: str) -> str:
    if field_name.endswith("_ids"):
        return field_name[:-4] + "_unresolved"
    if field_name.endswith("_id"):
        return field_name[:-3] + "_unresolved"
    return field_name + "_unresolved"


def set_unresolved_reference_payload(
        doc: Dict[str, Any],
        unresolved_field: str,
        diagnostics_entries: List[Dict[str, Any]],
        unresolved_entries: List[Dict[str, Any]],
) -> None:
    if unresolved_entries:
        doc[unresolved_field] = dedupe_dict_list(unresolved_entries)
    else:
        doc.pop(unresolved_field, None)

    block = doc.setdefault("_import_diagnostics", {})
    existing = block.get("unresolved_import_references", []) or []
    keep = [entry for entry in existing if entry.get("field") != unresolved_field]
    combined = dedupe_dict_list(keep + diagnostics_entries)

    if combined:
        block["unresolved_import_references"] = combined
    else:
        block.pop("unresolved_import_references", None)

    if not block:
        doc.pop("_import_diagnostics", None)


# --------------------------------------------------------------------------------------
# Mongo creation / update
# --------------------------------------------------------------------------------------

def ensure_collection(db: Database, name: str, validator_doc: Dict[str, Any]) -> None:
    try:
        db.create_collection(name, validator=validator_doc)
    except CollectionInvalid:
        try:
            db.command("collMod", name, validator=validator_doc)
        except OperationFailure:
            pass


def maybe_drop_collections(db: Database, validators: Dict[str, Any], drop_collections: bool) -> None:
    if not drop_collections:
        return
    for coll_name in validators.keys():
        if coll_name in db.list_collection_names():
            db[coll_name].drop()
    if "_ontology_meta" in db.list_collection_names():
        db["_ontology_meta"].drop()


def write_ontology_meta(db: Database, ontology_bp: Dict[str, Any], validators: Dict[str, Any]) -> None:
    meta = db["_ontology_meta"]
    meta.delete_many({})
    meta.insert_one(
        {
            "createdAt": utcnow(),
            "ontology_info": ontology_bp.get("ontology_info", {}),
            "class_count": len(ontology_bp.get("classes", {})),
            "data_property_count": len(ontology_bp.get("data_properties", {})),
            "object_property_count": len(ontology_bp.get("object_properties", {})),
            "named_individual_count": len(ontology_bp.get("named_individuals", {})),
            "collections": sorted(validators.keys()),
            "notes": "Generated by create_mongodb_data_layer_with_import_placeholders.py",
        }
    )


def choose_identity_query(doc: Dict[str, Any]) -> Dict[str, Any]:
    if doc.get("iri"):
        return {"iri": doc["iri"]}
    return {"name": doc["name"], "_ontology_type": doc.get("_ontology_type")}


def find_existing_doc_id(coll: Collection, query: Dict[str, Any]) -> Optional[ObjectId]:
    found = coll.find_one(query, {"_id": 1})
    return found["_id"] if found else None


def missing_required_fields(doc: Dict[str, Any], specs: Dict[str, FieldSpec]) -> List[str]:
    missing: List[str] = []
    for name, spec in specs.items():
        if not spec.required:
            continue
        if name not in doc:
            missing.append(name)
            continue
        value = doc[name]
        if value is None:
            missing.append(name)
            continue
        if isinstance(value, list) and len(value) == 0:
            missing.append(name)
    return missing


def has_required_fields(doc: Dict[str, Any], specs: Dict[str, FieldSpec]) -> bool:
    return len(missing_required_fields(doc, specs)) == 0


def upsert_final_doc(coll: Collection, doc: Dict[str, Any]) -> ObjectId:
    query = choose_identity_query(doc)
    existing_id = find_existing_doc_id(coll, query)
    if existing_id:
        coll.update_one({"_id": existing_id}, {"$set": doc})
        return existing_id
    return coll.insert_one(dict(doc)).inserted_id


def _normalize_base_doc(
        ind_name: str,
        ind: Dict[str, Any],
        chosen_type: str,
        specs: Dict[str, FieldSpec],
        data_map: Dict[Tuple[str, str], str],
) -> Dict[str, Any]:
    doc: Dict[str, Any] = {
        "name": ind.get("name", ind_name),
        "_ontology_type": chosen_type,
        "metadata": {
            "createdAt": utcnow(),
            "createdBy": "create_mongodb_data_layer_with_import_placeholders",
            "updatedAt": utcnow(),
            "notes": f"Imported from ontology named_individuals: {ind_name}",
        },
    }

    if ind.get("iri"):
        doc["iri"] = ind["iri"]

    labels = (ind.get("annotations") or {}).get("rdfs:label") or []
    if labels:
        first_label = labels[0]
        if isinstance(first_label, dict):
            doc["label"] = first_label.get("value")
        else:
            doc["label"] = str(first_label)

    for prop_name, raw_values in (ind.get("data_assertions") or {}).items():
        field_name = data_map.get((chosen_type, prop_name))
        if not field_name or field_name not in specs:
            continue
        spec = specs[field_name]
        values = as_list(raw_values)
        normalized = [parse_scalar_by_bson(v, spec.bson_type) for v in values]
        normalized = [v for v in normalized if v is not None]
        if not normalized:
            continue
        doc[field_name] = normalized if spec.is_array else normalized[0]

    return doc


def _resolve_object_refs_into_doc(
        ind_name: str,
        ind: Dict[str, Any],
        chosen_type: str,
        specs: Dict[str, FieldSpec],
        obj_map: Dict[Tuple[str, str], str],
        prepared_docs: Dict[str, Dict[str, Any]],
        collection_by_individual: Dict[str, str],
        id_by_individual: Dict[str, ObjectId],
        class_to_collection: Dict[str, str],
        named: Dict[str, Any],
        ontology_bp: Dict[str, Any],
        imported_iris: List[str],
        current_bases: Set[str],
        db: Database,
) -> int:
    """
    Try to resolve as many object-property targets as possible into objectId fields.
    Returns the number of field assignments newly added/changed on this doc.
    """
    doc = prepared_docs[ind_name]
    changed = 0

    for prop_name, raw_targets in (ind.get("object_assertions") or {}).items():
        field_name = obj_map.get((chosen_type, prop_name))
        if not field_name or field_name not in specs:
            continue

        spec = specs[field_name]
        refs: List[ObjectId] = []
        unresolved_entries: List[Dict[str, Any]] = []
        diagnostics_entries: List[Dict[str, Any]] = []
        unresolved_field = unresolved_field_name_for_reference(field_name)

        for raw_target in as_list(raw_targets):
            if isinstance(raw_target, dict) and "value" in raw_target:
                raw_target = raw_target["value"]
            if not isinstance(raw_target, str):
                continue

            target_name = iri_to_local_name(raw_target.lstrip("#")) if raw_target.startswith(
                "#") else iri_to_local_name(raw_target)
            target_oid = id_by_individual.get(target_name)

            # Fallback 1: maybe the target already exists in DB under its identity query.
            if target_oid is None:
                target_doc = prepared_docs.get(target_name)
                if target_doc is not None:
                    target_coll_name = collection_by_individual.get(target_name)
                    if target_coll_name:
                        target_coll = db[target_coll_name]
                        target_oid = find_existing_doc_id(target_coll, choose_identity_query(target_doc))
                        if target_oid is not None:
                            id_by_individual[target_name] = target_oid

            # Fallback 2: try class-based collection lookup from current blueprint.
            if target_oid is None:
                target_asserted = (named.get(target_name) or {}).get("asserted_types") or []
                target_type = choose_concrete_type(ontology_bp, as_list(target_asserted), class_to_collection)
                target_coll_name = class_to_collection.get(target_type) if target_type else None
                if target_coll_name:
                    target_coll = db[target_coll_name]
                    found = target_coll.find_one(
                        {"$or": [{"name": target_name}, {"iri": raw_target}]},
                        {"_id": 1},
                    )
                    if found:
                        target_oid = found["_id"]
                        id_by_individual[target_name] = target_oid

            if target_oid is not None:
                refs.append(target_oid)
                continue

            # Keep unresolved imported targets instead of silently losing them.
            import_source = match_import_source(raw_target, imported_iris, current_bases)
            if import_source is not None:
                unresolved_entries.append(
                    {
                        "iri": raw_target,
                        "local_name": target_name,
                        "source": "imported_ontology",
                        "import_iri": import_source,
                    }
                )
                diagnostics_entries.append(
                    {
                        "property": prop_name,
                        "field": unresolved_field,
                        "iri": raw_target,
                        "local_name": target_name,
                        "import_source": import_source,
                    }
                )

        if refs:
            new_value = refs if spec.is_array else refs[0]
            if doc.get(field_name) != new_value:
                doc[field_name] = new_value
                changed += 1
        else:
            if field_name in doc:
                doc.pop(field_name, None)
                changed += 1

        previous_unresolved = doc.get(unresolved_field)
        set_unresolved_reference_payload(doc, unresolved_field, diagnostics_entries, unresolved_entries)
        if previous_unresolved != doc.get(unresolved_field):
            changed += 1

    return changed


def seed_or_import_instances(
        db: Database,
        validators: Dict[str, Any],
        ontology_bp: Dict[str, Any],
        imported_iris: Optional[List[str]] = None,
) -> Dict[str, Any]:
    imported_iris = imported_iris or []
    current_bases = get_current_ontology_bases(ontology_bp)
    named = ontology_bp.get("named_individuals") or {}
    class_to_collection = build_class_to_collection(validators, ontology_bp)
    field_specs_by_coll = get_collection_field_specs(validators)
    data_map, obj_map = build_prop_maps(validators, ontology_bp, class_to_collection)

    chosen_type_by_individual: Dict[str, str] = {}
    collection_by_individual: Dict[str, str] = {}
    prepared_docs: Dict[str, Dict[str, Any]] = {}
    id_by_individual: Dict[str, ObjectId] = {}

    skipped_by_collection: Dict[str, int] = {}
    skipped_by_combo: Dict[str, int] = {}
    skipped_examples: Dict[str, List[str]] = {}

    summary = {
        "inserted_or_upserted": 0,
        "reference_updates": 0,
        "delayed_inserted": 0,
        "resolution_rounds": 0,
        "skipped_no_type": 0,
        "skipped_no_collection": 0,
        "skipped_still_incomplete": 0,
        "skipped_by_collection": skipped_by_collection,
        "skipped_by_combo": skipped_by_combo,
        "skipped_examples": skipped_examples,
        "docs_with_unresolved_imports": 0,
        "unresolved_import_references": 0,
        "unresolved_import_examples": [],
        "imported_ontologies_detected": imported_iris,
    }

    # Build prepared docs from scalar/data assertions only.
    for ind_name, ind in named.items():
        ind = ind or {}
        asserted_types = as_list(ind.get("asserted_types"))
        chosen_type = choose_concrete_type(ontology_bp, asserted_types, class_to_collection)
        if not chosen_type:
            summary["skipped_no_type"] += 1
            continue

        coll_name = class_to_collection.get(chosen_type)
        if not coll_name or coll_name not in validators:
            summary["skipped_no_collection"] += 1
            continue

        specs = field_specs_by_coll.get(coll_name, {})
        doc = _normalize_base_doc(ind_name, ind, chosen_type, specs, data_map)

        prepared_docs[ind_name] = doc
        chosen_type_by_individual[ind_name] = chosen_type
        collection_by_individual[ind_name] = coll_name

        coll = db[coll_name]
        existing_id = find_existing_doc_id(coll, choose_identity_query(doc))
        if existing_id is not None:
            id_by_individual[ind_name] = existing_id

    # Iteratively resolve references and insert unlockable docs.
    progress = True
    while progress:
        progress = False
        summary["resolution_rounds"] += 1

        changed_fields_this_round = 0
        for ind_name, ind in named.items():
            if ind_name not in prepared_docs:
                continue
            chosen_type = chosen_type_by_individual[ind_name]
            coll_name = collection_by_individual[ind_name]
            specs = field_specs_by_coll.get(coll_name, {})
            changed_fields_this_round += _resolve_object_refs_into_doc(
                ind_name=ind_name,
                ind=ind,
                chosen_type=chosen_type,
                specs=specs,
                obj_map=obj_map,
                prepared_docs=prepared_docs,
                collection_by_individual=collection_by_individual,
                id_by_individual=id_by_individual,
                class_to_collection=class_to_collection,
                named=named,
                ontology_bp=ontology_bp,
                imported_iris=imported_iris,
                current_bases=current_bases,
                db=db,
            )

        inserted_this_round = 0
        updated_this_round = 0

        for ind_name, doc in prepared_docs.items():
            coll_name = collection_by_individual[ind_name]
            coll = db[coll_name]
            specs = field_specs_by_coll.get(coll_name, {})

            if not has_required_fields(doc, specs):
                continue

            existing_id = id_by_individual.get(ind_name)
            final_id = upsert_final_doc(coll, doc)

            if existing_id is None:
                id_by_individual[ind_name] = final_id
                inserted_this_round += 1
            else:
                updated_this_round += 1
                id_by_individual[ind_name] = final_id

        if inserted_this_round > 0 or changed_fields_this_round > 0:
            progress = True

        summary["delayed_inserted"] += inserted_this_round
        summary["reference_updates"] += updated_this_round

        if not progress:
            break

    persisted_count = 0
    for ind_name in prepared_docs:
        if ind_name in id_by_individual:
            persisted_count += 1
    summary["inserted_or_upserted"] = persisted_count

    unresolved_docs = 0
    unresolved_refs = 0
    unresolved_examples: List[str] = []
    for ind_name, doc in prepared_docs.items():
        unresolved_block = (doc.get("_import_diagnostics") or {}).get("unresolved_import_references") or []
        if unresolved_block:
            unresolved_docs += 1
            unresolved_refs += len(unresolved_block)
            if len(unresolved_examples) < 8:
                unresolved_examples.append(ind_name)

    summary["docs_with_unresolved_imports"] = unresolved_docs
    summary["unresolved_import_references"] = unresolved_refs
    summary["unresolved_import_examples"] = unresolved_examples

    for ind_name, doc in prepared_docs.items():
        if ind_name in id_by_individual:
            continue

        coll_name = collection_by_individual[ind_name]
        specs = field_specs_by_coll.get(coll_name, {})
        missing = missing_required_fields(doc, specs)

        if missing:
            summary["skipped_still_incomplete"] += 1
            skipped_by_collection[coll_name] = skipped_by_collection.get(coll_name, 0) + 1
            combo_key = f"{coll_name} :: " + ", ".join(sorted(missing))
            skipped_by_combo[combo_key] = skipped_by_combo.get(combo_key, 0) + 1
            if combo_key not in skipped_examples:
                skipped_examples[combo_key] = []
            if len(skipped_examples[combo_key]) < 8:
                skipped_examples[combo_key].append(ind_name)

    return summary


def print_skip_diagnostics(summary: Dict[str, Any]) -> None:
    skipped_by_collection: Dict[str, int] = summary.get("skipped_by_collection", {})
    skipped_by_combo: Dict[str, int] = summary.get("skipped_by_combo", {})
    skipped_examples: Dict[str, List[str]] = summary.get("skipped_examples", {})
    unresolved_docs = summary.get("docs_with_unresolved_imports", 0)
    unresolved_refs = summary.get("unresolved_import_references", 0)
    unresolved_examples = summary.get("unresolved_import_examples", [])
    imported_ontologies = summary.get("imported_ontologies_detected", [])

    print("\n" + "=" * 88)
    print("SKIP DIAGNOSTICS")
    print("=" * 88)

    if imported_ontologies:
        print("\nDetected imported ontologies:")
        for iri in imported_ontologies:
            print(f"  - {iri}")

    if unresolved_docs:
        print("\nUnresolved imported references preserved as placeholders:")
        print(f"  - documents affected: {unresolved_docs}")
        print(f"  - unresolved references: {unresolved_refs}")
        if unresolved_examples:
            print(f"  - example docs: {', '.join(unresolved_examples[:5])}")

    if not skipped_by_collection:
        if not unresolved_docs:
            print("No incomplete-instance skips.")
        return

    print("\nSkipped by collection:")
    for coll_name, count in sorted(skipped_by_collection.items(), key=lambda x: (-x[1], x[0])):
        print(f"  - {coll_name}: {count}")

    print("\nSkipped by missing required-field combination:")
    for combo_key, count in sorted(skipped_by_combo.items(), key=lambda x: (-x[1], x[0])):
        examples = skipped_examples.get(combo_key, [])
        ex_text = ", ".join(examples[:5])
        print(f"  - {combo_key}: {count}")
        if ex_text:
            print(f"      examples: {ex_text}")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def ensure_collection_block(db: Database, validators: Dict[str, Any], drop_collections: bool) -> None:
    maybe_drop_collections(db, validators, drop_collections)
    for coll_name, validator_doc in validators.items():
        ensure_collection(db, coll_name, validator_doc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create MongoDB data layer and import ontology instances.")
    parser.add_argument("--mongo_uri", required=True, help="MongoDB connection string")
    parser.add_argument("--db", required=True, help="Target database name")
    parser.add_argument("--validators", required=True, help="Path to mongo_validators.json")
    parser.add_argument("--ontology_blueprint", required=True, help="Path to ontology_blueprint(.json)")
    parser.add_argument(
        "--source_owl",
        default=None,
        help="Optional mixed/source OWL to re-read explicit instance assertions and detect owl:imports",
    )
    parser.add_argument(
        "--drop_collections",
        action="store_true",
        help="Drop validator collections before recreating them",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    validators = load_json(args.validators)
    ontology_bp = load_json(args.ontology_blueprint)
    imported_iris: List[str] = []

    if args.source_owl:
        imported_iris = parse_owl_imports(args.source_owl)
        owl_instances = parse_mixed_owl_instances(args.source_owl)
        ontology_bp = merge_owl_assertions_into_blueprint(ontology_bp, owl_instances)

    client = MongoClient(args.mongo_uri)
    db = client[args.db]

    ensure_collection_block(db, validators, args.drop_collections)
    write_ontology_meta(db, ontology_bp, validators)

    summary = seed_or_import_instances(db, validators, ontology_bp, imported_iris=imported_iris)

    print("=" * 88)
    print("MongoDB data layer creation completed")
    print("=" * 88)
    print(f"Database: {args.db}")
    print(f"Collections in validator: {len(validators)}")
    print(f"Named individuals available : {len(ontology_bp.get('named_individuals', {}))}")
    print(f"Inserted/upserted docs      : {summary['inserted_or_upserted']}")
    print(f"Delayed inserted docs       : {summary['delayed_inserted']}")
    print(f"Reference-like updates      : {summary['reference_updates']}")
    print(f"Resolution rounds           : {summary['resolution_rounds']}")
    print(f"Skipped (no type)           : {summary['skipped_no_type']}")
    print(f"Skipped (no collection)     : {summary['skipped_no_collection']}")
    print(f"Skipped (still incomplete)  : {summary['skipped_still_incomplete']}")
    print(f"Docs with unresolved imports: {summary['docs_with_unresolved_imports']}")
    print(f"Unresolved import refs      : {summary['unresolved_import_references']}")
    if args.source_owl:
        print(f"Used source_owl fallback    : {args.source_owl}")

    print_skip_diagnostics(summary)


if __name__ == "__main__":
    main()
