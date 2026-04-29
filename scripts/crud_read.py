from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import MongoClient

from blueprint_registry import BlueprintRegistry


class ReadDAL:
    SYSTEM_FIELDS = {"_id", "_ontology_type"}

    def __init__(self, registry: BlueprintRegistry, db: Any):
        self.registry = registry
        self.db = db

    def get_by_id(
            self,
            class_name_or_collection: str,
            document_id: str | ObjectId,
            depth: int = 0,
            include_metadata: bool = True,
    ) -> Optional[Dict[str, Any]]:
        collection_name = self._resolve_collection_name(class_name_or_collection)
        if not collection_name:
            return None

        oid = self._coerce_object_id(document_id)
        if oid is None:
            return None

        doc = self.db[collection_name].find_one({"_id": oid})
        if doc is None:
            return None

        expanded = self._expand_document(
            collection_name=collection_name,
            document=doc,
            depth=depth,
            visited={(collection_name, oid)},
        )
        if not include_metadata and isinstance(expanded, dict):
            expanded.pop("metadata", None)
        return expanded

    def get_one(
            self,
            class_name_or_collection: str,
            query: Dict[str, Any],
            depth: int = 0,
            include_metadata: bool = True,
    ) -> Optional[Dict[str, Any]]:
        collection_name = self._resolve_collection_name(class_name_or_collection)
        if not collection_name:
            return None

        doc = self.db[collection_name].find_one(query)
        if doc is None:
            return None

        oid = doc.get("_id")
        visited = {(collection_name, oid)} if oid is not None else set()
        expanded = self._expand_document(
            collection_name=collection_name,
            document=doc,
            depth=depth,
            visited=visited,
        )
        if not include_metadata and isinstance(expanded, dict):
            expanded.pop("metadata", None)
        return expanded

    def list_documents(
            self,
            class_name_or_collection: str,
            query: Optional[Dict[str, Any]] = None,
            limit: int = 50,
            depth: int = 0,
            include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        collection_name = self._resolve_collection_name(class_name_or_collection)
        if not collection_name:
            return []

        query = query or {}
        results: List[Dict[str, Any]] = []
        cursor = self.db[collection_name].find(query).limit(limit)

        for doc in cursor:
            oid = doc.get("_id")
            visited = {(collection_name, oid)} if oid is not None else set()
            expanded = self._expand_document(
                collection_name=collection_name,
                document=doc,
                depth=depth,
                visited=visited,
            )
            if not include_metadata and isinstance(expanded, dict):
                expanded.pop("metadata", None)
            results.append(expanded)

        return results

    def _expand_document(
            self,
            collection_name: str,
            document: Dict[str, Any],
            depth: int,
            visited: set[tuple[str, ObjectId]],
    ) -> Dict[str, Any]:
        out = deepcopy(document)
        if depth <= 0:
            return out

        reference_specs = self.registry.get_reference_fields(collection_name)

        for field_name, spec in reference_specs.items():
            if field_name not in out:
                continue

            value = out.get(field_name)
            if value is None:
                continue

            target_collections = self.registry.get_target_collections(collection_name, field_name)
            if not target_collections:
                continue

            if spec.is_array:
                expanded_items: List[Any] = []
                for raw in value:
                    expanded_items.append(
                        self._expand_reference_value(
                            target_collections=target_collections,
                            raw_value=raw,
                            depth=depth,
                            visited=visited,
                        )
                    )
                out[field_name] = expanded_items
            else:
                out[field_name] = self._expand_reference_value(
                    target_collections=target_collections,
                    raw_value=value,
                    depth=depth,
                    visited=visited,
                )

        return out

    def _expand_reference_value(
            self,
            target_collections: Sequence[str],
            raw_value: Any,
            depth: int,
            visited: set[tuple[str, ObjectId]],
    ) -> Any:
        oid = self._coerce_object_id(raw_value)
        if oid is None:
            return raw_value

        for target_collection in target_collections:
            if (target_collection, oid) in visited:
                return {"_id": oid, "_circular_ref": True, "_collection": target_collection}

            doc = self.db[target_collection].find_one({"_id": oid})
            if doc is None:
                continue

            next_visited = set(visited)
            next_visited.add((target_collection, oid))
            expanded = self._expand_document(
                collection_name=target_collection,
                document=doc,
                depth=depth - 1,
                visited=next_visited,
            )
            expanded["_collection"] = target_collection
            return expanded

        return raw_value

    def _resolve_collection_name(self, class_name_or_collection: str) -> Optional[str]:
        if not class_name_or_collection:
            return None
        if class_name_or_collection in self.registry.collection_specs:
            return class_name_or_collection
        return self.registry.get_collection_for_class(class_name_or_collection)

    @staticmethod
    def _coerce_object_id(value: Any) -> Optional[ObjectId]:
        if isinstance(value, ObjectId):
            return value
        if isinstance(value, str):
            try:
                return ObjectId(value)
            except (InvalidId, TypeError):
                return None
        return None

    @classmethod
    def from_files(
            cls,
            ontology_blueprint_path: str | Path,
            validators_path: str | Path,
            mongo_uri: str,
            db_name: str,
    ) -> "ReadDAL":
        ontology_bp = json.loads(Path(ontology_blueprint_path).read_text(encoding="utf-8"))
        validators = json.loads(Path(validators_path).read_text(encoding="utf-8"))
        registry = BlueprintRegistry(ontology_bp, validators)
        client = MongoClient(mongo_uri)
        db = client[db_name]
        return cls(registry=registry, db=db)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Read DAL demo")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_get = sub.add_parser("get_by_id")
    p_get.add_argument("ontology_blueprint", type=str)
    p_get.add_argument("validators", type=str)
    p_get.add_argument("mongo_uri", type=str)
    p_get.add_argument("db_name", type=str)
    p_get.add_argument("class_or_collection", type=str)
    p_get.add_argument("document_id", type=str)
    p_get.add_argument("--depth", type=int, default=0)

    p_one = sub.add_parser("get_one")
    p_one.add_argument("ontology_blueprint", type=str)
    p_one.add_argument("validators", type=str)
    p_one.add_argument("mongo_uri", type=str)
    p_one.add_argument("db_name", type=str)
    p_one.add_argument("class_or_collection", type=str)
    p_one.add_argument("query_json", type=str)
    p_one.add_argument("--depth", type=int, default=0)

    p_list = sub.add_parser("list")
    p_list.add_argument("ontology_blueprint", type=str)
    p_list.add_argument("validators", type=str)
    p_list.add_argument("mongo_uri", type=str)
    p_list.add_argument("db_name", type=str)
    p_list.add_argument("class_or_collection", type=str)
    p_list.add_argument("--query_json", type=str, default="{}")
    p_list.add_argument("--limit", type=int, default=50)
    p_list.add_argument("--depth", type=int, default=0)

    args = parser.parse_args()
    dal = ReadDAL.from_files(args.ontology_blueprint, args.validators, args.mongo_uri, args.db_name)

    if args.mode == "get_by_id":
        result = dal.get_by_id(args.class_or_collection, args.document_id, depth=args.depth)
    elif args.mode == "get_one":
        result = dal.get_one(args.class_or_collection, json.loads(args.query_json), depth=args.depth)
    else:
        result = dal.list_documents(
            args.class_or_collection,
            query=json.loads(args.query_json),
            limit=args.limit,
            depth=args.depth,
        )

    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    _cli()
