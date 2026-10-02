"""Build isolated Milvus collections from completed embedding artifacts."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import time

from embedding.models import MODEL_SPECS


INDEX_SCHEMA_VERSION = "milvus-index-v1"
DEFAULT_MILVUS_URI = "http://127.0.0.1:19530"
INDEX_NAMES = {"dense_vector": "dense_hnsw", "bm25_sparse": "bm25_inverted", "learned_sparse": "learned_sparse_sindi"}


def _canonical(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_artifact(path: Path) -> tuple[dict, list[dict]]:
    manifest_path = path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Embedding manifest is not complete")
    if manifest["embeddings_file_sha256"] != _sha_file(path):
        raise ValueError("Embedding file checksum differs from manifest")
    config = manifest["config"]
    spec = next((s for s in MODEL_SPECS.values() if s.model_id == config["model_id"]), None)
    if spec is None or spec.revision != config["model_revision"] or spec.dimension != config["dimension"]:
        raise ValueError("Embedding model, revision, or dimension does not match pinned model spec")
    rows, ids = [], set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                checksum = row.pop("record_checksum")
                if checksum != sha256(_canonical(row)).hexdigest():
                    raise ValueError("record checksum mismatch")
                row["record_checksum"] = checksum
                if row["chunk_id"] in ids or not row["chunk_id"]:
                    raise ValueError("missing or duplicate chunk_id")
                if row["config_hash"] != manifest["config_hash"] or row["model_id"] != spec.model_id or row["model_revision"] != spec.revision:
                    raise ValueError("mixed embedding configurations")
                if row["method"] != config["chunking_method"] or row["chunking_config_hash"] != config["chunking_config_hash"]:
                    raise ValueError("mixed chunking configurations")
                if not isinstance(row["text"], str) or sha256(row["text"].encode("utf-8")).hexdigest() != row["content_hash"]:
                    raise ValueError("text/content_hash mismatch")
                if not row["text"].strip() or len(row["text"].encode("utf-8")) > 65535:
                    raise ValueError("text must fit Milvus VARCHAR(65535)")
                if len(row["dense_vector"]) != spec.dimension or not all(math.isfinite(float(v)) for v in row["dense_vector"]):
                    raise ValueError("invalid dense vector")
                norm = math.sqrt(sum(float(v) ** 2 for v in row["dense_vector"]))
                if not 0.99 <= norm <= 1.01:
                    raise ValueError("dense vector is not L2-normalized")
                sparse = row["sparse_vector"]
                if spec.has_sparse:
                    if not isinstance(sparse, dict) or not sparse:
                        raise ValueError("BGE-M3 learned sparse vector is missing")
                    for key, value in sparse.items():
                        if int(key) < 0 or not math.isfinite(float(value)) or float(value) <= 0:
                            raise ValueError("invalid learned sparse vector")
                elif sparse is not None:
                    raise ValueError("Qwen3 artifact cannot contain a learned sparse vector")
                if len(row["chunk_id"].encode("utf-8")) > 128 or len(row["document_id"].encode("utf-8")) > 512:
                    raise ValueError("chunk_id or document_id exceeds Milvus field length")
                if len(row["source_checksum"]) > 64 or len(row["content_hash"]) != 64:
                    raise ValueError("source/content checksum exceeds Milvus field length")
                if (not isinstance(row["page_range"], list) or len(row["page_range"]) != 2
                        or not all(isinstance(page, int) and page >= 1 for page in row["page_range"])
                        or row["page_range"][0] > row["page_range"][1]):
                    raise ValueError("invalid page_range")
                ids.add(row["chunk_id"])
                rows.append(row)
            except (KeyError, TypeError, ValueError, OverflowError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid embedding on line {line_number}: {exc}") from exc
    if len(rows) != manifest["chunk_count"] or not rows:
        raise ValueError("Embedding row count differs from manifest")
    return manifest, rows


def make_plan(path: str | Path, *, collection_prefix: str = "assignment", hnsw_m: int = 16,
              hnsw_ef_construction: int = 200) -> tuple[dict, list[dict]]:
    """Verify input and describe the collection without contacting Milvus."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,80}", collection_prefix):
        raise ValueError("collection_prefix must start with a letter and contain only letters, digits, underscores")
    if hnsw_m < 4 or hnsw_m > 64 or hnsw_ef_construction < hnsw_m:
        raise ValueError("HNSW requires 4 <= M <= 64 and efConstruction >= M")
    source = Path(path)
    manifest, rows = _read_artifact(source)
    config = manifest["config"]
    model = "bge_m3" if config["model_id"] == "BAAI/bge-m3" else "qwen3_0_6b"
    indexes = {
        "dense_vector": {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": hnsw_m, "efConstruction": hnsw_ef_construction}},
        "bm25_sparse": {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25", "params": {"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": 1.2, "bm25_b": 0.75}},
    }
    if config["model_id"] == "BAAI/bge-m3":
        indexes["learned_sparse"] = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP", "params": {"inverted_index_algo": "SINDI"}}
    run_identity = {"embedding_config_hash": manifest["config_hash"],
                    "embedding_file_sha256": manifest["embeddings_file_sha256"], "indexes": indexes}
    run_hash = sha256(_canonical(run_identity)).hexdigest()[:12]
    collection = f"{collection_prefix}_{config['chunking_method']}_{model}_{run_hash}"
    if len(collection) > 255:
        raise ValueError("Collection name exceeds Milvus limit")
    plan = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "collection": collection,
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "embedding_config_hash": manifest["config_hash"],
        "embedding_file_sha256": manifest["embeddings_file_sha256"],
        "embedding_manifest_path": str(source.with_name("manifest.json").resolve()),
        "embedding_path": str(source.resolve()),
        "row_count": len(rows),
        "chunking_method": config["chunking_method"],
        "chunking_config_hash": config["chunking_config_hash"],
        "indexes": indexes,
        "requires_target_vec_index_version": 10 if "learned_sparse" in indexes else None,
    }
    return plan, rows


def _schema(client, plan: dict):
    from pymilvus import DataType, Function, FunctionType

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False, description=f"{INDEX_SCHEMA_VERSION}:{plan['embedding_file_sha256']}")
    schema.add_field("chunk_id", DataType.VARCHAR, max_length=128, is_primary=True)
    schema.add_field("document_id", DataType.VARCHAR, max_length=512)
    schema.add_field("page_start", DataType.INT64)
    schema.add_field("page_end", DataType.INT64)
    schema.add_field("text", DataType.VARCHAR, max_length=65535, enable_analyzer=True)
    schema.add_field("source", DataType.JSON)
    schema.add_field("source_spans", DataType.JSON)
    schema.add_field("source_checksum", DataType.VARCHAR, max_length=64)
    schema.add_field("content_hash", DataType.VARCHAR, max_length=64)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field("bm25_sparse", DataType.SPARSE_FLOAT_VECTOR)
    if "learned_sparse" in plan["indexes"]:
        schema.add_field("learned_sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(name="text_bm25", input_field_names=["text"],
                                 output_field_names=["bm25_sparse"], function_type=FunctionType.BM25))
    return schema


def _index_params(client, plan: dict):
    params = client.prepare_index_params()
    for field, settings in plan["indexes"].items():
        params.add_index(field_name=field, index_name=INDEX_NAMES[field], **settings)
    return params


def _payload(row: dict, learned_sparse: bool) -> dict:
    start, end = row["page_range"]
    payload = {
        "chunk_id": row["chunk_id"], "document_id": row["document_id"],
        "page_start": start, "page_end": end, "text": row["text"],
        "source": row["source"], "source_spans": row["source_spans"],
        "source_checksum": row["source_checksum"], "content_hash": row["content_hash"],
        "dense_vector": row["dense_vector"],
    }
    if learned_sparse:
        payload["learned_sparse"] = {int(key): float(value) for key, value in row["sparse_vector"].items()}
    return payload


def _verify_schema(client, plan: dict) -> dict:
    remote = client.describe_collection(plan["collection"])
    if remote.get("description") != f"{INDEX_SCHEMA_VERSION}:{plan['embedding_file_sha256']}":
        raise ValueError("Existing collection has a different embedding artifact")
    fields = {field["name"]: field for field in remote["fields"]}
    required = {"chunk_id", "document_id", "page_start", "page_end", "text", "source", "source_spans", "source_checksum", "content_hash", "dense_vector", "bm25_sparse"}
    if "learned_sparse" in plan["indexes"]:
        required.add("learned_sparse")
    if set(fields) != required:
        raise ValueError("Existing collection has a different schema")
    if not any(f.get("name") == "text_bm25" for f in remote.get("functions", [])):
        raise ValueError("Existing collection lacks the BM25 function")
    return remote


def _verify_indexes(client, plan: dict, *, wait_seconds: int = 120) -> dict:
    deadline = time.monotonic() + wait_seconds
    while True:
        descriptions = {}
        ready = True
        for field, expected in plan["indexes"].items():
            description = client.describe_index(plan["collection"], INDEX_NAMES[field])
            if description.get("field_name") != field or description.get("index_type") != expected["index_type"] or description.get("metric_type") != expected["metric_type"]:
                raise ValueError(f"Index {field} does not match requested type/metric: {description}")
            for key, value in expected["params"].items():
                actual = description.get(key, description.get("params", {}).get(key))
                if str(actual) != str(value):
                    raise ValueError(f"Index {field} parameter {key} is {actual!r}, expected {value!r}")
            state = str(description.get("state", ""))
            if state in {"Failed", "IndexStateFailed"}:
                raise ValueError(f"Index {field} failed: {description}")
            if "indexed_rows" not in description:
                raise ValueError(f"Milvus did not report indexed_rows for {field}; physical index cannot be verified")
            if field == "learned_sparse" and "index_version" in description and int(description["index_version"]) < 10:
                raise ValueError("Learned sparse index is on an older physical index version")
            if (state not in {"Finished", "IndexStateFinished"}
                    or int(description.get("pending_index_rows", 0)) > 0
                    or int(description["indexed_rows"]) < plan["row_count"]):
                ready = False
            descriptions[field] = description
        if ready:
            return descriptions
        if time.monotonic() >= deadline:
            raise TimeoutError("Milvus indexes did not finish before timeout")
        time.sleep(2)


def build_index(path: str | Path, output_dir: str | Path, *, client=None,
                collection_prefix: str = "assignment", batch_size: int = 100,
                hnsw_m: int = 16, hnsw_ef_construction: int = 200,
                target_vec_index_version: int | None = None, dry_run: bool = False) -> dict:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    plan, rows = make_plan(path, collection_prefix=collection_prefix, hnsw_m=hnsw_m,
                           hnsw_ef_construction=hnsw_ef_construction)
    if dry_run:
        return {"status": "validated", **plan}
    if plan["requires_target_vec_index_version"] is not None and (target_vec_index_version or 0) < 10:
        raise ValueError("BGE-M3 SINDI requires server dataCoord.targetVecIndexVersion >= 10; pass the verified server setting")
    owned_client = client is None
    if owned_client:
        from pymilvus import MilvusClient

        uri = os.environ.get("MILVUS_URI") or DEFAULT_MILVUS_URI
        client = MilvusClient(uri=uri, token=os.environ.get("MILVUS_TOKEN", ""), db_name=os.environ.get("MILVUS_DB_NAME", "default"))
    try:
        server_version = client.get_server_version()
        if not re.match(r"^v?3\.", str(server_version)):
            raise ValueError(f"Milvus 3.x required, got {server_version}")
        if client.has_collection(plan["collection"]):
            _verify_schema(client, plan)
        else:
            client.create_collection(collection_name=plan["collection"], schema=_schema(client, plan),
                                     index_params=_index_params(client, plan))
            _verify_schema(client, plan)
        indexed_names = set(client.list_indexes(plan["collection"]))
        expected_names = {INDEX_NAMES[field] for field in plan["indexes"]}
        if indexed_names != expected_names:
            raise ValueError(f"Existing index names differ: {indexed_names} != {expected_names}")
        sparse = "learned_sparse" in plan["indexes"]
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            result = client.upsert(collection_name=plan["collection"], data=[_payload(row, sparse) for row in batch])
            if result.get("upsert_count") != len(batch):
                raise ValueError("Milvus upsert count differs from batch size")
        client.flush(plan["collection"])
        client.load_collection(plan["collection"])
        descriptions = _verify_indexes(client, plan)
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            got = client.get(plan["collection"], ids=[row["chunk_id"] for row in batch],
                             output_fields=["chunk_id", "content_hash"])
            actual = {row["chunk_id"]: row["content_hash"] for row in got}
            expected = {row["chunk_id"]: row["content_hash"] for row in batch}
            if actual != expected:
                raise ValueError("Milvus read-back does not match embedding chunk IDs and hashes")
        result = {"status": "complete", **plan, "server_version": server_version,
                  "target_vec_index_version_confirmed": target_vec_index_version,
                  "index_descriptions": descriptions, "verified_rows": len(rows)}
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        destination = output / f"{plan['collection']}.json"
        temporary = output / f".{plan['collection']}.tmp"
        temporary.write_bytes(_canonical(result) + b"\n")
        os.replace(temporary, destination)
        return {**result, "index_manifest_path": str(destination.resolve())}
    finally:
        if owned_client:
            client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Milvus collection from an embedding artifact")
    parser.add_argument("embeddings_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--collection-prefix", default="assignment")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--target-vec-index-version", type=int,
                        help="Confirmed server dataCoord.targetVecIndexVersion (>=10 for SINDI)")
    parser.add_argument("--dry-run", action="store_true", help="Validate artifact and print collection/index plan")
    args = parser.parse_args()
    result = build_index(args.embeddings_jsonl, args.output_dir,
                         collection_prefix=args.collection_prefix, batch_size=args.batch_size,
                         hnsw_m=args.hnsw_m, hnsw_ef_construction=args.hnsw_ef_construction,
                         target_vec_index_version=args.target_vec_index_version, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
