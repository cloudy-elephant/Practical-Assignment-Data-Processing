"""Run separate retrieval paths and write a traceable RRF candidate list."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import tempfile
from time import monotonic, time_ns

from embedding.models import MODEL_SPECS, _clean_dense
from indexing.build import DEFAULT_MILVUS_URI, _verify_indexes, _verify_schema

from .models import QueryEncoder


TRACE_VERSION = "retrieval-v1"
OUTPUT_FIELDS = ["chunk_id", "document_id", "page_start", "page_end", "text", "source", "source_spans", "content_hash"]


def _canonical(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _index_manifest(path: Path) -> tuple[dict, str]:
    payload = path.read_bytes()
    manifest = json.loads(payload)
    if manifest.get("status") != "complete" or manifest.get("schema_version") != "milvus-index-v1":
        raise ValueError("Expected a complete milvus-index-v1 manifest")
    if manifest.get("verified_rows") != manifest.get("row_count") or not manifest.get("row_count"):
        raise ValueError("Index manifest has an incomplete row verification")
    spec = next((item for item in MODEL_SPECS.values() if item.model_id == manifest.get("model_id")), None)
    if spec is None or manifest.get("model_revision") != spec.revision:
        raise ValueError("Index manifest model/revision differs from pinned model")
    fields = set(manifest["indexes"])
    expected = {"dense_vector", "bm25_sparse"} | ({"learned_sparse"} if spec.has_sparse else set())
    if fields != expected:
        raise ValueError("Index manifest has incompatible retrieval paths")
    indexes = manifest["indexes"]
    if (indexes["dense_vector"].get("index_type") != "HNSW"
            or indexes["dense_vector"].get("metric_type") != "COSINE"
            or indexes["bm25_sparse"].get("index_type") != "SPARSE_INVERTED_INDEX"
            or indexes["bm25_sparse"].get("metric_type") != "BM25"
            or indexes["bm25_sparse"].get("params", {}).get("inverted_index_algo") != "DAAT_MAXSCORE"):
        raise ValueError("Index manifest has incompatible dense or BM25 configuration")
    if spec.has_sparse and (indexes["learned_sparse"].get("index_type") != "SPARSE_INVERTED_INDEX"
                            or indexes["learned_sparse"].get("metric_type") != "IP"
                            or indexes["learned_sparse"].get("params", {}).get("inverted_index_algo") != "SINDI"):
        raise ValueError("Index manifest has incompatible learned sparse configuration")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,254}", manifest["collection"]):
        raise ValueError("Invalid collection name in index manifest")
    return manifest, sha256(payload).hexdigest()


def _filter(document_id: str | None, page: int | None) -> str:
    conditions = []
    if document_id is not None:
        if not document_id or len(document_id.encode("utf-8")) > 512:
            raise ValueError("document_id must be nonempty and <= 512 bytes")
        conditions.append(f"document_id == {json.dumps(document_id, ensure_ascii=False)}")
    if page is not None:
        if page <= 0:
            raise ValueError("page must be positive")
        conditions.extend((f"page_start <= {page}", f"page_end >= {page}"))
    return " and ".join(conditions)


def _paths(query: str, encoded: dict, has_sparse: bool, ef: int) -> dict:
    result = {
        "dense": {"anns_field": "dense_vector", "data": [encoded["dense_vector"]],
                  "search_params": {"metric_type": "COSINE", "params": {"ef": ef}}},
        "bm25": {"anns_field": "bm25_sparse", "data": [query],
                 "search_params": {"metric_type": "BM25", "params": {}}},
    }
    if has_sparse:
        sparse = encoded.get("sparse_vector")
        if not isinstance(sparse, dict) or not sparse:
            raise ValueError("BGE-M3 query requires learned sparse weights")
        result["learned_sparse"] = {"anns_field": "learned_sparse", "data": [sparse],
                                    "search_params": {"metric_type": "IP", "params": {"drop_ratio_search": 0.0}}}
    return result


def _parse_hits(raw, *, path: str, document_id: str | None, page: int | None, limit: int) -> list[dict]:
    if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], list):
        raise ValueError(f"Milvus {path} search returned an unexpected shape")
    if len(raw[0]) > limit:
        raise ValueError(f"Milvus {path} search returned more than limit")
    hits, seen = [], set()
    for rank, item in enumerate(raw[0], 1):
        entity = item["entity"]
        chunk_id = str(entity["chunk_id"])
        returned_id = item.get("id")
        if not chunk_id or chunk_id in seen or (returned_id is not None and str(returned_id) != chunk_id):
            raise ValueError(f"Milvus {path} returned a duplicate or mismatched chunk_id")
        seen.add(chunk_id)
        score = float(item["distance"])
        if not math.isfinite(score):
            raise ValueError(f"Milvus {path} returned a nonfinite score")
        if document_id is not None and entity["document_id"] != document_id:
            raise ValueError(f"Milvus {path} returned a chunk outside document filter")
        if page is not None and not entity["page_start"] <= page <= entity["page_end"]:
            raise ValueError(f"Milvus {path} returned a chunk outside page filter")
        if sha256(entity["text"].encode("utf-8")).hexdigest() != entity["content_hash"]:
            raise ValueError(f"Milvus {path} returned text with mismatched content hash")
        hits.append({"chunk_id": chunk_id, "rank": rank, "score": score,
                     "document_id": entity["document_id"],
                     "page_range": [entity["page_start"], entity["page_end"]],
                     "text": entity["text"], "source": entity["source"],
                     "source_spans": entity["source_spans"], "content_hash": entity["content_hash"]})
    return hits


def fuse(paths: dict[str, list[dict]], *, rrf_k: int = 60, limit: int = 20) -> list[dict]:
    """Fuse ranks from independent paths while retaining every path's evidence."""
    if rrf_k <= 0 or limit <= 0:
        raise ValueError("rrf_k and limit must be positive")
    candidates: dict[str, dict] = {}
    for path, hits in paths.items():
        for hit in hits:
            chunk_id = hit["chunk_id"]
            metadata = {key: hit[key] for key in ("document_id", "page_range", "text", "source", "source_spans", "content_hash")}
            if chunk_id not in candidates:
                candidates[chunk_id] = {"chunk_id": chunk_id, **metadata, "paths": {}, "rrf_score": 0.0}
            elif any(candidates[chunk_id][key] != value for key, value in metadata.items()):
                raise ValueError(f"Paths disagree on metadata for {chunk_id}")
            if path in candidates[chunk_id]["paths"]:
                raise ValueError(f"Duplicate {chunk_id} in {path}")
            candidates[chunk_id]["paths"][path] = {"rank": hit["rank"], "score": hit["score"]}
            candidates[chunk_id]["rrf_score"] += 1.0 / (rrf_k + hit["rank"])
    ordered = sorted(candidates.values(), key=lambda item: (-item["rrf_score"],
                     min(p["rank"] for p in item["paths"].values()), item["chunk_id"]))
    return [{"fused_rank": rank, **candidate} for rank, candidate in enumerate(ordered[:limit], 1)]


def retrieve(index_manifest_path: str | Path, question: str, output_dir: str | Path, *,
             document_id: str | None = None, page: int | None = None,
             path_limit: int = 20, candidate_limit: int = 20, rrf_k: int = 60,
             hnsw_ef: int = 64, device: str = "auto", local_files_only: bool = False,
             client=None, query_encoder=None) -> dict:
    """Search matching model paths and save a retrieval-only trace."""
    if not question or not question.strip():
        raise ValueError("question must be nonempty")
    if path_limit <= 0 or path_limit > 16384 or candidate_limit <= 0 or rrf_k <= 0:
        raise ValueError("path_limit, candidate_limit, and rrf_k must be positive; path_limit <= 16384")
    if hnsw_ef < path_limit:
        raise ValueError("hnsw_ef must be >= path_limit")
    manifest_path = Path(index_manifest_path)
    manifest, manifest_sha = _index_manifest(manifest_path)
    expression = _filter(document_id, page)
    spec = next(item for item in MODEL_SPECS.values() if item.model_id == manifest["model_id"])
    encoder = query_encoder if query_encoder is not None else QueryEncoder(spec.model_id, device=device, local_files_only=local_files_only)
    start_total = monotonic()
    encoded = encoder.encode(question)
    encoded["dense_vector"] = _clean_dense(encoded["dense_vector"], spec.dimension)
    path_requests = _paths(question, encoded, spec.has_sparse, hnsw_ef)
    owned_client = client is None
    if owned_client:
        from pymilvus import MilvusClient

        uri = os.environ.get("MILVUS_URI") or DEFAULT_MILVUS_URI
        client = MilvusClient(uri=uri, token=os.environ.get("MILVUS_TOKEN", ""),
                              db_name=os.environ.get("MILVUS_DB_NAME", "default"))
    try:
        if not client.has_collection(manifest["collection"]):
            raise ValueError("Collection from index manifest does not exist")
        _verify_schema(client, manifest)
        _verify_indexes(client, manifest, wait_seconds=0)

        def search_one(name: str, request: dict):
            start = monotonic()
            raw = client.search(collection_name=manifest["collection"], filter=expression,
                                limit=path_limit, output_fields=OUTPUT_FIELDS,
                                consistency_level="Strong", **request)
            return name, _parse_hits(raw, path=name, document_id=document_id, page=page, limit=path_limit), monotonic() - start

        with ThreadPoolExecutor(max_workers=len(path_requests)) as pool:
            futures = [pool.submit(search_one, name, request) for name, request in path_requests.items()]
            results = [future.result() for future in futures]
        path_hits = {name: hits for name, hits, _ in results}
        durations = {name: seconds for name, _, seconds in results}
        candidates = fuse(path_hits, rrf_k=rrf_k, limit=candidate_limit)
        trace = {
            "schema_version": TRACE_VERSION, "status": "complete", "stage": "retrieval",
            "question": question, "document_filter": document_id, "page_filter": page,
            "filter_expression": expression, "collection": manifest["collection"],
            "index_manifest_path": str(manifest_path.resolve()),
            "index_manifest_sha256": manifest_sha, "model_id": spec.model_id,
            "model_revision": spec.revision, "query_prompt": encoded["query_prompt"],
            "query_model_token_count": encoded["model_token_count"],
            "query_precision": encoded["precision"], "path_limit": path_limit,
            "candidate_limit": candidate_limit, "rrf_k": rrf_k, "hnsw_ef": hnsw_ef,
            "paths": {name: {"field": path_requests[name]["anns_field"],
                             "search_params": path_requests[name]["search_params"],
                             "elapsed_seconds": durations[name], "hits": path_hits[name]}
                      for name in path_requests},
            "fused_candidates": candidates, "elapsed_seconds": monotonic() - start_total,
        }
        query_hash = sha256(_canonical({"manifest": manifest_sha, "question": question,
                                        "document_id": document_id, "page": page,
                                        "path_limit": path_limit, "candidate_limit": candidate_limit,
                                        "rrf_k": rrf_k, "hnsw_ef": hnsw_ef})).hexdigest()[:16]
        destination = Path(output_dir) / manifest["collection"] / f"{query_hash}_{time_ns()}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".trace_", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(_canonical(trace) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        return {**trace, "trace_path": str(destination.resolve())}
    finally:
        if owned_client:
            client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model-matched Milvus retrieval and RRF fusion")
    parser.add_argument("index_manifest", type=Path)
    parser.add_argument("question")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--document-id")
    parser.add_argument("--page", type=int)
    parser.add_argument("--path-limit", type=int, default=20)
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--hnsw-ef", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    result = retrieve(args.index_manifest, args.question, args.output_dir,
                      document_id=args.document_id, page=args.page,
                      path_limit=args.path_limit, candidate_limit=args.candidate_limit,
                      rrf_k=args.rrf_k, hnsw_ef=args.hnsw_ef, device=args.device,
                      local_files_only=args.local_files_only)
    print(json.dumps({"trace_path": result["trace_path"], "collection": result["collection"],
                      "candidate_count": len(result["fused_candidates"]),
                      "path_counts": {name: len(value["hits"]) for name, value in result["paths"].items()}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
