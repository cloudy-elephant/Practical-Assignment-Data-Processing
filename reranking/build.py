"""Rerank retrieval trace candidates and write bounded evidence artifacts."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from time import monotonic, time_ns

from .model import INSTRUCTION, MODEL_ID, MODEL_REVISION, Qwen3Reranker


SCHEMA_VERSION = "reranking-v1"


def _canonical(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _read_trace(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    trace = json.loads(raw)
    if trace.get("schema_version") != "retrieval-v1" or trace.get("status") != "complete" or trace.get("stage") != "retrieval":
        raise ValueError("Expected a complete retrieval-v1 trace")
    if not isinstance(trace.get("question"), str) or not trace["question"].strip():
        raise ValueError("Retrieval trace has no question")
    upstream = Path(trace["index_manifest_path"])
    if sha256(upstream.read_bytes()).hexdigest() != trace["index_manifest_sha256"]:
        raise ValueError("Index manifest checksum differs from retrieval trace")
    manifest = json.loads(upstream.read_text(encoding="utf-8"))
    if (manifest.get("status") != "complete" or manifest.get("collection") != trace["collection"]
            or manifest.get("model_id") != trace["model_id"] or manifest.get("model_revision") != trace["model_revision"]):
        raise ValueError("Retrieval trace and index manifest disagree")
    candidates = trace.get("fused_candidates")
    if not isinstance(candidates, list) or len(candidates) > trace["candidate_limit"]:
        raise ValueError("Invalid fused candidate list")
    seen = set()
    for rank, item in enumerate(candidates, 1):
        try:
            identifier = item["chunk_id"]
            text = item["text"]
            if not isinstance(identifier, str) or not identifier or identifier in seen:
                raise ValueError("missing or duplicate chunk_id")
            if item["fused_rank"] != rank or not math.isfinite(float(item["rrf_score"])):
                raise ValueError("invalid fused rank or score")
            if not isinstance(text, str) or not text.strip() or sha256(text.encode("utf-8")).hexdigest() != item["content_hash"]:
                raise ValueError("text/content_hash mismatch")
            if not item["source_spans"] or not item["page_range"] or not item["paths"]:
                raise ValueError("missing citation or retrieval metadata")
            seen.add(identifier)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid candidate {rank}: {exc}") from exc
    return trace, sha256(raw).hexdigest()


def rerank(trace_path: str | Path, output_dir: str | Path, *,
           max_candidates: int = 20, evidence_limit: int = 5,
           batch_size: int = 4, max_input_tokens: int = 2048,
           device: str = "auto", local_files_only: bool = False,
           scorer=None) -> dict:
    if max_candidates <= 0 or evidence_limit <= 0 or evidence_limit > max_candidates or batch_size <= 0:
        raise ValueError("Require 0 < evidence_limit <= max_candidates and positive batch_size")
    source = Path(trace_path)
    trace, trace_sha = _read_trace(source)
    candidates = trace["fused_candidates"][:max_candidates]
    started = monotonic()
    if candidates:
        scorer = scorer if scorer is not None else Qwen3Reranker(
            device=device, local_files_only=local_files_only, max_input_tokens=max_input_tokens,
        )
        scores = scorer.score(trace["question"], [item["text"] for item in candidates], batch_size=batch_size)
        if len(scores) != len(candidates):
            raise ValueError("Reranker returned the wrong number of scores")
    else:
        scores = []
    reranked = []
    for item, result in zip(candidates, scores):
        score = float(result["score"])
        count = result["input_tokens"]
        if not math.isfinite(score) or not isinstance(count, int) or count <= 0 or count > max_input_tokens:
            raise ValueError("Reranker returned an invalid score or token count")
        reranked.append({**item, "rerank_score": score, "rerank_input_tokens": count})
    reranked.sort(key=lambda item: (-item["rerank_score"], item["fused_rank"], item["chunk_id"]))
    reranked = [{"rerank_rank": rank, **item} for rank, item in enumerate(reranked, 1)]
    result = {
        "schema_version": SCHEMA_VERSION, "status": "complete", "stage": "reranking",
        "question": trace["question"], "collection": trace["collection"],
        "document_filter": trace["document_filter"], "page_filter": trace["page_filter"],
        "retrieval_trace_path": str(source.resolve()), "retrieval_trace_sha256": trace_sha,
        "index_manifest_path": trace["index_manifest_path"],
        "index_manifest_sha256": trace["index_manifest_sha256"],
        "document_embedding_model_id": trace["model_id"],
        "document_embedding_model_revision": trace["model_revision"],
        "reranker_model_id": MODEL_ID, "reranker_model_revision": MODEL_REVISION,
        "instruction": INSTRUCTION, "score_type": "yes_minus_no_logit",
        "device": getattr(scorer, "device", None) if candidates else None,
        "dtype": str(getattr(scorer, "dtype", None)) if candidates else None,
        "max_input_tokens": max_input_tokens, "max_candidates": max_candidates,
        "evidence_limit": evidence_limit, "batch_size": batch_size,
        "candidates_scored": len(reranked), "reranked_candidates": reranked,
        "selected_evidence": reranked[:evidence_limit],
        "elapsed_seconds": monotonic() - started,
    }
    identity = sha256(_canonical({"trace": trace_sha, "model_revision": MODEL_REVISION,
                                  "instruction": INSTRUCTION, "max_candidates": max_candidates,
                                  "evidence_limit": evidence_limit, "max_input_tokens": max_input_tokens})).hexdigest()[:16]
    destination = Path(output_dir) / trace["collection"] / f"{identity}_{time_ns()}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".rerank_", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(_canonical(result) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    return {**result, "reranking_trace_path": str(destination.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank fused candidates with pinned Qwen3-Reranker-0.6B")
    parser.add_argument("retrieval_trace", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--evidence-limit", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    result = rerank(args.retrieval_trace, args.output_dir,
                    max_candidates=args.max_candidates, evidence_limit=args.evidence_limit,
                    batch_size=args.batch_size, max_input_tokens=args.max_input_tokens,
                    device=args.device, local_files_only=args.local_files_only)
    print(json.dumps({"reranking_trace_path": result["reranking_trace_path"],
                      "candidates_scored": result["candidates_scored"],
                      "selected_chunk_ids": [item["chunk_id"] for item in result["selected_evidence"]]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
