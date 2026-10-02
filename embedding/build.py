"""Validate chunks and write resumable, model-specific embedding artifacts."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

from .models import MODEL_SPECS, count_model_tokens, load_encoder, load_tokenizer


SCHEMA_VERSION = "embedding-v2"


def _json_bytes(value: dict | list) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash(value: dict | list) -> str:
    return sha256(_json_bytes(value)).hexdigest()


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _read_chunks(path: Path, limit: int | None) -> tuple[list[dict], str, str]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    chunks, ids, methods, configs = [], set(), set(), set()
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
                chunk_id = chunk["chunk_id"]
                text = chunk["text"]
                embedding_text = chunk["embedding_text"]
                if not isinstance(chunk_id, str) or not chunk_id or chunk_id in ids:
                    raise ValueError("missing or duplicate chunk_id")
                if not isinstance(text, str) or sha256(text.encode("utf-8")).hexdigest() != chunk["content_hash"]:
                    raise ValueError("content_hash does not match text")
                if not isinstance(embedding_text, str) or not embedding_text.strip():
                    raise ValueError("embedding_text must be nonempty")
                if not isinstance(chunk["source_spans"], list) or not chunk["source_spans"]:
                    raise ValueError("source_spans must be nonempty")
                methods.add(chunk["method"])
                configs.add(chunk["config_hash"])
                ids.add(chunk_id)
                chunks.append(chunk)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid chunk on line {number}: {exc}") from exc
            if limit is not None and len(chunks) >= limit:
                break
    if not chunks:
        raise ValueError("No chunks found")
    if len(methods) != 1 or len(configs) != 1:
        raise ValueError("A run requires one chunking method and one chunking config")
    return chunks, methods.pop(), configs.pop()


def _resolved_device(device: str) -> str:
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if device == "cpu":
        return "cpu"
    import torch

    available = torch.cuda.is_available()
    if device == "cuda" and not available:
        raise ValueError("CUDA requested but unavailable")
    return "cuda" if available else "cpu"


def _load_existing(path: Path, config_hash: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = {}
    with path.open("rb+") as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                stream.truncate(start)  # interrupted append; full earlier rows remain usable
                break
            try:
                row = json.loads(line)
                checksum = row.pop("record_checksum")
                if checksum != _hash(row):
                    raise ValueError("record checksum mismatch")
                if row["config_hash"] != config_hash:
                    raise ValueError("config hash mismatch")
                rows[row["chunk_id"]] = {**row, "record_checksum": checksum}
            except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid embedding record at byte {start}: {exc}") from exc
    return rows


def build_embeddings(
    chunks_path: str | Path,
    output_dir: str | Path,
    *,
    model_name: str,
    batch_size: int = 8,
    device: str = "auto",
    local_files_only: bool = False,
    validate_only: bool = False,
    limit: int | None = None,
    tokenizer=None,
    encoder=None,
) -> dict:
    """Embed one chunking run; validated records are reused after interruption."""
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown model: {model_name}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    source = Path(chunks_path)
    chunks, method, chunk_config_hash = _read_chunks(source, limit)
    spec = MODEL_SPECS[model_name]
    tokenizer = tokenizer if tokenizer is not None else load_tokenizer(spec, local_files_only=local_files_only)
    token_counts = [count_model_tokens(tokenizer, chunk["embedding_text"]) for chunk in chunks]
    too_long = [(chunk["chunk_id"], count) for chunk, count in zip(chunks, token_counts) if count > spec.max_input_tokens]
    if too_long:
        examples = ", ".join(f"{identifier} ({count})" for identifier, count in too_long[:5])
        raise ValueError(f"{len(too_long)} chunks exceed {spec.model_id} limit {spec.max_input_tokens}: {examples}")
    if validate_only:
        return {"status": "validated", "model": spec.model_id, "chunks": len(chunks), "max_model_tokens": max(token_counts)}

    resolved_device = _resolved_device(device)
    config = {
        "schema_version": SCHEMA_VERSION,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "dimension": spec.dimension,
        "max_input_tokens": spec.max_input_tokens,
        "tokenizer_id": spec.model_id,
        "tokenizer_revision": spec.revision,
        "device": resolved_device,
        "precision_policy": "fp16" if spec.name == "bge-m3" and resolved_device == "cuda" else ("auto" if resolved_device == "cuda" else "fp32"),
        "normalization": "l2",
        "dense_metric": "COSINE",
        "sparse_metric": "IP" if spec.has_sparse else None,
        "document_prompt": "",
        "chunking_method": method,
        "chunking_config_hash": chunk_config_hash,
    }
    config_hash = _hash(config)
    run_dir = Path(output_dir) / method / spec.name / spec.revision[:12] / config_hash[:12]
    data_path = run_dir / "embeddings.jsonl"
    manifest_path = run_dir / "manifest.json"
    existing = _load_existing(data_path, config_hash)
    input_hash = _file_hash(source)
    selected_hash = _hash([{"chunk_id": c["chunk_id"], "embedding_text_sha256": sha256(c["embedding_text"].encode("utf-8")).hexdigest()} for c in chunks])

    def manifest(status: str, reused: int, encoded: int) -> dict:
        return {
            "status": status,
            "config": config,
            "config_hash": config_hash,
            "chunks_path": str(source.resolve()),
            "chunks_file_sha256": input_hash,
            "selected_chunk_set_sha256": selected_hash,
            "chunk_count": len(chunks),
            "reused": reused,
            "encoded": encoded,
            "embeddings_path": str(data_path.resolve()),
            "embeddings_file_sha256": _file_hash(data_path) if status == "complete" else None,
        }

    def reusable(chunk: dict) -> bool:
        row = existing.get(chunk["chunk_id"])
        return bool(row and row["embedding_text_sha256"] == sha256(chunk["embedding_text"].encode("utf-8")).hexdigest()
                    and row["content_hash"] == chunk["content_hash"]
                    and row["parsed_document_checksum"] == chunk["parsed_document_checksum"])

    pending = [(c, n) for c, n in zip(chunks, token_counts) if not reusable(c)]
    reused = len(chunks) - len(pending)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_atomic(manifest_path, _json_bytes(manifest("in_progress", reused, 0)) + b"\n")
    encoded = 0
    if pending:
        encoder = encoder if encoder is not None else load_encoder(spec, device=resolved_device, local_files_only=local_files_only)
        with data_path.open("ab") as stream:
            for start in range(0, len(pending), batch_size):
                batch = pending[start:start + batch_size]
                vectors = encoder.encode([chunk["embedding_text"] for chunk, _ in batch])
                if len(vectors) != len(batch):
                    raise ValueError("Encoder returned the wrong batch size")
                for (chunk, token_count), vector in zip(batch, vectors):
                    dense = vector["dense_vector"]
                    sparse = vector["sparse_vector"]
                    if len(dense) != spec.dimension or (spec.has_sparse and not isinstance(sparse, dict)) or (not spec.has_sparse and sparse is not None):
                        raise ValueError("Encoder returned incompatible vectors")
                    row = {
                        "schema_version": SCHEMA_VERSION,
                        "chunk_id": chunk["chunk_id"],
                        "document_id": chunk["document_id"],
                        "source": chunk["source"],
                        "source_checksum": chunk["source_checksum"],
                        "text": chunk["text"],
                        "embedding_text_sha256": sha256(chunk["embedding_text"].encode("utf-8")).hexdigest(),
                        "content_hash": chunk["content_hash"],
                        "parsed_document_checksum": chunk["parsed_document_checksum"],
                        "page_range": chunk["page_range"],
                        "source_spans": chunk["source_spans"],
                        "method": method,
                        "chunking_config_hash": chunk_config_hash,
                        "model_id": spec.model_id,
                        "model_revision": spec.revision,
                        "dimension": spec.dimension,
                        "model_token_count": token_count,
                        "precision": encoder.precision,
                        "dense_vector": dense,
                        "sparse_vector": sparse,
                        "config_hash": config_hash,
                    }
                    row["record_checksum"] = _hash(row)
                    stream.write(_json_bytes(row) + b"\n")
                    existing[row["chunk_id"]] = row
                    encoded += 1
                stream.flush()
                os.fsync(stream.fileno())
                _write_atomic(manifest_path, _json_bytes(manifest("in_progress", reused, encoded)) + b"\n")

    # Remove stale or duplicate journal entries and put records in source order.
    final_rows = [existing[chunk["chunk_id"]] for chunk in chunks]
    _write_atomic(data_path, b"".join(_json_bytes(row) + b"\n" for row in final_rows))
    result = manifest("complete", reused, encoded)
    _write_atomic(manifest_path, _json_bytes(result) + b"\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and embed chunk artifacts")
    parser.add_argument("chunks_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model", required=True, choices=MODEL_SPECS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--limit", type=int, help="Embed first N chunks for a smoke run")
    args = parser.parse_args()
    result = build_embeddings(
        args.chunks_jsonl, args.output_dir, model_name=args.model,
        batch_size=args.batch_size, device=args.device,
        local_files_only=args.local_files_only, validate_only=args.validate_only,
        limit=args.limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
