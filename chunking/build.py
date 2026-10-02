"""CLI for writing one chunks.jsonl per chunking configuration."""

import argparse
import json
from pathlib import Path

from .core import ALL_METHODS, COMPARISON_METHODS, chunk_document


def build_chunk_files(
    parsed_documents: str | Path,
    output_dir: str | Path,
    *,
    methods=COMPARISON_METHODS,
    target_tokens=512,
    max_tokens=768,
    overlap_tokens=64,
    tokenizer="cl100k_base",
) -> dict[str, int]:
    """Read parsed-document JSONL and write separate method artifacts."""
    source = Path(parsed_documents)
    output = Path(output_dir)
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods) or any(method not in ALL_METHODS for method in methods):
        raise ValueError("Methods must be unique supported chunkers")
    output.mkdir(parents=True, exist_ok=True)
    paths = {method: output / method / "chunks.jsonl" for method in methods}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    counts = dict.fromkeys(methods, 0)
    with source.open(encoding="utf-8") as input_file:
        handles = {method: path.open("w", encoding="utf-8") for method, path in paths.items()}
        try:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                try:
                    document = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number}") from exc
                for method, handle in handles.items():
                    chunks = chunk_document(
                        document,
                        method=method,
                        target_tokens=target_tokens,
                        max_tokens=max_tokens,
                        overlap_tokens=overlap_tokens,
                        tokenizer=tokenizer,
                    )
                    for chunk in chunks:
                        handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    counts[method] += len(chunks)
        finally:
            for handle in handles.values():
                handle.close()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build traceable chunk artifacts from parsed PDF JSONL")
    parser.add_argument("parsed_documents", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=COMPARISON_METHODS)
    parser.add_argument("--target-tokens", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--overlap-tokens", type=int, default=64)
    parser.add_argument("--tokenizer", default="cl100k_base")
    args = parser.parse_args()
    counts = build_chunk_files(
        args.parsed_documents,
        args.output_dir,
        methods=args.methods,
        target_tokens=args.target_tokens,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        tokenizer=args.tokenizer,
    )
    for method, count in counts.items():
        print(f"{method}: {count} chunks -> {args.output_dir / method / 'chunks.jsonl'}")


if __name__ == "__main__":
    main()
