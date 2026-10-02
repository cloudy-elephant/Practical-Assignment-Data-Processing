# Chunking pipeline

The chunkers read the `parsed-document-v1` JSONL produced by `pdf_processing`.
They all emit the same record schema, so embeddings can compare the same source
document under different chunk boundaries.

```bash
conda activate assignment
python -m pdf_processing.extract data/financial/ATSAR2023+bursa.pdf artifacts/parsed_documents.jsonl
python -m chunking.build artifacts/parsed_documents.jsonl artifacts/chunks
```

The default run writes `fixed_token/chunks.jsonl`,
`paragraph_recursive/chunks.jsonl`, and `layout_paragraph/chunks.jsonl` under
`artifacts/chunks/`. Add `--methods fixed_token paragraph_recursive
layout_paragraph page_control` to include the old one-page-per-chunk control.

Defaults: `cl100k_base` tokenizer, 512-token target, 768-token maximum, and
64-token overlap for **fixed-token only**. The paragraph and layout methods
have zero overlap. Layout grouping uses only page number, extractor reading
order, paragraph adjacency, and bounding boxes; it does not infer headings or
table cells. The historical page control can exceed the maximum by design.
The `cl100k_base` vocabulary is stored under `resources/` with the SHA-256
published in `tiktoken`, so the default run does not download tokenizer data.

Each chunk stores its text, source paragraph IDs, character spans within those
paragraphs, page range, bounding boxes, tokenizer count, hashes, and full
chunking configuration. It also records the parsed-document checksum and source
PDF details for reproducibility and citations. `embedding_text` currently equals display `text`; no
unrecorded context is added. Before embedding, check each chunk again with the
actual embedding model tokenizer and input limit. This tokenizer is a stable
chunking yardstick, not a claim that BGE-M3 and Qwen3 count tokens identically.

Other options: `--target-tokens`, `--max-tokens`, `--overlap-tokens`, and
`--tokenizer` (a `tiktoken` encoding name).
