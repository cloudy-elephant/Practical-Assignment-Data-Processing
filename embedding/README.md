# Embedding stage

This stage reads one chunking run's `chunks.jsonl` and writes model-specific
embedding artifacts. BGE-M3 produces a 1024-dimensional dense vector plus
learned sparse token weights; Qwen3-Embedding-0.6B produces a 1024-dimensional
dense vector. Both use the **same `embedding_text`** from each chunk. Model
revisions are pinned in [`models.py`](models.py).

Run from the repository root in the `assignment` environment:

```bash
conda activate assignment
python -m embedding.build artifacts/chunks/layout_paragraph/chunks.jsonl embeddings --model bge-m3 --validate-only
python -m embedding.build artifacts/chunks/layout_paragraph/chunks.jsonl embeddings --model qwen3-0.6b --validate-only
python -m embedding.build artifacts/chunks/layout_paragraph/chunks.jsonl embeddings --model bge-m3 --batch-size 4
python -m embedding.build artifacts/chunks/layout_paragraph/chunks.jsonl embeddings --model qwen3-0.6b --batch-size 4
```

Use`--validate-only` downloads/loads the pinned tokenizer as needed, checks the
actual token count including special tokens against each model's input limit,
and writes no embeddings. Use `--local-files-only` for a pre-cached offline run.
Use `--limit 1` for a small real-model smoke run, or `--device cpu` to force CPU.
Model weights are downloaded on the first full run and can be large.
With the pinned `transformers==4.57.6`, BGE-M3's `.bin` checkpoint requires
PyTorch 2.6 or newer. The Intel macOS `assignment` environment pins PyTorch
2.2.2, so BGE-M3 inference belongs in the course GPU environment; tokenizer
validation and Qwen3 inference work locally.

The output path includes chunking method, model, pinned revision, and embedding
config hash. Each `embeddings.jsonl` row retains source text, page range and
source spans for indexing/citation, plus a vector, model token count and record
checksum. BGE sparse weights are keyed by **token ID strings**, ready for a
Milvus sparse vector mapping after converting keys to integers. `manifest.json`
records the input checksum, selected chunk-set checksum, configuration, output
checksum and completion status.

Interrupted jobs append completed batches and resume by reusing records whose
chunk ID, embedding text, content hash, parsed document checksum and config
match. A complete run compacts the journal to one row per current chunk. Do not
index a run unless its manifest says `complete` and its file checksum matches.
Any over-limit chunk fails before model weights load; rerun chunking with a
smaller maximum instead of silently truncating.

For later retrieval, Qwen document encoding uses an empty prompt. Its query
side must use the fixed instruction in [`models.py`](models.py). BGE query
encoding must use the corresponding BGE model's dense and sparse outputs.
Gemini is an optional later extension and is not part of this core stage.
