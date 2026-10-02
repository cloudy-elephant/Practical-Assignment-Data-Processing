# Reranking stage

This stage reads a completed retrieval trace and scores its fused candidates
with the same pinned `Qwen/Qwen3-Reranker-0.6B` model for both the BGE-M3 and
Qwen3 embedding tracks. It uses a fixed financial-report instruction and the
model's yes/no relevance logits. Higher raw scores mean greater relevance.

After retrieval, run from the repository root in the `assignment` environment:

```bash
conda activate assignment
python -m reranking.build PATH/TO/retrieval_trace.json reranking_traces \
  --max-candidates 20 --evidence-limit 5 --batch-size 4
```

The command prints `reranking_trace_path`. The JSON output contains every
scored candidate in reranked order and `selected_evidence` for the next
generation stage. Each candidate retains its `chunk_id`, original fused rank,
per-path rank and score, source text, document, pages and source spans. The
artifact also records the retrieval trace checksum, index manifest checksum,
reranker revision, instruction, token limit and runtime settings.

By default each question/passage pair may use at most 2048 model tokens. The
command counts the full formatted pair, including the model's fixed prefix
and suffix, before loading weights or scoring. It fails on an over-limit pair
instead of silently truncating it. `--max-input-tokens` can be raised up to
8192 when the course GPU has enough memory. `--device cpu` forces local CPU;
`--local-files-only` uses cached model files. Model weights download on the
first online run.

The model and instruction stay fixed across retrieval-track comparisons.
`--max-candidates` bounds reranker work; `--evidence-limit` bounds the evidence
passed to generation. No answer is generated at this stage. If retrieval
returned zero candidates, the stage emits an empty evidence list without
loading the model.
