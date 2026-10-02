# Retrieval and RRF fusion

This stage reads a **complete** `index_manifests/<collection>.json` and queries
its matching Milvus collection. Qwen3 searches `dense_vector` and
`bm25_sparse`; BGE-M3 additionally searches `learned_sparse`. Every path gets
the same optional document/page filter. A page filter includes chunks whose
page range spans that page.

Run in the `assignment` environment after building the collection:

```bash
conda activate assignment
python -m retrieval.search index_manifests/COLLECTION.json \
  "What was the revenue in 2023?" retrieval_traces \
  --document-id YOUR_DOCUMENT_ID --path-limit 20 --candidate-limit 20
```

The local default URI is `http://127.0.0.1:19530`. Set `MILVUS_URI` and
`MILVUS_TOKEN` when connecting to a remote service that needs them.
`MILVUS_DB_NAME` optionally selects a non-default database. `--page 5` limits
results to chunks overlapping page 5. `--hnsw-ef 64` sets dense HNSW query
effort and must be at least `--path-limit`. `--rrf-k 60` sets the rank-fusion
constant. On an offline machine with cached model files, add
`--local-files-only`. Qwen3 query encoding works on the local `assignment`
CPU environment; BGE-M3 requires the compatible course GPU environment
described in the embedding README.

Qwen3 document vectors were generated without an instruction. Query vectors
use the fixed finance-retrieval instruction from `embedding/models.py`, with
the model's `Instruct: ...\nQuery:` prefix. BGE-M3 generates matching dense
and learned sparse query vectors. Query text is checked against the pinned
model tokenizer before encoding. BM25 receives the original question text;
Milvus applies the analyzer and BM25 function configured during indexing.

The command checks the collection schema and physical index descriptions,
searches each path in parallel, validates source text and filters, then
combines path ranks by reciprocal rank fusion. It writes one JSON trace under
`retrieval_traces/<collection>/`. The trace contains each path's rank, score
and latency plus the fused candidates and citation metadata. Scores from
different paths are retained separately; RRF uses ranks because their score
scales differ. The [reranking stage](../reranking/README.md) scores these
candidates with a fixed Qwen3-Reranker-0.6B and selects evidence for generation.
