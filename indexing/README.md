# Milvus indexing stage

This stage reads a **complete** embedding artifact and builds one isolated
Milvus collection per embedding file and index configuration. It checks the
embedding manifest and every row checksum before connecting. The collection
stores stable `chunk_id` primary keys, source text, document/page/span citation
metadata, and the model's 1024-dimensional dense vector. BGE-M3 collections
also store learned sparse weights. Both tracks generate BM25 vectors from the
original chunk text using a Milvus function.

First inspect the plan without a Milvus connection:

```bash
conda activate assignment
python -m indexing.build PATH/TO/embeddings.jsonl index_manifests --dry-run
```

For local Milvus 3.x started from `milvus3-local/docker-compose.yaml`, the
default URI is `http://127.0.0.1:19530`:

```bash
python -m indexing.build PATH/TO/embeddings.jsonl index_manifests
```

For a remote service, set `MILVUS_URI` and `MILVUS_TOKEN` if required.

For BGE-M3, the Milvus operator must first set
`dataCoord.targetVecIndexVersion` to at least `10` on the server. Confirm that
server setting, then pass `--target-vec-index-version 10` to the command. This
flag records the operator-confirmed setting; it cannot change the server.
Milvus 3.0 leaves the new index versions opt-in. The client additionally reads
back each index description and requires its requested type, metric, algorithm,
finished state, and indexed row count before declaring the run complete. Very
small collections may remain below Milvus's physical-index threshold; in that
case the command does not issue a completed index manifest.

The index settings start at HNSW `M=16`, `efConstruction=200`, COSINE; BGE
learned sparse `SPARSE_INVERTED_INDEX` / IP / SINDI; and BM25
`SPARSE_INVERTED_INDEX` / BM25 / `DAAT_MAXSCORE`, `k1=1.2`, `b=0.75`.
`--hnsw-m` and `--hnsw-ef-construction` select another isolated collection.
The service must have enough capacity to build HNSW; use a course Milvus
service, not Milvus Lite.

The command upserts by `chunk_id`, flushes and loads the collection, checks
index descriptions, then reads back all chunk IDs and content hashes. Rerunning
the same command is safe. A successful run writes
`index_manifests/<collection>.json` with artifact/index configuration, server
version, physical index descriptions and verification count. It never drops an
existing collection. `MILVUS_DB_NAME` optionally selects a non-default
database.

The repository includes a local Docker Compose configuration under
`milvus3-local/`; `--dry-run` does not need a running service.
