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

For BGE-M3, the server requires `dataCoord.targetVecIndexVersion >= 10`.
The local [`user.yaml`](../milvus3-local/user.yaml) sets it to `10`; a remote
service requires its operator to configure it separately. Confirm that the
running server has applied the setting, then run:

```bash
python -m indexing.build PATH/TO/BGE_M3/embeddings.jsonl index_manifests --target-vec-index-version 10
```

This flag records the operator-confirmed setting; it cannot change or
automatically verify the server configuration. A valid Compose configuration
or a healthy container alone does not verify that SINDI index construction
succeeds; run an actual BGE-M3 index build against the service.
Milvus 3.0 leaves the new index versions opt-in. The client additionally reads
back each index description and requires its requested type, metric, algorithm,
finished state, and indexed row count before declaring the run complete. These
are API checks, not independent verification of physical index files. Milvus
may mark small skipped segments as finished and include their rows in the
indexed count, so a complete manifest alone does not prove physical indexing.

Milvus defaults to **1024 rows per segment** for
[`indexCoord.segment.minSegmentNumRowsToEnableIndex`](https://milvus.io/docs/configure_indexcoord.md).
Because this assignment uses few documents, the local
[`user.yaml`](../milvus3-local/user.yaml) lowers the threshold to **1 row per
segment** while retaining the same indexing method. The main
[`docker-compose.yaml`](../milvus3-local/docker-compose.yaml) directly mounts
it at `/milvus/configs/user.yaml`, including when started with
`-f milvus3-local/docker-compose.yaml`; no override file is needed. For an
existing local service, apply both settings by recreating the container:

```bash
docker compose -f milvus3-local/docker-compose.yaml up -d --force-recreate --wait standalone
```

For collections created with the old threshold, use a new
`--collection-prefix` (for example, `assignment_small`) to build a fresh
collection. Remote servers require the operator to set the threshold separately;
this client does not configure it. Before comparing indexes, separately verify
the relevant segments' index files and loading status.

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
version, API-reported index descriptions and verification count. It never drops an
existing collection. `MILVUS_DB_NAME` optionally selects a non-default
database.

The repository includes a local Docker Compose configuration under
`milvus3-local/`; `--dry-run` does not need a running service.
