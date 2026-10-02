# Financial Document RAG Assignment — Pipeline Design v0.1

## 1. Scope and design principles

The primary input is the English financial-report PDF used in the existing assignment. The pipeline must run from raw PDF to an answer with a traceable source. It retains the existing PyMuPDF/OCR extraction workflow, but replaces page-only indexing with explicit chunking and modern retrieval representations. Questions, a labelled evaluation set, grading weights, and student TODOs will be designed later.

The core implementation is delivered as a Git repository and runs in a course-provided GPU environment connected to a persistent Milvus 3.0 service. Students do not need to install a GPU stack or Milvus on their personal computers. Gemini Embedding 2 is an optional multimodal extension with a `YOUR_API_KEY` placeholder. Tables and figures remain traceable to their source pages; reliable cell-level table reasoning and visual retrieval are outside the core scope.

```text
PDF → extraction/OCR → normalized document → chunks → representations
    → Milvus indexes → retrieval → fusion → reranking → evidence
    → generator → answer with citations
```

Each stage writes a versioned artifact. The next stage reads that artifact rather than repeating upstream work. This makes it possible to replace one component and compare results while holding the others fixed.

## 2. Data preprocessing

**Input:** PDF bytes and a document manifest containing document ID, filename, checksum, and source metadata.

1. Route text-based pages through the existing PyMuPDF extractor and scanned pages through the existing OCR path. A page with little or unusable extracted text is flagged for review or OCR fallback.
2. Preserve the existing page, paragraph, line, text, bbox, and OCR confidence information. Assign stable IDs to pages and paragraphs. Do not flatten a page into one string at this stage.
3. Normalize text conservatively: remove duplicate extraction artifacts and empty blocks; retain financial signs, decimal points, years, units, and table-like spacing. Preserve both original text and normalized text when a transformation may affect citation or exact matching.
4. Record quality flags for empty pages, low OCR confidence, suspected multi-column reading-order errors, and unusually long paragraphs. Keep the original PDF and page coordinates for audit and citation.

**Output:** `parsed_documents.jsonl`, with a document → page → paragraph/line hierarchy and source coordinates. This is an extension/adapter around the old preprocessing workflow, not a replacement for it.

## 3. Chunking

All methods consume the same parsed documents and emit the same chunk schema. The old **one-page-per-chunk** method remains a historical control. Three methods form the assignment comparison:

| Method | Boundary rule | Role |
| --- | --- | --- |
| Fixed token | Fixed token window with optional overlap | Simple baseline |
| Paragraph-aware recursive | Merge paragraphs, then split oversized paragraphs by sentence/token | General text method |
| Layout-aware paragraph | Use page, reading order, paragraph adjacency and bbox before token-limit fallback | Main method exploiting preprocessing output |

The initial target is approximately 512 tokens per chunk, with a documented maximum and overlap policy. These are starting settings, not claims of optimality. The selected tokenizer is recorded; before embedding, every chunk is checked against the actual tokenizer and input limit of each model. Text spanning pages or columns must carry an accurate page range and source-span mapping. Layout-aware grouping must use only structure that the extractor actually emits; heading levels, table cells, and manually assigned `group_id` values cannot be assumed.

**Output:** one `chunks.jsonl` per chunking configuration. Each record contains `chunk_id`, `document_id`, `text`, `embedding_text`, `page_range`, source paragraph IDs, bboxes/source spans, token counts, method/config version, and a content hash. `text` remains suitable for display and citation; `embedding_text` may include compact source context, but any added context must be recorded.

## 4. Embedding and retrieval representations

For a controlled comparison, encode the **same chunk set** with each model. Keep model and dimension in the run ID; do not mix vectors from different models in one vector field.

| Track | Document representation | Query representation when retrieval runs | Status |
| --- | --- | --- | --- |
| BGE-M3 | 1024-d dense vector and learned sparse token weights | Corresponding dense and sparse outputs | Core |
| Qwen3-Embedding-0.6B | 1024-d dense vector | Same model with a fixed retrieval instruction | Core comparison |
| Gemini Embedding 2 | Text dense vector; later page/image embedding | Matching API task configuration | Optional |

BGE-M3's multi-vector output can be discussed and retained as a later experiment, but indexing all token vectors is not necessary for the core path. Gemini uses `YOUR_API_KEY` in example configuration; without a real key, its extension is documented but is not counted as a successful runtime test. Fake/random embeddings are never used as a retrieval fallback.

**Output:** embedding artifacts keyed by `chunk_id`, with model ID/revision, dimension, tokenizer/configuration, normalization/metric choice, and checksum. An embedding job can resume and skip unchanged chunks.

## 5. Milvus 3.0 indexing

Use separate experiment collections or clearly versioned vector fields so BGE-M3, Qwen3, chunking methods, and index settings cannot be confused. The primary key is a stable `chunk_id`; `document_id`, source pages, text, and citation metadata accompany every indexed chunk.

| Retrieval path | Source | Physical index / scoring |
| --- | --- | --- |
| Dense semantic | BGE-M3 or Qwen3 dense | Milvus HNSW / cosine |
| Learned sparse | BGE-M3 sparse | Milvus `SPARSE_INVERTED_INDEX`, IP, **SINDI** |
| Lexical | Original chunk text | Milvus full-text inverted index, BM25 |

The BGE-M3 collection supports all three paths. The Qwen3 collection supports dense and BM25. A Gemini collection is created only if that extension runs. Start HNSW with documented `M`, `efConstruction`, and query `ef` values, then tune against an exact dense FLAT reference if the collection is large enough for ANN behaviour to matter. Compare SINDI with DAAT_MAXSCORE on the **same BGE-M3 sparse vectors**; this isolates index execution from representation quality. Measure index build time, footprint, query latency, and agreement with an exact sparse-score reference.

The deployment must enable the Milvus 3.0 index version required for SINDI before index creation, and must verify the index actually built with the requested algorithm. A successful client call alone is insufficient evidence. Milvus 3.0 release notes state that new index versions remain opt-in.

## 6. Retrieval, fusion, and generation

At query time, use the matching document model and its query-side configuration. BGE-M3 runs dense, learned sparse, and BM25 searches in parallel; Qwen3 runs dense and BM25. Apply identical document/page filters to all paths. Merge candidates by stable `chunk_id`, retain each path's rank and score, and fuse ranks with RRF. Pass a bounded candidate set to the same locally hosted Qwen3-Reranker-0.6B in every experiment, then select a bounded evidence set for generation. Keep the reranker model/configuration fixed when comparing retrieval paths.

The generator receives the user question and evidence chunks with `document_id`, page, and source spans. It returns an answer with citations to the underlying pages/chunks, or states that the evidence is insufficient. The generation model is a configurable, course-provided chat endpoint, held fixed across retrieval experiments. A small smoke query verifies the pipeline now; the actual financial-question set and answer-quality study are specified later.

**Output:** a retrieval trace containing per-path candidates/ranks, fused ranks, reranker scores, selected evidence, model/config versions, timing, final answer, and citations. This trace is necessary for debugging and later evaluation.

## 7. Runnable paths and comparison order

1. **Historical control:** old page-level indexing with its Milvus IVF_FLAT + Elasticsearch keyword retrieval and RRF. Record its actual embedding model rather than assuming every old index used the same one.
2. **Core BGE-M3:** selected chunker → BGE-M3 dense + sparse → HNSW + SINDI + BM25 → RRF → fixed reranker → generator.
3. **Core Qwen3:** same chunks → Qwen3 dense → HNSW + BM25 → RRF → same reranker and generator.
4. **Optional Gemini:** same textual chunks initially; page/image inputs only when a visual-retrieval exercise is added.

First compare chunkers while holding embedding, indexing, and retrieval settings fixed. Next fix the chunker and compare dense models. Then add BM25, learned sparse, SINDI, fusion, and reranking one change at a time. Later evaluation can add labelled financial questions and report retrieval quality, citation accuracy, answer quality, latency, and cost. Index-algorithm comparisons should primarily report efficiency and recall relative to exact search; they should not be presented as a new embedding-quality comparison.

## 8. Deliverables of a complete run

`source_manifest` → `parsed_documents` → `chunks` → `embeddings` → Milvus collection/index manifests → `retrieval_trace` → `answer_with_citations`.

A run is complete when a PDF can be ingested, each core index can be queried, the selected evidence resolves to the source PDF and page, and the generator can produce an answer or an explicit insufficient-evidence response. Each artifact stores its upstream checksum and configuration so a changed parser, chunker, model, or index leads to a new reproducible run.

## 9. Delivery and execution environment

The Git repository is the source of truth. Stage-specific Python modules implement preprocessing, chunking, embedding, indexing, retrieval, and generation. Teaching notebooks call those modules and display intermediate artifacts; they do not contain a second, divergent pipeline implementation. Configuration files pin model revisions, Python dependencies, chunking parameters, Milvus connection settings, and experiment IDs. Credentials are supplied through environment variables and are not committed.

The course GPU environment hosts notebooks/scripts and local embedding/reranking models. Milvus 3.0 runs as a persistent course service with a documented endpoint and isolated experiment collections. Before students run the assignment, an environment check verifies GPU availability, model access, Milvus connectivity/version, and that the new index version for SINDI is enabled. Artifacts and logs are written to persistent storage so interrupted runs can resume.

Colab is not a required execution path. A future Colab notebook may act only as a client to the same remote Milvus service; Milvus Lite cannot replace the required HNSW and SINDI index experiments because its vector index implementation is FLAT-only.

## References

- [Existing DSTA assignment](https://github.com/6estates/dsta-assignment/tree/main)
- [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3)
- [Qwen3-Embedding-0.6B model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [Gemini Embedding 2 documentation](https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2)
- [Milvus sparse inverted index and SINDI](https://milvus.io/docs/sparse-inverted-index.md)
- [Milvus 3.0 release notes](https://milvus.io/docs/release_notes.md)
