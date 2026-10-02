#Document RAG Pipeline

[中文版本](README_CN.md)

This repository implements a traceable retrieval-augmented generation (RAG) pipeline for English financial-report PDFs. It extracts and chunks text from source PDFs, builds a Milvus index, retrieves and reranks evidence for a question, and generates an answer with source citations.

The pipeline consists of the root-level `pdf_processing/`, `chunking/`, `embedding/`, `indexing/`, `retrieval/`, `reranking/`, and `generation/` modules. Indexing and retrieval require an accessible Milvus service; answer generation also requires a chat-model endpoint. The repository does not include a prebuilt index.

## Quick start: environment and Milvus

Run these commands from the **repository root**. First follow the official instructions to install [Conda](https://docs.conda.io/projects/conda/en/latest/), [Docker Engine with Compose on Ubuntu](https://docs.docker.com/engine/install/ubuntu/), or [Docker Desktop on macOS](https://docs.docker.com/desktop/setup/install/mac-install/), and start Docker. Then choose one set of Tesseract commands for your operating system:

Ubuntu / Debian:
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
```

macOS with Homebrew:
```bash
brew install tesseract
```

Create a Python 3.11 environment:

```bash
conda create -n financial-rag python=3.11 -y
conda activate financial-rag
```

Before installing the project dependencies, choose one PyTorch command for your hardware.

**Linux x86_64 with CUDA 12.4**:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Linux CPU:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

For another CUDA version or Apple Silicon macOS, use the matching command in the [official PyTorch installation list](https://docs.pytorch.org/get-started/previous-versions/). On Intel macOS, skip the PyTorch installation command: `requirement.txt` installs its pinned CPU version. That environment can run the Qwen3 example below but cannot load BGE-M3.

Then install project dependencies and check OCR in the activated environment:

```bash
python -m pip install -r requirement.txt
python -c 'import torch; print(torch.__version__, "CUDA available:", torch.cuda.is_available())'
tesseract --list-langs
```

Start the repository's Milvus 3.x service and wait for its container health checks. The Qwen3 path below uses HNSW/COSINE and BM25, so it does not require the BGE-M3 SINDI server setting.

```bash
docker compose -f milvus3-local/docker-compose.yaml up -d --wait
docker compose -f milvus3-local/docker-compose.yaml ps
curl -fsS http://127.0.0.1:9091/healthz
export MILVUS_URI=http://127.0.0.1:19530
```

The first run downloads the pinned tokenizers and model weights. For a remote Milvus service, replace `MILVUS_URI` with its address and set `MILVUS_TOKEN` and `MILVUS_DB_NAME` if needed.

## Worked example: PDF to retrieval results

This example uses `data/financial/ATSAR2023+bursa.pdf`, `layout_paragraph` chunking, `qwen3-0.6b` embeddings, Milvus HNSW/COSINE plus BM25 retrieval, and `Qwen/Qwen3-Reranker-0.6B` reranking. After setup, run the following commands **in the same terminal**. The script reads each stage's output path from its JSON result and finally prints the fused retrieval candidates and the top five reranked evidence passages.

```bash
set -euo pipefail
export MILVUS_URI="${MILVUS_URI:-http://127.0.0.1:19530}"
run_dir=artifacts/sample_run
mkdir -p "$run_dir"

python -m pdf_processing.extract data/financial/ATSAR2023+bursa.pdf "$run_dir/parsed_documents.jsonl" --ocr auto
python -m chunking.build "$run_dir/parsed_documents.jsonl" "$run_dir/chunks" --methods layout_paragraph

python -m embedding.build "$run_dir/chunks/layout_paragraph/chunks.jsonl" "$run_dir/embeddings" --model qwen3-0.6b --validate-only
python -m embedding.build "$run_dir/chunks/layout_paragraph/chunks.jsonl" "$run_dir/embeddings" --model qwen3-0.6b --batch-size 4 | tee "$run_dir/embedding_run.json"
EMBEDDINGS_PATH=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["embeddings_path"])' "$run_dir/embedding_run.json")

python -m indexing.build "$EMBEDDINGS_PATH" "$run_dir/index_manifests" --collection-prefix financialrag --dry-run
python -m indexing.build "$EMBEDDINGS_PATH" "$run_dir/index_manifests" --collection-prefix financialrag | tee "$run_dir/index_run.json"
INDEX_MANIFEST_PATH=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["index_manifest_path"])' "$run_dir/index_run.json")

python -m retrieval.search "$INDEX_MANIFEST_PATH" "What was the financial year end reported by the company?" "$run_dir/retrieval_traces" --path-limit 20 --candidate-limit 20 | tee "$run_dir/retrieval_run.json"
RETRIEVAL_TRACE_PATH=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["trace_path"])' "$run_dir/retrieval_run.json")
python -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["fused_candidates"], ensure_ascii=False, indent=2))' "$RETRIEVAL_TRACE_PATH"

python -m reranking.build "$RETRIEVAL_TRACE_PATH" "$run_dir/reranking_traces" --max-candidates 20 --evidence-limit 5 --batch-size 4 | tee "$run_dir/reranking_run.json"
RERANKING_TRACE_PATH=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["reranking_trace_path"])' "$run_dir/reranking_run.json")
python -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["selected_evidence"], ensure_ascii=False, indent=2))' "$RERANKING_TRACE_PATH"
```

`fused_candidates` contains the RRF-fused retrieval results. `selected_evidence` contains the reranked passages ready for answer generation. The full records are saved at `RETRIEVAL_TRACE_PATH` and `RERANKING_TRACE_PATH`. This example stops after evidence selection, so it needs no chat endpoint; see “Answer a question” below to generate an answer. Indexing verifies Milvus's physical indexes, and [Milvus may not build one for a small dataset](https://milvus.io/docs/performance_faq.md). If the sample document is too small, add more input documents and rebuild; a `--dry-run` result alone does not mean an index was built.

## End-to-end workflow

```text
Indexing: PDF → PyMuPDF / OCR when needed → parsed_documents.jsonl
              → three chunking methods → chunks.jsonl
              → BGE-M3 or Qwen3-Embedding → embeddings.jsonl + manifest.json
              → Milvus 3.x collection + index manifest

Q&A: question → matching query encoder + multi-path Milvus search → RRF fusion
              → Qwen3-Reranker → top five evidence passages
              → OpenAI-compatible chat endpoint
              → answer, [1]-style citations, and generation trace
```

Each stage reads the preceding stage's artifact. The PDF hash, `chunk_id`, page numbers, paragraph character spans, and checksums flow through the pipeline so an answer can be traced back to the source PDF. The current code validates citation numbers and source metadata; it does not automatically verify that every factual claim in an answer is supported by its cited evidence.

## Project structure

| Path | Purpose | Main output |
| --- | --- | --- |
| [`data/`](data/README.md) | Financial-report PDFs and input samples | Input data |
| [`pdf_processing/`](pdf_processing/README.md) | Extract text page by page, apply OCR when needed, and retain paragraphs, lines, and coordinates | `parsed_documents.jsonl` |
| [`chunking/`](chunking/README.md) | Generate different chunking runs with a shared schema | One `chunks.jsonl` per method |
| [`embedding/`](embedding/README.md) | Check model input lengths and generate vectors with resumable processing | `embeddings.jsonl`, `manifest.json` |
| [`indexing/`](indexing/README.md) | Build a Milvus collection and verify its indexes and data | Collection, index manifest |
| [`retrieval/`](retrieval/README.md) | Search multiple paths and fuse their ranks with RRF | Retrieval trace |
| [`reranking/`](reranking/README.md) | Score candidates with a fixed reranker and select evidence | Reranking trace |
| [`generation/`](generation/README.md) | Call a chat endpoint and save the answer and citations | Generation trace |
| `tests/` | Unit tests and tests with mocked dependencies for each stage | Test results |

### Models and retrieval paths

| Embedding model | Document representation | Milvus retrieval paths |
| --- | --- | --- |
| `bge-m3` (`BAAI/bge-m3`) | 1024-dimensional dense + learned sparse | HNSW/COSINE, learned sparse/SINDI/IP, full-text BM25 |
| `qwen3-0.6b` (`Qwen/Qwen3-Embedding-0.6B`) | 1024-dimensional dense | HNSW/COSINE, full-text BM25 |

Model revisions are pinned in [`embedding/models.py`](embedding/models.py). Both tracks can process the same chunks, but they produce separate embedding artifacts and collections. Milvus builds BM25 vectors from the original chunk text. At query time, the matching model encodes the question for dense and learned sparse search, while BM25 receives the original question text. The Qwen3 query encoder uses a fixed retrieval instruction. Because path scores have different scales, RRF combines ranks instead; by default, each hit contributes `1 / (60 + rank)`. The original score and rank from every path are retained.

## Additional environment notes

- The dependency file is [`requirement.txt`](requirement.txt) (singular). A Linux GPU environment is recommended for full model inference. For offline runs, cache the pinned weights and tokenizers beforehand and add `--local-files-only`.
- OCR requires the Tesseract executable and English language data. By default, `--ocr auto` sends only pages with fewer than 40 extracted characters to OCR. For a digital PDF, `--ocr never` is useful for an initial check.
- Indexing and retrieval require an accessible **Milvus 3.x** service. The local service can be started with `milvus3-local/docker-compose.yaml`. The default URI is `http://127.0.0.1:19530`; set `MILVUS_URI` for a remote service. The BGE-M3 SINDI path also requires `dataCoord.targetVecIndexVersion >= 10` on the server, which an operator must confirm.
- Generation requires an OpenAI-compatible chat endpoint and a model name. API keys are supplied through environment variables and are never saved in artifacts.

On Intel macOS, the dependency file pins PyTorch 2.2.2 for local CPU checks. With the current `transformers==4.57.6`, this version cannot load BGE-M3's `.bin` weights. Parsing, chunking, tokenizer validation, and some Qwen3 checks can run locally; the complete BGE-M3 track requires a compatible environment.

## Build an index from a PDF

The following example uses [`data/financial/ATSAR2023+bursa.pdf`](data/financial/ATSAR2023+bursa.pdf). `artifacts/`, `embeddings/`, and `index_manifests/` are example output directories. Replace `PATH/TO/...` in later commands with the actual paths printed by the programs.

**1. Parse the PDF.** PyMuPDF extracts text, and pages with little extractable text are passed to Tesseract automatically. The output is one JSONL document record containing a PDF SHA-256-based document ID, page/paragraph/line IDs, original and normalized text, bounding boxes, extraction methods, and quality flags.

```bash
python -m pdf_processing.extract data/financial/ATSAR2023+bursa.pdf artifacts/parsed_documents.jsonl
```

**2. Chunk the document.** The default run writes `fixed_token`, `paragraph_recursive`, and `layout_paragraph` outputs. It uses `cl100k_base` with a 512-token target and a 768-token maximum. Only fixed-token windows use the 64-token overlap. The paragraph method splits oversized paragraphs. The layout method also considers page numbers, reading order, and bounding boxes; it does not infer table cells or heading hierarchy.

```bash
python -m chunking.build artifacts/parsed_documents.jsonl artifacts/chunks
```

Add `--methods fixed_token paragraph_recursive layout_paragraph page_control` to generate whole-page chunks; these may exceed the token maximum. Each chunk's `source_spans` point to source paragraphs and character ranges. Currently, `embedding_text` equals the displayed `text`.

**3. Embed the chunks.** The example below uses `layout_paragraph`. Both models can process the same `chunks.jsonl`, producing separate embedding artifacts. Check lengths with each model's tokenizer before encoding; an overlong input raises an error rather than being silently truncated. The command prints `embeddings_path`. The actual file is stored under directories grouped by chunking method, model revision, and configuration hash.

```bash
python -m embedding.build artifacts/chunks/layout_paragraph/chunks.jsonl embeddings --model qwen3-0.6b --validate-only
python -m embedding.build artifacts/chunks/layout_paragraph/chunks.jsonl embeddings --model qwen3-0.6b --batch-size 4

# Run BGE-M3 in a compatible GPU environment
python -m embedding.build artifacts/chunks/layout_paragraph/chunks.jsonl embeddings --model bge-m3 --validate-only
python -m embedding.build artifacts/chunks/layout_paragraph/chunks.jsonl embeddings --model bge-m3 --batch-size 4
```

Use `--limit 1` for a single-chunk smoke run with real model weights; this does not index the full report. Rerunning the same configuration after an interruption reuses verified records. Proceed to indexing only when `manifest.json` reports `status` as `complete` and its file checksum matches.

**4. Build the Milvus index.** First use `--dry-run` to validate the embedding artifact and index plan without connecting to Milvus. Once connected, the program creates an isolated collection for each embedding artifact and index configuration, upserts records by `chunk_id`, and verifies the physical index descriptions, indexed row count, and data read back from Milvus. The printed `index_manifest_path` is the entry point for Q&A.

```bash
python -m indexing.build PATH/TO/embeddings.jsonl index_manifests --dry-run

# export MILVUS_URI=http://YOUR_MILVUS_HOST:19530  # Set for a remote service
# export MILVUS_TOKEN=YOUR_TOKEN                 # Set if the service requires a token
# export MILVUS_DB_NAME=YOUR_DATABASE           # Defaults to default
python -m indexing.build PATH/TO/embeddings.jsonl index_manifests
```

For **BGE-M3**, after an operator confirms that the server's index version is at least 10, add `--target-vec-index-version 10` to the indexing command. This option only records the confirmed setting; it **does not change the Milvus server configuration**. If the dataset is too small for Milvus to build a verifiable physical index, the program does not write a `complete` index manifest.

## Answer a question

The simplest entry point is [`generation.pipeline.ask`](generation/pipeline.py): pass a completed index manifest, a question, and an output directory. It runs retrieval, reranking, and generation in sequence. You can also open [`generation/rag_demo.ipynb`](generation/rag_demo.ipynb) to inspect candidates, evidence, and answers stage by stage; the notebook does not build an index.

```bash
# export MILVUS_URI=http://YOUR_MILVUS_HOST:19530  # Set for a remote service
# export MILVUS_TOKEN=YOUR_TOKEN                    # Set if the service requires a token
export RAG_CHAT_MODEL=YOUR_CHAT_MODEL
export RAG_CHAT_BASE_URL=http://YOUR_CHAT_ENDPOINT/v1  # Omit for the official OpenAI endpoint
export RAG_CHAT_API_KEY=YOUR_API_KEY                    # OPENAI_API_KEY also works
export INDEX_MANIFEST_PATH=PATH/TO/index_manifests/COLLECTION.json
```

```python
import os
from generation.pipeline import ask

result = ask(
    os.environ["INDEX_MANIFEST_PATH"],
    "What was the financial year end reported by the company?",
    "artifacts/rag_queries",
)
print(result["answer"])
print(result["citations"])
print(result["answer_path"])
```

To inspect each stage or change its parameters, run the commands separately:

```bash
python -m retrieval.search PATH/TO/index_manifests/COLLECTION.json \
  "What was the financial year end reported by the company?" retrieval_traces \
  --path-limit 20 --candidate-limit 20

python -m reranking.build PATH/TO/retrieval_trace.json reranking_traces \
  --max-candidates 20 --evidence-limit 5

python -m generation.answer PATH/TO/reranking_trace.json generation_traces
```

The preceding command prints the trace path needed by the next one. By default, retrieval takes 20 hits per path and retains at most 20 candidates after RRF fusion. `--document-id` and `--page` narrow the search; a page filter also includes chunks spanning that page. The pinned `Qwen/Qwen3-Reranker-0.6B` reranks up to 20 candidates and selects the top five as evidence. The generator receives only the question and those passages and must cite factual claims with numbers such as `[1]`. If there is no evidence, or the model explicitly returns `INSUFFICIENT_EVIDENCE`, the result states that the evidence is insufficient.

## Artifacts and traceability

| Artifact | What to inspect |
| --- | --- |
| `parsed_documents.jsonl` | PDF hash, page numbers, extraction method, quality flags, paragraph/line coordinates |
| `<method>/chunks.jsonl` | `chunk_id`, text, token count, page range, `source_spans`, configuration |
| `embeddings.jsonl` + `manifest.json` | Model/revision, vector dimensions, input/output checksums, completion status |
| `index_manifests/<collection>.json` | Collection, index types and parameters, server version, verified row count |
| Retrieval / reranking / generation traces | Per-path ranks and latency, RRF and reranking scores, selected evidence, answer, and citations |

Artifacts include upstream paths and checksums to detect mismatched data or configurations. **Keep the source PDF, each stage's artifacts, and an accessible Milvus collection** to trace an answer's citations back to the source. Retrieval, reranking, and generation traces record candidates, scores, selected evidence, and final citations.

## Validation and troubleshooting

```bash
python -m pytest tests
```

Tests mainly cover stage logic and mocked dependencies. Passing them does not mean the code has connected to a real Milvus service, run every model, or called a chat endpoint. A full deployment check should build an index from a real PDF, ask at least one question, and inspect `index_manifest_path`, `answer_path`, and the PDF pages referenced by the citations.

- **OCR fails:** Check that the Tesseract executable and English language data are installed. For digital PDFs, try `--ocr never` first.
- **BGE-M3 fails to load:** Check that PyTorch is at least 2.6 and the weights are accessible. The local Intel macOS CPU dependency setup does not meet the current loading requirement.
- **Index construction fails:** Check `MILVUS_URI`, the Milvus 3.x version, and server capacity. For BGE-M3, also confirm the server index version needed by SINDI. A `--dry-run` result does not mean an index was actually built.
- **Retrieval or generation fails:** Use the completed index manifest matching the collection. Check the model name, endpoint, and upstream trace checksums. Generation raises an error if the model returns an invalid answer or omits citations.

See each module's README for stage-specific parameters and artifact details.
