# Financial Document RAG Pipeline

[English version](README.md)

本仓库实现一条面向英文财报 PDF 的可追溯 RAG（检索增强生成）流程：从原始 PDF 提取文本、切块并建立 Milvus 索引，再对问题检索和重排证据，生成带来源引用的回答。

流程由根目录的 `pdf_processing/`、`chunking/`、`embedding/`、`indexing/`、`retrieval/`、`reranking/` 和 `generation/` 模块组成。索引与检索需要可访问的 Milvus 服务；生成答案还需要聊天模型接口。仓库不附带预建索引。

## 快速开始：配置环境与 Milvus

以下命令从**仓库根目录**运行。先按官方说明安装 [Conda](https://docs.conda.io/projects/conda/en/latest/)、[Ubuntu Docker Engine 与 Compose](https://docs.docker.com/engine/install/ubuntu/) 或 [macOS Docker Desktop](https://docs.docker.com/desktop/setup/install/mac-install/)，并启动 Docker。再根据操作系统选择一组 Tesseract 英文语言数据安装命令：

Ubuntu / Debian：
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
```

macOS + Homebrew：
```bash
brew install tesseract
```

创建 Python 3.11 环境：

```bash
conda create -n financial-rag python=3.11 -y
conda activate financial-rag
```

在安装项目依赖前，按设备选择一条 PyTorch 命令。

**Linux x86_64 + CUDA 12.4**：

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

**Linux CPU**：

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

其他 CUDA 版本或 Apple Silicon macOS 请使用 [PyTorch 官方安装命令](https://docs.pytorch.org/get-started/previous-versions/)。Intel macOS 跳过 PyTorch 安装命令，`requirement.txt` 会安装固定的 CPU 版本；该环境可运行下面的 Qwen3 样例，但不适合 BGE-M3。

然后在已激活的环境中安装项目依赖并检查 OCR：

```bash
python -m pip install -r requirement.txt
python -c 'import torch; print(torch.__version__, "CUDA available:", torch.cuda.is_available())'
tesseract --list-langs
```

启动仓库提供的 Milvus 3.x 服务，并等待容器健康检查通过。下面的 Qwen3 路线使用 HNSW/COSINE 和 BM25，无需 BGE-M3 的 SINDI 服务端设置。

```bash
docker compose -f milvus3-local/docker-compose.yaml up -d --wait
docker compose -f milvus3-local/docker-compose.yaml ps
curl -fsS http://127.0.0.1:9091/healthz
export MILVUS_URI=http://127.0.0.1:19530
```

首次运行需要下载固定版本的 tokenizer 和模型权重。若使用远程 Milvus，请将 `MILVUS_URI` 改为服务地址，并按需设置 `MILVUS_TOKEN`、`MILVUS_DB_NAME`。

## 一条完整样例：PDF 到检索结果

这条路线固定使用 `data/financial/ATSAR2023+bursa.pdf`、`layout_paragraph` 切块、`qwen3-0.6b` 嵌入、Milvus HNSW/COSINE＋BM25 检索，以及 `Qwen/Qwen3-Reranker-0.6B` 重排。环境配置完成后，在**同一个终端**依次运行下列命令。脚本从各阶段的 JSON 输出自动读取下一阶段所需路径，最后打印融合检索候选和前 5 条重排证据。

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

`fused_candidates` 是 RRF 融合后的检索结果，`selected_evidence` 是重排后可用于生成回答的证据。完整记录分别保存在 `RETRIEVAL_TRACE_PATH` 和 `RERANKING_TRACE_PATH`。本样例停在证据选择阶段，因此无需配置聊天接口；如需生成答案，继续阅读下方“对一条问题执行问答”。索引阶段会核验 Milvus 中的物理索引；[Milvus 对小数据集可能不建立物理索引](https://milvus.io/docs/performance_faq.md)。若样本文档的数据量不足，应增加输入文档后重新建库，不能把 `--dry-run` 当作建库成功。

## 完整流程

```text
建库：PDF → PyMuPDF / 按需 OCR → parsed_documents.jsonl
          → 三种切块方案 → chunks.jsonl
          → BGE-M3 或 Qwen3-Embedding → embeddings.jsonl + manifest.json
          → Milvus 3.x collection + index manifest

问答：问题 → 同模型查询编码 + Milvus 多路检索 → RRF 排名融合
          → Qwen3-Reranker → 前 5 条证据 → OpenAI 兼容聊天接口
          → 答案、[1] 式引用和 generation trace
```

每阶段读取上一阶段的产物。PDF 哈希、`chunk_id`、页码、段落字符范围和校验值沿流程传递，方便从答案回查原 PDF。当前代码验证引用编号及来源元数据是否有效；它不会自动验证回答中每项事实是否真的被引用证据支持。

## 项目结构

| 路径 | 职责 | 主要产物 |
| --- | --- | --- |
| [`data/`](data/README.md) | 财报 PDF 和输入样本 | 输入数据 |
| [`pdf_processing/`](pdf_processing/README.md) | 逐页提取文本，必要时 OCR，保留段落、行和坐标 | `parsed_documents.jsonl` |
| [`chunking/`](chunking/README.md) | 按统一 schema 生成不同切块方案 | 各方案的 `chunks.jsonl` |
| [`embedding/`](embedding/README.md) | 检查模型输入长度并生成向量，支持中断续跑 | `embeddings.jsonl`、`manifest.json` |
| [`indexing/`](indexing/README.md) | 在 Milvus 建库并核验索引和数据 | collection、index manifest |
| [`retrieval/`](retrieval/README.md) | 多路搜索并用 RRF 融合 | retrieval trace |
| [`reranking/`](reranking/README.md) | 固定重排模型打分并选证据 | reranking trace |
| [`generation/`](generation/README.md) | 调用聊天接口并保存引用和答案 | generation trace |
| `tests/` | 各阶段的单元测试和模拟依赖测试 | 测试结果 |

### 模型与检索路径

| 嵌入模型 | 文档表示 | Milvus 检索路径 |
| --- | --- | --- |
| `bge-m3`（`BAAI/bge-m3`） | 1024 维 dense + learned sparse | HNSW/COSINE、learned sparse/SINDI/IP、全文 BM25 |
| `qwen3-0.6b`（`Qwen/Qwen3-Embedding-0.6B`） | 1024 维 dense | HNSW/COSINE、全文 BM25 |

模型 revision 固定在 [`embedding/models.py`](embedding/models.py)。两条路线可处理同一份切块，但会形成不同的嵌入产物和 collection。BM25 由 Milvus 根据 chunk 原文构建。查询时，dense 和 learned sparse 使用对应模型编码问题，BM25 使用问题原文；Qwen3 查询侧带固定检索指令。各路径分数尺度不同，因此用 RRF 按排名融合，默认每条命中贡献 `1 / (60 + rank)`，同时保留各路径原始分数和排名。

## 环境补充说明

- 依赖文件是 [`requirement.txt`](requirement.txt)（单数）。完整模型推理建议使用 Linux GPU 环境；离线运行时需预先缓存固定 revision 的权重和 tokenizer，并加上 `--local-files-only`。
- OCR 需要系统安装 Tesseract 和英文语言数据。`--ocr auto` 默认只处理提取字符少于 40 的页面；电子 PDF 可先用 `--ocr never` 检查。
- 索引与检索需要可访问的 **Milvus 3.x** 服务。本地可用 `milvus3-local/docker-compose.yaml` 启动；默认连接 `http://127.0.0.1:19530`，远程服务可用 `MILVUS_URI` 覆盖。BGE-M3 的 SINDI 路线还要求服务端 `dataCoord.targetVecIndexVersion >= 10`，需要服务管理员确认。
- 生成需要 OpenAI 兼容的 chat endpoint 和模型名。密钥由环境变量提供，不会写进产物。

在 Intel macOS 上，依赖文件为本地 CPU 检查固定了 PyTorch 2.2.2；配合当前 `transformers==4.57.6`，它不足以加载 BGE-M3 的 `.bin` 权重。可在本地做解析、切块、tokenizer 验证和部分 Qwen3 检查；完整 BGE-M3 路线应在兼容环境运行。

## 从 PDF 建库

以下以 [`data/financial/ATSAR2023+bursa.pdf`](data/financial/ATSAR2023+bursa.pdf) 为例。`artifacts/`、`embeddings/`、`index_manifests/` 是示例输出目录。后续步骤中的 `PATH/TO/...` 要替换为程序打印的真实路径。

**1. 解析 PDF。** PyMuPDF 提取文字，低文本量页自动切换到 Tesseract。输出一条 JSONL 文档记录，含 PDF SHA-256 文档 ID、页/段/行 ID、原文、标准化文本、bbox、提取方式和质量标记。

```bash
python -m pdf_processing.extract data/financial/ATSAR2023+bursa.pdf artifacts/parsed_documents.jsonl
```

**2. 切块。** 默认同时写出 `fixed_token`、`paragraph_recursive` 和 `layout_paragraph`。默认 `cl100k_base`，目标 512、上限 768 token；64 token 重叠仅用于固定窗口法。段落法拆分超长段落；版面法还参考页码、阅读顺序和 bbox，但不推断表格单元格或标题层级。

```bash
python -m chunking.build artifacts/parsed_documents.jsonl artifacts/chunks
```

可加 `--methods fixed_token paragraph_recursive layout_paragraph page_control` 生成整页切块；整页法可能超过 token 上限。切块记录中的 `source_spans` 指向原段落及字符范围，`embedding_text` 当前等于展示用 `text`。

**3. 嵌入。** 下面以 `layout_paragraph` 为例。两种模型都可以处理同一份 `chunks.jsonl`，但会分别生成嵌入产物。先做模型 tokenizer 校验，再编码；模型输入过长会报错，不会静默截断。命令结果会打印 `embeddings_path`，实际文件位于按切块方法、模型 revision 和配置哈希分层的目录。

```bash
python -m embedding.build artifacts/chunks/layout_paragraph/chunks.jsonl embeddings --model qwen3-0.6b --validate-only
python -m embedding.build artifacts/chunks/layout_paragraph/chunks.jsonl embeddings --model qwen3-0.6b --batch-size 4

# 在兼容的 GPU 环境运行 BGE-M3
python -m embedding.build artifacts/chunks/layout_paragraph/chunks.jsonl embeddings --model bge-m3 --validate-only
python -m embedding.build artifacts/chunks/layout_paragraph/chunks.jsonl embeddings --model bge-m3 --batch-size 4
```

`--limit 1` 可运行真实模型的单 chunk smoke test；它不是整份报告的完整索引。中断后重跑同一配置会复用已校验记录。只有 `manifest.json` 的 `status` 为 `complete` 且文件校验值匹配，才能进入索引阶段。

**4. 建 Milvus 索引。** 先用 `--dry-run` 离线检查嵌入产物和建库计划。接入 Milvus 后，程序为不同嵌入产物及索引设置创建隔离的 collection，按 `chunk_id` upsert，并核验物理索引描述、已索引行数及读回数据。命令输出的 `index_manifest_path` 是问答入口。

```bash
python -m indexing.build PATH/TO/embeddings.jsonl index_manifests --dry-run

# export MILVUS_URI=http://YOUR_MILVUS_HOST:19530  # 连接远程服务时设置
# export MILVUS_TOKEN=YOUR_TOKEN              # 服务需要 token 时设置
# export MILVUS_DB_NAME=YOUR_DATABASE       # 默认 default
python -m indexing.build PATH/TO/embeddings.jsonl index_manifests
```

使用 **BGE-M3** 时，管理员确认服务端索引版本至少为 10 后，建库命令还需加 `--target-vec-index-version 10`。该参数只记录已确认的设置，**不会修改 Milvus 服务配置**。数据量过小而无法形成可核验的物理索引时，程序不会生成 `complete` 的索引 manifest。

## 对一条问题执行问答

最直接的入口是 [`generation.pipeline.ask`](generation/pipeline.py)：传入已完成的 index manifest、问题和输出目录，依次运行检索、重排和生成。也可打开 [`generation/rag_demo.ipynb`](generation/rag_demo.ipynb) 逐阶段检查候选、证据和答案；notebook 不会建索引。

```bash
# export MILVUS_URI=http://YOUR_MILVUS_HOST:19530  # 连接远程服务时设置
# export MILVUS_TOKEN=YOUR_TOKEN                   # 服务需要 token 时设置
export RAG_CHAT_MODEL=YOUR_CHAT_MODEL
export RAG_CHAT_BASE_URL=http://YOUR_CHAT_ENDPOINT/v1  # 使用 OpenAI 官方接口时可省略
export RAG_CHAT_API_KEY=YOUR_API_KEY                    # 也可使用 OPENAI_API_KEY
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

若需分别检查或调整参数，可逐步运行：

```bash
python -m retrieval.search PATH/TO/index_manifests/COLLECTION.json \
  "What was the financial year end reported by the company?" retrieval_traces \
  --path-limit 20 --candidate-limit 20

python -m reranking.build PATH/TO/retrieval_trace.json reranking_traces \
  --max-candidates 20 --evidence-limit 5

python -m generation.answer PATH/TO/reranking_trace.json generation_traces
```

后两个 trace 路径由前一步命令打印。检索默认每路取 20 条，RRF 融合后最多保留 20 个候选；`--document-id` 或 `--page` 可以限缩范围，页过滤也包括跨越该页的 chunk。固定的 `Qwen/Qwen3-Reranker-0.6B` 最多重排 20 个候选，默认取前 5 个证据。生成器只收到问题和这些证据，要求事实陈述引用 `[1]` 等编号；没有证据或模型明确返回 `INSUFFICIENT_EVIDENCE` 时，结果会说明证据不足。

## 产物与溯源

| 产物 | 重点检查 |
| --- | --- |
| `parsed_documents.jsonl` | PDF 哈希、页码、提取方式、质量标记、段落/行坐标 |
| `<method>/chunks.jsonl` | `chunk_id`、文本、token 数、页范围、`source_spans`、配置 |
| `embeddings.jsonl` + `manifest.json` | 模型/revision、向量维度、输入和输出校验值、完成状态 |
| `index_manifests/<collection>.json` | collection、索引类型和参数、服务版本、核验行数 |
| retrieval / reranking / generation trace | 各路径排名和耗时、RRF 与重排分数、所选证据、答案和引用 |

产物含上游路径和校验值，可检查配置与数据是否错配。**保留原 PDF、各阶段文件以及可访问的 Milvus collection**，才能从答案引用追溯到原文。检索、重排和生成的 trace 记录候选结果、分数、所选证据及最终引用。

## 验证与常见问题

```bash
python -m pytest tests
```

测试主要覆盖阶段逻辑和模拟依赖；通过测试不代表已连接真实 Milvus、运行全部模型或调用聊天服务。完整部署应至少用真实 PDF 建库并问一条问题，核对 `index_manifest_path`、`answer_path` 和引用所指的 PDF 页码。

- **OCR 报错：** 检查 Tesseract 可执行文件及英文语言数据；电子 PDF 可先用 `--ocr never`。
- **BGE-M3 无法加载：** 检查 PyTorch 是否 ≥ 2.6，以及权重是否可访问；Intel macOS 本地 CPU 依赖设置不满足当前加载要求。
- **索引构建失败：** 检查 `MILVUS_URI`、Milvus 3.x 版本和服务资源；BGE-M3 还需确认 SINDI 所需的服务端索引版本。`--dry-run` 不代表实际建库成功。
- **检索或生成失败：** 使用与 collection 对应的完整 index manifest；确认模型名、endpoint 和上游 trace 校验值。模型若输出无效或缺少引用，生成阶段会报错。

各阶段的参数和产物细节见对应模块的 README。
