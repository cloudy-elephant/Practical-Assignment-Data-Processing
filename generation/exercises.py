"""Teaching adapter using the repository's retrieval, reranking and generation."""

import json
import os
from pathlib import Path

from embedding.models import MODEL_SPECS
from retrieval.models import QueryEncoder
from retrieval.search import _filter, merge_traces, retrieve
from reranking.build import rerank
from reranking.model import Qwen3Reranker
from .answer import generate, prepare_answer


class RAGPipeline:
    """Separate inspection steps while keeping the production stage implementations."""

    def __init__(self, index_manifests, *, project_root, output_dir,
                 embedding_model="qwen3-0.6b", result_view="fused",
                 document_id=None, page=None, path_limit=20, candidate_limit=20,
                 evidence_limit=5, device="auto", local_files_only=False,
                 max_context_chars=45000, chat_model="", chat_base_url="",
                 chat_api_key="", send_api_requests=False, allow_keyless_endpoint=False):
        _filter(document_id, page)
        if result_view != "fused":
            raise ValueError("Question answering requires result_view='fused'; use retrieval_only for comparisons")
        if not 0 < evidence_limit <= candidate_limit:
            raise ValueError("Require 0 < evidence_limit <= candidate_limit")
        self.index_manifests = dict(index_manifests)
        self.project_root = Path(project_root)
        self.output_dir = Path(output_dir)
        self.embedding_model = embedding_model
        self.result_view = result_view
        self.document_id, self.page = document_id, page
        self.path_limit, self.candidate_limit = path_limit, candidate_limit
        self.evidence_limit = evidence_limit
        self.device, self.local_files_only = device, local_files_only
        self.max_context_chars = max_context_chars
        self.chat_model, self.chat_base_url = chat_model, chat_base_url
        self.chat_api_key = chat_api_key
        self.send_api_requests = send_api_requests
        self.allow_keyless_endpoint = allow_keyless_endpoint
        self.encoders = {}
        self.reranker = None
        self._answer_bundle = None
        self._prepared = None

    def _get_manifest(self, model_name):
        if model_name not in MODEL_SPECS:
            raise ValueError(f"Unknown embedding configuration: {model_name}")
        value = self.index_manifests.get(model_name, "")
        if not str(value).strip():
            print(f"Please configure INDEX_MANIFESTS[{model_name!r}].")
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete" or payload.get("schema_version") != "milvus-index-v1":
            raise ValueError("A complete milvus-index-v1 manifest is required")
        if payload.get("model_id") != MODEL_SPECS[model_name].model_id:
            raise ValueError("Selected embedding and manifest model_id do not match")
        return path.resolve()

    @staticmethod
    def hits_from_trace(trace, view):
        if view == "fused":
            return trace["fused_candidates"]
        if view not in trace["paths"]:
            raise ValueError(f"Unsupported retrieval view: {view}")
        return trace["paths"][view]["hits"]

    def retrieve(self, task_id, queries, *, model_name=None, view=None,
                 evidence_limit=None, retrieval_only=False):
        queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        model_name, view = model_name or self.embedding_model, view or self.result_view
        limit = self.evidence_limit if evidence_limit is None else evidence_limit
        if not 0 < limit <= self.candidate_limit:
            raise ValueError("Require 0 < evidence_limit <= candidate_limit")
        if not retrieval_only and view != "fused":
            raise ValueError("Question answering requires fused retrieval and reranking")
        bundle = {"task_id": task_id, "model_name": model_name, "view": view,
                  "queries": queries, "traces": [], "chunks": [],
                  "retrieval_only": retrieval_only}
        if not retrieval_only:
            self._answer_bundle, self._prepared = bundle, None
        if not queries:
            print("TODO: fill context_questions or enable reference solutions.")
            return bundle
        manifest = self._get_manifest(model_name)
        if manifest is None:
            return bundle
        cache_key = (model_name, self.device, self.local_files_only)
        if cache_key not in self.encoders:
            self.encoders[cache_key] = QueryEncoder(MODEL_SPECS[model_name].model_id,
                                                   device=self.device, local_files_only=self.local_files_only)
        folder = self.output_dir / task_id / model_name
        for question in queries:
            trace = retrieve(manifest, question, folder / "retrieval",
                             document_id=self.document_id, page=self.page,
                             path_limit=self.path_limit, candidate_limit=self.candidate_limit,
                             hnsw_ef=max(64, self.path_limit), device=self.device,
                             local_files_only=self.local_files_only, query_encoder=self.encoders[cache_key])
            bundle["traces"].append(trace)
        if retrieval_only:
            merged = {}
            for number, trace in enumerate(bundle["traces"], 1):
                for rank, hit in enumerate(self.hits_from_trace(trace, view)[:limit], 1):
                    item = merged.setdefault(hit["chunk_id"], {**hit, "query_matches": []})
                    item["query_matches"].append({"query_number": number, "query": trace["question"], "rank": rank})
            bundle["chunks"] = list(merged.values())
            return bundle
        combined = (bundle["traces"][0] if len(bundle["traces"]) == 1 else
                    merge_traces([t["trace_path"] for t in bundle["traces"]], folder / "retrieval",
                                 candidate_limit=self.candidate_limit))
        if combined["fused_candidates"] and self.reranker is None:
            self.reranker = Qwen3Reranker(device=self.device, local_files_only=self.local_files_only)
        ranked = rerank(combined["trace_path"], folder / "reranking",
                        max_candidates=self.candidate_limit, evidence_limit=limit,
                        device=self.device, local_files_only=self.local_files_only, scorer=self.reranker)
        bundle.update({"combined_trace": combined, "reranking": ranked,
                       "chunks": ranked["selected_evidence"]})
        return bundle

    @staticmethod
    def show_chunks(bundle):
        for number, item in enumerate(bundle["chunks"], 1):
            print(f"\n[{number}] chunk_id={item['chunk_id']}; document={item['document_id']}; PDF pages={item['page_range']}")
            print("Ranks:", item.get("query_matches", item.get("paths")))
            print(item["text"])
        print("\nEvidence chunks:", len(bundle["chunks"]))
        for trace in bundle["traces"]:
            print("Retrieval trace:", trace["trace_path"])
        if "reranking" in bundle:
            print("Reranking trace:", bundle["reranking"]["reranking_trace_path"])

    @staticmethod
    def select_chunks(bundle, chunk_ids=None):
        if chunk_ids is None:
            return bundle["chunks"]
        mapping = {item["chunk_id"]: item for item in bundle["chunks"]}
        if len(set(chunk_ids)) != len(chunk_ids) or any(cid not in mapping for cid in chunk_ids):
            raise ValueError("SELECTED_CHUNK_IDS must be unique IDs from the displayed chunks")
        return [mapping[cid] for cid in chunk_ids]

    def build_prompt(self, answer_prompt, chunks):
        self._prepared = None
        if not answer_prompt.strip():
            print("TODO: fill answer_prompt or enable reference solutions.")
            return []
        if not self._answer_bundle or "reranking" not in self._answer_bundle:
            print("Run the question-answering retrieval and reranking first.")
            return []
        prepared = prepare_answer(self._answer_bundle["reranking"]["reranking_trace_path"],
                                  question=answer_prompt, chunk_ids=[c["chunk_id"] for c in chunks],
                                  max_context_chars=self.max_context_chars)
        if chunks != prepared["selected_evidence"]:
            raise ValueError("Evidence must match the saved reranking trace")
        self._prepared = prepared
        return [dict(message) for message in prepared["messages"]]

    def _check_prompt(self, task_id, bundle, chunks, messages):
        if (bundle is not self._answer_bundle or bundle["task_id"] != task_id
                or bundle["retrieval_only"] or self._prepared is None
                or messages != self._prepared["messages"] or chunks != self._prepared["selected_evidence"]):
            raise ValueError("Rebuild the prompt from this question's reranked evidence")

    def preview_prompt(self, task_id, bundle, answer_prompt, chunks, messages):
        if not messages:
            return
        self._check_prompt(task_id, bundle, chunks, messages)
        if answer_prompt.strip() != self._prepared["question"]:
            raise ValueError("Answer question differs from the prepared prompt")
        folder = self.output_dir / task_id / bundle["model_name"] / "fused"
        folder.mkdir(parents=True, exist_ok=True)
        artifact = {"task_id": task_id, "embedding_model": bundle["model_name"],
                    "result_view": bundle["view"], "retrieval_queries": bundle["queries"],
                    "answer_prompt": self._prepared["question"], "messages": messages,
                    "selected_chunks": chunks, "reranking_trace_path": bundle["reranking"]["reranking_trace_path"],
                    "retrieval_trace_paths": [t["trace_path"] for t in bundle["traces"]]}
        (folder / "prompt_and_evidence.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
        (folder / "prompt.txt").write_text("\n\n".join(m["role"].upper() + "\n" + m["content"] for m in messages), encoding="utf-8")
        print(messages[1]["content"])
        print("\nSaved prompt and evidence to:", folder)

    def qa_search(self, task_id, messages, chunks, bundle):
        if not messages:
            print("No valid prompt has been built; skipping API.")
            return None
        self._check_prompt(task_id, bundle, chunks, messages)
        if not self.send_api_requests:
            print("Prompt is ready. Configure the API and set rag.send_api_requests=True.")
            return None
        if not self.chat_model.strip():
            raise ValueError("Please set CHAT_MODEL")
        api_key = self.chat_api_key or os.environ.get("RAG_CHAT_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key and not (self.chat_base_url and self.allow_keyless_endpoint):
            raise ValueError("Set an API key or enable ALLOW_KEYLESS_ENDPOINT for a custom endpoint")
        result = generate(bundle["reranking"]["reranking_trace_path"],
                          self.output_dir / task_id / bundle["model_name"] / "generation",
                          question=self._prepared["question"], chunk_ids=[c["chunk_id"] for c in chunks],
                          max_context_chars=self.max_context_chars, model=self.chat_model,
                          base_url=self.chat_base_url or None, api_key=api_key,
                          expected_messages=messages)
        print(result["answer"])
        print("Citation mapping:", result["citations"])
        print("Answer trace:", result["answer_path"])
        return result
