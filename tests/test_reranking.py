from hashlib import sha256
import json

import pytest

from embedding.models import MODEL_SPECS
from reranking.build import rerank
from reranking.model import Qwen3Reranker


def trace_file(tmp_path, *, candidates=3):
    spec = MODEL_SPECS["qwen3-0.6b"]
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"status": "complete", "collection": "assignment_test",
                                 "model_id": spec.model_id, "model_revision": spec.revision}))
    items = []
    for rank in range(1, candidates + 1):
        text = f"Passage {rank} about revenue"
        items.append({"fused_rank": rank, "chunk_id": f"chunk:{rank}", "document_id": "report",
                      "page_range": [rank, rank], "text": text, "source": {"filename": "report.pdf"},
                      "source_spans": [{"paragraph_id": f"p{rank}", "page_number": rank}],
                      "content_hash": sha256(text.encode()).hexdigest(),
                      "paths": {"dense": {"rank": rank, "score": 0.9 / rank}},
                      "rrf_score": 1 / (60 + rank)})
    trace = {"schema_version": "retrieval-v1", "status": "complete", "stage": "retrieval",
             "question": "What was revenue?", "collection": "assignment_test",
             "document_filter": "report", "page_filter": None,
             "index_manifest_path": str(index), "index_manifest_sha256": sha256(index.read_bytes()).hexdigest(),
             "model_id": spec.model_id, "model_revision": spec.revision,
             "candidate_limit": 20, "fused_candidates": items}
    path = tmp_path / "retrieval.json"
    path.write_text(json.dumps(trace))
    return path


class FakeScorer:
    device = "cpu"
    dtype = "float32"

    def __init__(self):
        self.calls = []

    def score(self, question, passages, *, batch_size):
        self.calls.append((question, passages, batch_size))
        return [{"score": score, "input_tokens": 30} for score in [0.2, 2.0, -1.0][:len(passages)]]


def test_reranks_and_selects_evidence_with_source_metadata(tmp_path):
    path = trace_file(tmp_path)
    scorer = FakeScorer()
    result = rerank(path, tmp_path / "out", scorer=scorer, evidence_limit=2)
    assert scorer.calls == [("What was revenue?", [f"Passage {i} about revenue" for i in range(1, 4)], 4)]
    assert [item["chunk_id"] for item in result["reranked_candidates"]] == ["chunk:2", "chunk:1", "chunk:3"]
    assert [item["chunk_id"] for item in result["selected_evidence"]] == ["chunk:2", "chunk:1"]
    assert result["selected_evidence"][0]["fused_rank"] == 2
    assert result["selected_evidence"][0]["source_spans"][0]["paragraph_id"] == "p2"
    stored = json.loads(open(result["reranking_trace_path"], encoding="utf-8").read())
    assert stored["retrieval_trace_sha256"] == sha256(path.read_bytes()).hexdigest()


def test_rejects_changed_index_manifest(tmp_path):
    path = trace_file(tmp_path)
    trace = json.loads(path.read_text())
    index = tmp_path / "index.json"
    index.write_text(index.read_text() + " ")
    with pytest.raises(ValueError, match="checksum"):
        rerank(path, tmp_path / "out", scorer=FakeScorer())
    assert not (tmp_path / "out").exists()
    assert trace["index_manifest_sha256"] != sha256(index.read_bytes()).hexdigest()


def test_rejects_invalid_content_hash_before_scoring(tmp_path):
    path = trace_file(tmp_path)
    trace = json.loads(path.read_text())
    trace["fused_candidates"][0]["text"] = "changed"
    path.write_text(json.dumps(trace))
    scorer = FakeScorer()
    with pytest.raises(ValueError, match="text/content_hash"):
        rerank(path, tmp_path / "out", scorer=scorer)
    assert scorer.calls == []


def test_empty_candidate_trace_does_not_load_model(tmp_path):
    path = trace_file(tmp_path, candidates=0)
    result = rerank(path, tmp_path / "out")
    assert result["candidates_scored"] == 0 and result["selected_evidence"] == []


def test_pair_length_fails_before_model_load():
    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            assert not add_special_tokens
            return [1, 2]

    model = object.__new__(Qwen3Reranker)
    model.max_input_tokens = 5
    model.prefix_ids = [3, 4]
    model.suffix_ids = [5, 6]
    model.tokenizer = Tokenizer()
    model.model = None
    with pytest.raises(ValueError, match="candidate 0 has 6 tokens"):
        model.score("Revenue?", ["Report revenue"], batch_size=1)
    assert model.model is None
