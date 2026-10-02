"""Focused generation tests without external model or Milvus services."""

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from generation.answer import INSUFFICIENT_ANSWER, generate


def _write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return sha256(path.read_bytes()).hexdigest()


def _trace(tmp_path, evidence):
    retrieval_path = tmp_path / "retrieval.json"
    retrieval_sha = _write(retrieval_path, {"question": "When did the financial year end?",
                                            "collection": "reports", "index_manifest_sha256": "index-sha"})
    reranking_path = tmp_path / "reranking.json"
    _write(reranking_path, {"schema_version": "reranking-v1", "status": "complete",
                            "stage": "reranking", "question": "When did the financial year end?",
                            "collection": "reports", "retrieval_trace_path": str(retrieval_path),
                            "retrieval_trace_sha256": retrieval_sha,
                            "index_manifest_path": str(tmp_path / "index.json"),
                            "index_manifest_sha256": "index-sha", "evidence_limit": 5,
                            "selected_evidence": evidence})
    return reranking_path


def _evidence():
    text = "The financial year ended 31 March 2023."
    return [{"chunk_id": "chunk-1", "document_id": "report", "page_range": [57, 57],
             "source": "report.pdf", "source_spans": [{"page": 57}], "text": text,
             "content_hash": sha256(text.encode()).hexdigest()}]


class FakeClient:
    def __init__(self, content):
        self.content = content
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
                               usage=SimpleNamespace(prompt_tokens=30, completion_tokens=12, total_tokens=42))


def test_answer_is_saved_with_resolved_citation(tmp_path):
    client = FakeClient("The year ended 31 March 2023 [1].")
    result = generate(_trace(tmp_path, _evidence()), tmp_path / "answers",
                      model="course-model", client=client)
    assert result["citations"][0]["chunk_id"] == "chunk-1"
    assert result["citations"][0]["page_range"] == [57, 57]
    assert json.loads(Path(result["answer_path"]).read_text())["answer"] == result["answer"]
    assert "[1] document_id=report" in client.calls[0]["messages"][1]["content"]


def test_no_evidence_skips_endpoint(tmp_path):
    client = FakeClient("should not be called")
    result = generate(_trace(tmp_path, []), tmp_path / "answers", model="course-model", client=client)
    assert result["answer"] == INSUFFICIENT_ANSWER
    assert result["citations"] == []
    assert client.calls == []


def test_rejects_unsupported_citation(tmp_path):
    client = FakeClient("The year ended in 2023 [2].")
    with pytest.raises(ValueError, match="valid evidence numbers"):
        generate(_trace(tmp_path, _evidence()), tmp_path / "answers", model="course-model", client=client)
