from hashlib import sha256
import json

import pytest

from embedding.models import MODEL_SPECS, QWEN_QUERY_INSTRUCTION
from retrieval.models import QWEN_QUERY_PREFIX, QueryEncoder
from retrieval.search import _filter, _parse_hits, fuse, retrieve


def make_manifest(tmp_path, model="qwen3-0.6b"):
    spec = MODEL_SPECS[model]
    indexes = {
        "dense_vector": {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}},
        "bm25_sparse": {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25", "params": {"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": 1.2, "bm25_b": 0.75}},
    }
    if spec.has_sparse:
        indexes["learned_sparse"] = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP", "params": {"inverted_index_algo": "SINDI"}}
    manifest = {"schema_version": "milvus-index-v1", "status": "complete", "verified_rows": 3,
                "row_count": 3, "collection": "assignment_test", "model_id": spec.model_id,
                "model_revision": spec.revision, "embedding_file_sha256": "a" * 64,
                "indexes": indexes}
    path = tmp_path / "index.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def hit(chunk_id, score, page=2):
    text = f"{chunk_id} revenue text"
    return {"id": chunk_id, "distance": score, "entity": {
        "chunk_id": chunk_id, "document_id": "report", "page_start": page,
        "page_end": page, "text": text, "source": {"filename": "report.pdf"},
        "source_spans": [{"paragraph_id": f"{chunk_id}:p1", "page_number": page}],
        "content_hash": sha256(text.encode()).hexdigest(),
    }}


class FakeEncoder:
    def __init__(self, sparse=False):
        self.sparse = sparse

    def encode(self, question):
        return {"dense_vector": [1.0] + [0.0] * 1023,
                "sparse_vector": {12: 0.5} if self.sparse else None,
                "model_token_count": 9, "query_prompt": QWEN_QUERY_PREFIX if not self.sparse else "",
                "precision": "fp32"}


class FakeClient:
    def __init__(self, manifest):
        self.manifest = manifest
        self.calls = []

    def has_collection(self, name):
        return name == self.manifest["collection"]

    def describe_collection(self, name):
        fields = {"chunk_id", "document_id", "page_start", "page_end", "text", "source",
                  "source_spans", "source_checksum", "content_hash", "dense_vector", "bm25_sparse"}
        if "learned_sparse" in self.manifest["indexes"]:
            fields.add("learned_sparse")
        return {"description": f"milvus-index-v1:{self.manifest['embedding_file_sha256']}",
                "fields": [{"name": field} for field in fields], "functions": [{"name": "text_bm25"}]}

    def describe_index(self, name, index_name):
        from indexing.build import INDEX_NAMES

        field = next(field for field, name in INDEX_NAMES.items() if name == index_name)
        settings = self.manifest["indexes"][field]
        return {"field_name": field, "index_type": settings["index_type"],
                "metric_type": settings["metric_type"], "state": "Finished", "indexed_rows": 3,
                **settings["params"]}

    def search(self, **kwargs):
        self.calls.append(kwargs)
        field = kwargs["anns_field"]
        if field == "dense_vector":
            return [[hit("chunk:a", 0.9), hit("chunk:b", 0.8)]]
        if field == "bm25_sparse":
            return [[hit("chunk:b", 4.0), hit("chunk:c", 2.0)]]
        return [[hit("chunk:b", 0.6)]]


def test_qwen_retrieval_applies_same_filter_and_fuses(tmp_path):
    path, manifest = make_manifest(tmp_path)
    client = FakeClient(manifest)
    result = retrieve(path, "What was revenue?", tmp_path / "traces", document_id="report", page=2,
                      client=client, query_encoder=FakeEncoder(), candidate_limit=3)
    assert set(result["paths"]) == {"dense", "bm25"}
    assert {call["filter"] for call in client.calls} == {'document_id == "report" and page_start <= 2 and page_end >= 2'}
    assert all(call["consistency_level"] == "Strong" for call in client.calls)
    assert result["fused_candidates"][0]["chunk_id"] == "chunk:b"
    assert result["fused_candidates"][0]["paths"] == {"dense": {"rank": 2, "score": 0.8}, "bm25": {"rank": 1, "score": 4.0}}
    assert json.loads(open(result["trace_path"], encoding="utf-8").read())["fused_candidates"] == result["fused_candidates"]


def test_bge_runs_three_paths_and_sparse_query(tmp_path):
    path, manifest = make_manifest(tmp_path, model="bge-m3")
    client = FakeClient(manifest)
    result = retrieve(path, "Revenue?", tmp_path / "traces", client=client, query_encoder=FakeEncoder(sparse=True))
    assert set(result["paths"]) == {"dense", "bm25", "learned_sparse"}
    sparse_call = next(call for call in client.calls if call["anns_field"] == "learned_sparse")
    assert sparse_call["data"] == [{12: 0.5}]
    assert result["fused_candidates"][0]["chunk_id"] == "chunk:b"


def test_filter_escapes_document_id_and_rejects_bad_page():
    assert _filter('a" or page_start >= 0', None) == 'document_id == "a\\" or page_start >= 0"'
    with pytest.raises(ValueError, match="page"):
        _filter(None, 0)


def test_fusion_rejects_disagreeing_source_metadata():
    base = {"chunk_id": "chunk:a", "rank": 1, "score": 0.5, "document_id": "report",
            "page_range": [1, 1], "text": "text", "source": {}, "source_spans": [], "content_hash": "hash"}
    with pytest.raises(ValueError, match="disagree"):
        fuse({"dense": [base], "bm25": [{**base, "text": "different"}]})


def test_parse_hits_uses_named_primary_key_when_id_is_absent():
    result = hit("chunk:a", 0.9)
    del result["id"]
    hits = _parse_hits([[result]], path="dense", document_id=None, page=None, limit=20)
    assert hits[0]["chunk_id"] == "chunk:a"

    result["id"] = "chunk:b"
    with pytest.raises(ValueError, match="mismatched chunk_id"):
        _parse_hits([[result]], path="dense", document_id=None, page=None, limit=20)


def test_rejects_wrong_index_algorithm_before_query(tmp_path):
    path, manifest = make_manifest(tmp_path, model="bge-m3")
    manifest["indexes"]["learned_sparse"]["params"]["inverted_index_algo"] = "DAAT_MAXSCORE"
    path.write_text(json.dumps(manifest))
    client = FakeClient(manifest)
    with pytest.raises(ValueError, match="learned sparse configuration"):
        retrieve(path, "Revenue?", tmp_path / "traces", client=client, query_encoder=FakeEncoder(sparse=True))
    assert client.calls == []


def test_qwen_query_encoder_uses_fixed_instruction_and_checks_length():
    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            assert text == QWEN_QUERY_PREFIX + "Revenue?"
            assert add_special_tokens
            return [1, 2, 3]

    class Model:
        def encode(self, texts, **kwargs):
            assert texts == ["Revenue?"] and kwargs["prompt"] == QWEN_QUERY_PREFIX
            return [[1.0] + [0.0] * 1023]

    encoder = object.__new__(QueryEncoder)
    encoder.spec = MODEL_SPECS["qwen3-0.6b"]
    encoder.prompt = QWEN_QUERY_PREFIX
    encoder.tokenizer = Tokenizer()
    encoder.encoder = type("Adapter", (), {"model": Model(), "precision": "fp32"})()
    result = encoder.encode("Revenue?")
    assert result["model_token_count"] == 3
    assert QWEN_QUERY_INSTRUCTION in result["query_prompt"]
