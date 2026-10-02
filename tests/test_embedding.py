import hashlib
import json

import pytest

from embedding.build import build_embeddings
from embedding.models import MODEL_SPECS


class Tokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens
        return list(text.split()) + ["<special>"]


class Encoder:
    precision = "fp32"

    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(texts)
        return [{"dense_vector": [1.0] + [0.0] * 1023, "sparse_vector": None} for _ in texts]


def chunks_file(path, texts):
    with path.open("w", encoding="utf-8") as stream:
        for number, text in enumerate(texts):
            row = {
                "chunk_id": f"chunk:{number}", "document_id": "doc", "source": {"sha256": "source"},
                "text": text, "embedding_text": text, "content_hash": hashlib.sha256(text.encode()).hexdigest(),
                "source_checksum": "source", "parsed_document_checksum": "parsed", "source_spans": [{"paragraph_id": "p1"}],
                "page_range": [1, 1], "method": "layout_paragraph", "config_hash": "chunk-config",
            }
            stream.write(json.dumps(row) + "\n")


def test_resume_reuses_unchanged_and_reencodes_changed(tmp_path):
    source = tmp_path / "chunks.jsonl"
    output = tmp_path / "out"
    chunks_file(source, ["first", "second"])
    first_encoder = Encoder()
    first = build_embeddings(source, output, model_name="qwen3-0.6b", device="cpu", tokenizer=Tokenizer(), encoder=first_encoder, batch_size=1)
    assert first["encoded"] == 2 and first["reused"] == 0
    assert first_encoder.calls == [["first"], ["second"]]

    no_work = build_embeddings(source, output, model_name="qwen3-0.6b", device="cpu", tokenizer=Tokenizer(), encoder=Encoder())
    assert no_work["encoded"] == 0 and no_work["reused"] == 2
    chunks_file(source, ["first", "changed"])
    changed_encoder = Encoder()
    changed = build_embeddings(source, output, model_name="qwen3-0.6b", device="cpu", tokenizer=Tokenizer(), encoder=changed_encoder)
    assert changed["encoded"] == 1 and changed["reused"] == 1
    assert changed_encoder.calls == [["changed"]]
    rows = [json.loads(line) for line in open(changed["embeddings_path"], encoding="utf-8")]
    assert [row["chunk_id"] for row in rows] == ["chunk:0", "chunk:1"]
    assert len(rows) == 2 and rows[1]["text"] == "changed"


def test_validate_only_never_loads_encoder(tmp_path):
    source = tmp_path / "chunks.jsonl"
    chunks_file(source, ["one two"])
    result = build_embeddings(source, tmp_path / "out", model_name="bge-m3", tokenizer=Tokenizer(), validate_only=True)
    assert result == {"status": "validated", "model": "BAAI/bge-m3", "chunks": 1, "max_model_tokens": 3}
    assert not (tmp_path / "out").exists()


def test_rejects_duplicate_chunk_ids(tmp_path):
    source = tmp_path / "chunks.jsonl"
    chunks_file(source, ["one", "two"])
    lines = source.read_text().splitlines()
    second = json.loads(lines[1])
    second["chunk_id"] = "chunk:0"
    source.write_text(lines[0] + "\n" + json.dumps(second) + "\n")
    with pytest.raises(ValueError, match="duplicate chunk_id"):
        build_embeddings(source, tmp_path / "out", model_name="bge-m3", tokenizer=Tokenizer(), validate_only=True)


def test_rejects_model_limit_before_encoder(tmp_path):
    source = tmp_path / "chunks.jsonl"
    chunks_file(source, ["x " * MODEL_SPECS["bge-m3"].max_input_tokens])
    with pytest.raises(ValueError, match="exceed"):
        build_embeddings(source, tmp_path / "out", model_name="bge-m3", device="cpu", tokenizer=Tokenizer(), encoder=Encoder())
    assert not (tmp_path / "out").exists()


def test_recovers_partial_last_record(tmp_path):
    source = tmp_path / "chunks.jsonl"
    chunks_file(source, ["one"])
    result = build_embeddings(source, tmp_path / "out", model_name="qwen3-0.6b", device="cpu", tokenizer=Tokenizer(), encoder=Encoder())
    with open(result["embeddings_path"], "ab") as stream:
        stream.write(b'{"chunk_id":')
    resumed = build_embeddings(source, tmp_path / "out", model_name="qwen3-0.6b", device="cpu", tokenizer=Tokenizer(), encoder=Encoder())
    assert resumed["reused"] == 1 and resumed["encoded"] == 0
    assert len(open(resumed["embeddings_path"], encoding="utf-8").readlines()) == 1
