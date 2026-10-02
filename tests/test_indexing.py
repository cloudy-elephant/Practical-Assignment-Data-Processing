from hashlib import sha256
import json

import pytest

from embedding.models import MODEL_SPECS
from indexing.build import _index_params, _schema, build_index, make_plan


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def artifact(tmp_path, model_name="qwen3-0.6b"):
    spec = MODEL_SPECS[model_name]
    folder = tmp_path / model_name
    folder.mkdir()
    path = folder / "embeddings.jsonl"
    text = "Revenue increased."
    row = {
        "schema_version": "embedding-v2", "chunk_id": "chunk:one", "document_id": "report",
        "source": {"sha256": "a" * 64}, "source_checksum": "a" * 64,
        "text": text, "content_hash": sha256(text.encode()).hexdigest(),
        "embedding_text_sha256": sha256(text.encode()).hexdigest(),
        "parsed_document_checksum": "b" * 64,
        "page_range": [1, 1], "source_spans": [{"paragraph_id": "p1", "page_number": 1}],
        "method": "layout_paragraph", "chunking_config_hash": "chunk-config",
        "model_id": spec.model_id, "model_revision": spec.revision, "dimension": 1024,
        "model_token_count": 5, "precision": "fp32", "config_hash": "embedding-config",
        "dense_vector": [1.0] + [0.0] * 1023,
        "sparse_vector": {"12": 0.5} if spec.has_sparse else None,
    }
    row["record_checksum"] = sha256(canonical(row)).hexdigest()
    path.write_bytes(canonical(row) + b"\n")
    manifest = {
        "status": "complete", "config_hash": "embedding-config", "chunk_count": 1,
        "embeddings_file_sha256": sha256(path.read_bytes()).hexdigest(),
        "config": {"model_id": spec.model_id, "model_revision": spec.revision,
                   "dimension": 1024, "chunking_method": "layout_paragraph",
                   "chunking_config_hash": "chunk-config"},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return path


class FakeSchema:
    def __init__(self, description):
        self.description = description
        self.fields = []
        self.functions = []

    def add_field(self, field_name, datatype, **kwargs):
        self.fields.append({"name": field_name})

    def add_function(self, function):
        self.functions.append({"name": function.name})


class FakeIndexParams:
    def __init__(self):
        self.indexes = []

    def add_index(self, **kwargs):
        self.indexes.append(kwargs)


class FakeClient:
    def __init__(self):
        self.collections = {}
        self.upserts = []

    def get_server_version(self):
        return "3.0.0"

    def has_collection(self, name):
        return name in self.collections

    def create_schema(self, **kwargs):
        return FakeSchema(kwargs["description"])

    def prepare_index_params(self):
        return FakeIndexParams()

    def create_collection(self, collection_name, schema, index_params):
        self.collections[collection_name] = {
            "description": schema.description, "fields": schema.fields,
            "functions": schema.functions,
            "indexes": {index["index_name"]: {
                "field_name": index["field_name"], "index_type": index["index_type"],
                "metric_type": index["metric_type"], "state": "Finished", "indexed_rows": 1,
                **index["params"],
            } for index in index_params.indexes},
            "rows": {},
        }

    def describe_collection(self, name):
        return self.collections[name]

    def list_indexes(self, name):
        return list(self.collections[name]["indexes"])

    def describe_index(self, name, index_name):
        return self.collections[name]["indexes"][index_name]

    def upsert(self, collection_name, data):
        self.upserts.extend(data)
        self.collections[collection_name]["rows"].update({row["chunk_id"]: row for row in data})
        return {"upsert_count": len(data)}

    def flush(self, name):
        pass

    def load_collection(self, name):
        pass

    def get(self, name, ids, output_fields):
        return [{field: self.collections[name]["rows"][key][field] for field in output_fields}
                for key in ids if key in self.collections[name]["rows"]]


def test_qwen_build_and_repeat(tmp_path):
    path = artifact(tmp_path)
    client = FakeClient()
    first = build_index(path, tmp_path / "manifests", client=client)
    assert first["status"] == "complete" and first["verified_rows"] == 1
    assert set(client.collections[first["collection"]]["indexes"]) == {"dense_hnsw", "bm25_inverted"}
    assert "learned_sparse" not in client.upserts[0]
    second = build_index(path, tmp_path / "manifests", client=client)
    assert second["collection"] == first["collection"]
    assert len(client.collections) == 1


def test_bge_requires_server_version_and_uses_sparse_token_ids(tmp_path):
    path = artifact(tmp_path, "bge-m3")
    plan, _ = make_plan(path)
    assert plan["indexes"]["learned_sparse"]["params"]["inverted_index_algo"] == "SINDI"
    client = FakeClient()
    with pytest.raises(ValueError, match="targetVecIndexVersion"):
        build_index(path, tmp_path / "manifests", client=client)
    assert not client.collections
    result = build_index(path, tmp_path / "manifests", client=client, target_vec_index_version=10)
    assert client.collections[result["collection"]]["indexes"]["learned_sparse_sindi"]["metric_type"] == "IP"
    assert client.upserts[0]["learned_sparse"] == {12: 0.5}


def test_tampered_embedding_rejected_before_write(tmp_path):
    path = artifact(tmp_path)
    with path.open("ab") as stream:
        stream.write(b"corrupt")
    client = FakeClient()
    with pytest.raises(ValueError, match="checksum"):
        build_index(path, tmp_path / "manifests", client=client)
    assert not client.collections


def test_schema_and_index_params_construct_with_installed_pymilvus(tmp_path):
    from pymilvus import MilvusClient

    plan, _ = make_plan(artifact(tmp_path, "bge-m3"))
    schema = _schema(MilvusClient, plan)
    indexes = _index_params(MilvusClient, plan)
    assert {field.name for field in schema.fields} >= {"chunk_id", "dense_vector", "learned_sparse", "bm25_sparse"}
    assert len(schema.functions) == 1
    assert len(indexes) == 3
