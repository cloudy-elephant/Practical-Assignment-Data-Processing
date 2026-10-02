"""Query-side encoders matching the pinned document embedding models."""

from __future__ import annotations

import math

from embedding.models import (
    BgeM3Encoder,
    MODEL_SPECS,
    QWEN_QUERY_INSTRUCTION,
    QwenEncoder,
    _clean_dense,
    count_model_tokens,
    load_tokenizer,
)


QWEN_QUERY_PREFIX = f"Instruct: {QWEN_QUERY_INSTRUCTION}\nQuery:"


class QueryEncoder:
    def __init__(self, model_id: str, *, device: str = "auto", local_files_only: bool = False):
        spec = next((item for item in MODEL_SPECS.values() if item.model_id == model_id), None)
        if spec is None:
            raise ValueError(f"Unsupported query model: {model_id}")
        self.spec = spec
        self.prompt = QWEN_QUERY_PREFIX if spec.name == "qwen3-0.6b" else ""
        self.tokenizer = load_tokenizer(spec, local_files_only=local_files_only)
        self.encoder = (QwenEncoder(spec, device=device, local_files_only=local_files_only)
                        if spec.name == "qwen3-0.6b" else
                        BgeM3Encoder(spec, device=device, local_files_only=local_files_only))

    def encode(self, question: str) -> dict:
        input_text = self.prompt + question
        token_count = count_model_tokens(self.tokenizer, input_text)
        if token_count > self.spec.max_input_tokens:
            raise ValueError(f"Query has {token_count} model tokens; limit is {self.spec.max_input_tokens}")
        if self.spec.name == "qwen3-0.6b":
            vectors = self.encoder.model.encode(
                [question], prompt=self.prompt, batch_size=1,
                normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False,
            )
            return {"dense_vector": _clean_dense(vectors[0], self.spec.dimension),
                    "sparse_vector": None, "model_token_count": token_count,
                    "query_prompt": self.prompt, "precision": self.encoder.precision}
        outputs = self.encoder.model.encode_queries(
            [question], batch_size=1, max_length=self.spec.max_input_tokens,
            return_dense=True, return_sparse=True, return_colbert_vecs=False,
        )
        sparse = {int(key): float(value) for key, value in outputs["lexical_weights"][0].items()}
        if not sparse or any(key < 0 or not math.isfinite(value) or value <= 0 for key, value in sparse.items()):
            raise ValueError("BGE-M3 returned an invalid query sparse vector")
        return {"dense_vector": _clean_dense(outputs["dense_vecs"][0], self.spec.dimension),
                "sparse_vector": sparse, "model_token_count": token_count,
                "query_prompt": "", "precision": self.encoder.precision}
