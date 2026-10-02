"""Pinned adapters for the two core embedding tracks."""

from dataclasses import dataclass
import math


QWEN_QUERY_INSTRUCTION = (
    "Given a question about a financial report, retrieve passages that answer the question."
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_id: str
    revision: str
    dimension: int
    max_input_tokens: int
    has_sparse: bool


MODEL_SPECS = {
    "bge-m3": ModelSpec(
        name="bge-m3",
        model_id="BAAI/bge-m3",
        revision="5617a9f61b028005a4858fdac845db406aefb181",
        dimension=1024,
        max_input_tokens=8192,
        has_sparse=True,
    ),
    "qwen3-0.6b": ModelSpec(
        name="qwen3-0.6b",
        model_id="Qwen/Qwen3-Embedding-0.6B",
        revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        dimension=1024,
        max_input_tokens=32768,
        has_sparse=False,
    ),
}


def load_tokenizer(spec: ModelSpec, *, local_files_only: bool = False):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )


def count_model_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True))


def _clean_dense(vector, expected_dimension: int) -> list[float]:
    values = [float(value) for value in vector]
    if len(values) != expected_dimension or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Expected a finite {expected_dimension}-dimensional dense vector")
    norm = math.sqrt(sum(value * value for value in values))
    if not 0.99 <= norm <= 1.01:
        raise ValueError(f"Dense vector is not L2-normalized (norm={norm:.5f})")
    return values


# TODO: Test BGE-M3 with PyTorch >= 2.6 in a separate Intel macOS conda environment;
# reconcile the existing pip-installed torch and NumPy < 2 requirement before updating assignment.
class BgeM3Encoder:
    def __init__(self, spec: ModelSpec, *, device: str, local_files_only: bool):
        import torch
        from FlagEmbedding import BGEM3FlagModel
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(
            repo_id=spec.model_id,
            revision=spec.revision,
            local_files_only=local_files_only,
            ignore_patterns=["onnx/*", "imgs/*", "README.md"],
        )
        # The pinned BGE-M3 revision ships PyTorch .bin/.pt weights. Recent
        # Transformers refuses those checkpoints with torch < 2.6.
        from packaging.version import Version

        if Version(torch.__version__.split("+")[0]) < Version("2.6"):
            raise RuntimeError(
                "BGE-M3 requires PyTorch >= 2.6 with this Transformers version "
                "because its pinned checkpoint has no safetensors weights. "
                "Run BGE-M3 in the course GPU environment with a compatible PyTorch build."
            )
        selected_device = None if device == "auto" else device
        use_fp16 = (torch.cuda.is_available() if device == "auto" else device.startswith("cuda"))
        self.model = BGEM3FlagModel(
            model_path,
            use_fp16=use_fp16,
            devices=selected_device,
            passage_max_length=spec.max_input_tokens,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        self.spec = spec
        self.precision = "fp16" if use_fp16 else "fp32"

    def encode(self, texts: list[str]) -> list[dict]:
        outputs = self.model.encode(
            texts,
            batch_size=len(texts),
            max_length=self.spec.max_input_tokens,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = outputs["dense_vecs"]
        sparse = outputs["lexical_weights"]
        if len(dense) != len(texts) or len(sparse) != len(texts):
            raise ValueError("BGE-M3 returned the wrong batch size")
        rows = []
        for vector, weights in zip(dense, sparse):
            cleaned = {}
            for token_id, weight in weights.items():
                token_number = int(token_id)
                value = float(weight)
                if token_number < 0 or not math.isfinite(value) or value <= 0:
                    raise ValueError("BGE-M3 returned invalid sparse token weights")
                cleaned[str(token_number)] = value
            rows.append({
                "dense_vector": _clean_dense(vector, self.spec.dimension),
                "sparse_vector": cleaned,
            })
        return rows


class QwenEncoder:
    def __init__(self, spec: ModelSpec, *, device: str, local_files_only: bool):
        import torch
        from sentence_transformers import SentenceTransformer

        selected_device = None if device == "auto" else device
        use_cuda = torch.cuda.is_available() if device == "auto" else device.startswith("cuda")
        self.model = SentenceTransformer(
            spec.model_id,
            revision=spec.revision,
            local_files_only=local_files_only,
            device=selected_device,
            trust_remote_code=False,
            model_kwargs={"dtype": "auto" if use_cuda else torch.float32},
        )
        self.model.max_seq_length = spec.max_input_tokens
        self.spec = spec
        self.precision = "auto" if use_cuda else "fp32"

    def encode(self, texts: list[str]) -> list[dict]:
        vectors = self.model.encode(
            texts,
            prompt="",
            batch_size=len(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if len(vectors) != len(texts):
            raise ValueError("Qwen3 returned the wrong batch size")
        return [
            {"dense_vector": _clean_dense(vector, self.spec.dimension), "sparse_vector": None}
            for vector in vectors
        ]


def load_encoder(spec: ModelSpec, *, device: str = "auto", local_files_only: bool = False):
    if spec.name == "bge-m3":
        return BgeM3Encoder(spec, device=device, local_files_only=local_files_only)
    if spec.name == "qwen3-0.6b":
        return QwenEncoder(spec, device=device, local_files_only=local_files_only)
    raise ValueError(f"Unsupported embedding model: {spec.name}")
