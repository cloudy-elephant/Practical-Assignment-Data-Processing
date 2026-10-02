"""Pinned Qwen3-Reranker-0.6B scoring with explicit prompt and length audit."""

from __future__ import annotations

import math


MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
MODEL_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
INSTRUCTION = (
    "Given a question about a financial report, determine whether the passage contains evidence that answers it."
)
PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. '
    'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Qwen3Reranker:
    def __init__(self, *, device: str = "auto", local_files_only: bool = False, max_input_tokens: int = 2048):
        if max_input_tokens <= 0 or max_input_tokens > 8192:
            raise ValueError("max_input_tokens must be between 1 and 8192")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        from transformers import AutoTokenizer
        import torch

        available = torch.cuda.is_available()
        if device == "cuda" and not available:
            raise ValueError("CUDA requested but unavailable")
        self.device = "cuda" if (device == "cuda" or (device == "auto" and available)) else "cpu"
        self.dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if self.device == "cuda" else torch.float32
        self.local_files_only = local_files_only
        self.max_input_tokens = max_input_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, local_files_only=local_files_only,
            trust_remote_code=False, padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.prefix_ids = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        if self.no_id == self.yes_id or any(value is None or value < 0 for value in (self.no_id, self.yes_id)):
            raise ValueError("Reranker yes/no token IDs are invalid")
        self.model = None

    def _tokens(self, question: str, passage: str) -> list[int]:
        body = f"<Instruct>: {INSTRUCTION}\n<Query>: {question}\n<Document>: {passage}"
        return self.prefix_ids + self.tokenizer.encode(body, add_special_tokens=False) + self.suffix_ids

    def score(self, question: str, passages: list[str], *, batch_size: int = 4) -> list[dict]:
        """Return raw yes-minus-no logit scores; never truncate input pairs."""
        if not question.strip() or batch_size <= 0:
            raise ValueError("Require a nonempty question and positive batch_size")
        if any(not isinstance(passage, str) or not passage.strip() for passage in passages):
            raise ValueError("Passages must be nonempty strings")
        sequences = [self._tokens(question, passage) for passage in passages]
        over_limit = [(index, len(ids)) for index, ids in enumerate(sequences) if len(ids) > self.max_input_tokens]
        if over_limit:
            index, count = over_limit[0]
            raise ValueError(f"Reranker candidate {index} has {count} tokens; limit is {self.max_input_tokens}")
        if not sequences:
            return []
        import torch
        from transformers import AutoModelForCausalLM

        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, revision=MODEL_REVISION, local_files_only=self.local_files_only,
                trust_remote_code=False, dtype=self.dtype,
            ).to(self.device).eval()
        scores = []
        for start in range(0, len(sequences), batch_size):
            batch_ids = sequences[start:start + batch_size]
            batch = self.tokenizer.pad({"input_ids": batch_ids}, padding=True, return_tensors="pt")
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with torch.inference_mode():
                logits = self.model(**batch).logits[:, -1, :]
                differences = (logits[:, self.yes_id].float() - logits[:, self.no_id].float()).tolist()
            for offset, raw in enumerate(differences):
                raw = float(raw)
                if not math.isfinite(raw):
                    raise ValueError("Reranker returned a nonfinite score")
                scores.append({"score": raw, "input_tokens": len(batch_ids[offset])})
        return scores
