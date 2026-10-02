"""Connect the existing retrieval, reranking and generation stages."""

from __future__ import annotations

from pathlib import Path

from .answer import generate
from reranking.build import rerank
from retrieval.search import retrieve


def ask(index_manifest_path: str | Path, question: str, output_dir: str | Path, *,
        model: str | None = None, base_url: str | None = None,
        document_id: str | None = None, page: int | None = None,
        path_limit: int = 20, candidate_limit: int = 20,
        evidence_limit: int = 5, device: str = "auto",
        local_files_only: bool = False) -> dict:
    """Run one question through retrieval, fixed reranking and cited answering."""
    output = Path(output_dir)
    retrieval = retrieve(index_manifest_path, question, output / "retrieval",
                         document_id=document_id, page=page, path_limit=path_limit,
                         candidate_limit=candidate_limit, device=device,
                         local_files_only=local_files_only)
    reranking = rerank(retrieval["trace_path"], output / "reranking",
                       max_candidates=candidate_limit, evidence_limit=evidence_limit,
                       device=device, local_files_only=local_files_only)
    return generate(reranking["reranking_trace_path"], output / "generation",
                    model=model, base_url=base_url)
