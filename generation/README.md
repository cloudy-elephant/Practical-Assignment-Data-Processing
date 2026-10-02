# Generation stage

This stage completes the new financial-report RAG pipeline. It reads the
`selected_evidence` from a `reranking-v1` trace, calls an OpenAI-compatible chat
endpoint, and saves an answer with validated `[1]` style citations. Each citation
resolves to a chunk ID, document ID, page range and source spans. If no evidence
was selected, the answer states that the evidence is insufficient without
calling the model.

Set `RAG_CHAT_MODEL` and either `RAG_CHAT_API_KEY` (or `OPENAI_API_KEY`) for an
OpenAI endpoint, or `RAG_CHAT_BASE_URL` for a course-provided compatible endpoint.
The key is never written to an artifact. From the repository root:

```bash
python -m generation.answer PATH/TO/reranking_trace.json generation_traces
```

For a complete query, import `ask` from `generation.pipeline` and call
`ask(index_manifest_path, question, output_dir)`, or use
[rag_demo.ipynb](rag_demo.ipynb). It runs the
existing retrieval and reranking modules before generation. The returned
`answer_path` points to a `generation-v1` JSON artifact, which records the
answer, citations, selected evidence, model settings and upstream checksums.

The notebook requires an existing Milvus index manifest and access to the
matching embedding and reranking model weights. It does not create an index.
