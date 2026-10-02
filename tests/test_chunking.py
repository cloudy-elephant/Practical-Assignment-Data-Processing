"""Chunk boundary and provenance checks for the parsed-document schema."""

import pytest

from chunking.core import ALL_METHODS, chunk_document


def _document():
    first = "Revenue was €10.5 million in 2023. " * 14
    second = "Operating profit increased. Cash was stable."
    third = "The following year opened with a new plan."
    paragraphs = (first, second, third)
    return {
        "schema_version": "parsed-document-v1",
        "document_id": "sha256:test-document",
        "source": {"sha256": "test-document"},
        "pages": [
            {
                "page_number": 1,
                "paragraphs": [
                    {"paragraph_id": "p1", "reading_order": 1, "bbox": [10, 10, 250, 100], "text": first},
                    {"paragraph_id": "p2", "reading_order": 2, "bbox": [10, 110, 250, 130], "text": second},
                ],
            },
            {
                "page_number": 2,
                "paragraphs": [
                    {"paragraph_id": "p3", "reading_order": 1, "bbox": [10, 10, 250, 30], "text": third},
                ],
            },
        ],
    }, dict(zip(("p1", "p2", "p3"), paragraphs))


@pytest.mark.parametrize("method", ALL_METHODS)
def test_chunk_provenance_and_stable_ids(method):
    document, originals = _document()
    kwargs = dict(method=method, target_tokens=20, max_tokens=30, overlap_tokens=4)
    chunks = chunk_document(document, **kwargs)
    assert chunks
    assert [chunk["chunk_id"] for chunk in chunks] == [
        chunk["chunk_id"] for chunk in chunk_document(document, **kwargs)
    ]
    for chunk in chunks:
        assert chunk["source"] == document["source"]
        assert len(chunk["parsed_document_checksum"]) == 64
        spans = chunk["source_spans"]
        assert spans
        assert chunk["page_range"] == [
            min(span["page_number"] for span in spans),
            max(span["page_number"] for span in spans),
        ]
        for span in spans:
            source_text = originals[span["paragraph_id"]][span["char_start"]:span["char_end"]]
            assert source_text in chunk["text"]
        if method != "page_control":
            assert chunk["token_counts"]["cl100k_base"] <= 30
    if method == "layout_paragraph":
        assert all(chunk["page_range"][0] == chunk["page_range"][1] for chunk in chunks)
    if method == "page_control":
        assert len(chunks) == 2
        assert any(chunk["token_counts"]["cl100k_base"] > 30 for chunk in chunks)


def test_fixed_windows_cover_all_source_text():
    document, originals = _document()
    chunks = chunk_document(document, method="fixed_token", target_tokens=20, max_tokens=30, overlap_tokens=4)
    for paragraph_id, text in originals.items():
        ranges = sorted(
            (span["char_start"], span["char_end"])
            for chunk in chunks for span in chunk["source_spans"]
            if span["paragraph_id"] == paragraph_id
        )
        cursor = 0
        for start, end in ranges:
            assert start <= cursor
            cursor = max(cursor, end)
        assert cursor == len(text)


def test_layout_keeps_narrow_table_cells_together():
    document, _ = _document()
    document["pages"] = [{
        "page_number": 1,
        "bbox": [0, 0, 600, 800],
        "paragraphs": [
            {
                "paragraph_id": f"cell-{index}",
                "reading_order": index + 1,
                "bbox": [40 if index % 2 == 0 else 220, 20 + index * 12,
                         130 if index % 2 == 0 else 310, 30 + index * 12],
                "text": f"Revenue {2020 + index} 100",
            }
            for index in range(20)
        ],
    }]
    chunks = chunk_document(document, method="layout_paragraph", target_tokens=80, max_tokens=100)
    assert len(chunks) <= 3
