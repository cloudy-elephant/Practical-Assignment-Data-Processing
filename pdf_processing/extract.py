"""Structured PDF extraction adapted from DSTA's PyMuPDF/OCR converters.

The upstream implementation lives in
``dsta-assignment/assignment_1/assignment_1_1``. This adapter preserves its
page/paragraph/line hierarchy and boxes while adding stable IDs, empty-page
handling, and per-page OCR fallback for the new pipeline.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pymupdf as fitz

from .ocr import extract_ocr_paragraphs


SCHEMA_VERSION = "parsed-document-v1"


def _union_bboxes(boxes):
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _extract_text_paragraphs(raw_blocks):
    """Keep the extractor's block and line order, rather than flattening a page."""
    paragraphs = []
    for block in raw_blocks:
        if block["type"] != 0:
            continue
        lines = []
        seen = set()
        for raw_line in block.get("lines", []):
            original_text = "".join(
                char["c"] for span in raw_line["spans"] for char in span.get("chars", [])
            )
            text = original_text.replace("\u00a0", " ").strip()
            bbox = list(raw_line["bbox"])
            marker = (text, tuple(round(value, 2) for value in bbox))
            if not text or marker in seen or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            seen.add(marker)
            lines.append({
                "text": text,
                "original_text": original_text,
                "bbox": bbox,
            })
        if lines:
            paragraphs.append({
                "text": "\n".join(line["text"] for line in lines),
                "original_text": "\n".join(line["original_text"] for line in lines),
                "bbox": _union_bboxes([line["bbox"] for line in lines]),
                "lines": lines,
            })
    return paragraphs


def _assign_ids(paragraphs, page_id):
    for para_number, paragraph in enumerate(paragraphs, start=1):
        paragraph["paragraph_id"] = f"{page_id}:para{para_number}"
        paragraph["reading_order"] = para_number
        for line_number, line in enumerate(paragraph["lines"], start=1):
            line["line_id"] = f"{paragraph['paragraph_id']}:line{line_number}"


def _quality_flags(paragraphs, method, min_ocr_confidence, page_width):
    flags = []
    if not paragraphs:
        flags.append("empty_page")
    if method == "tesseract" and paragraphs:
        scores = [p["confidence"] for p in paragraphs]
        if sum(scores) / len(scores) < min_ocr_confidence:
            flags.append("low_ocr_confidence")
    if any(len(p["text"]) > 3000 for p in paragraphs):
        flags.append("long_paragraph")
    midpoint = page_width / 2
    left = [p["bbox"] for p in paragraphs if p["bbox"][2] < midpoint]
    right = [p["bbox"] for p in paragraphs if p["bbox"][0] > midpoint]
    if len(left) >= 2 and len(right) >= 2:
        left_range = (min(b[1] for b in left), max(b[3] for b in left))
        right_range = (min(b[1] for b in right), max(b[3] for b in right))
        if min(left_range[1], right_range[1]) > max(left_range[0], right_range[0]):
            flags.append("possible_multicolumn_reading_order")
    return flags


def parse_pdf(
    path: str | Path,
    *,
    ocr_mode: str = "auto",
    min_text_chars: int = 40,
    ocr_lang: str = "eng",
    min_ocr_confidence: float = 0.5,
) -> dict:
    """Parse a PDF into a reproducible document → page → paragraph → line tree.

    ``ocr_mode`` is ``auto``, ``always``, or ``never``. Auto OCRs pages with
    fewer than ``min_text_chars`` extracted non-whitespace characters.
    """
    if ocr_mode not in {"auto", "always", "never"}:
        raise ValueError("ocr_mode must be auto, always, or never")
    source = Path(path)
    pdf_bytes = source.read_bytes()
    checksum = hashlib.sha256(pdf_bytes).hexdigest()
    document_id = f"sha256:{checksum}"
    pages = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            raw_blocks = page.get_text("rawdict")["blocks"]
            paragraphs = _extract_text_paragraphs(raw_blocks)
            extracted_chars = sum(len(p["text"].strip()) for p in paragraphs)
            use_ocr = ocr_mode == "always" or (ocr_mode == "auto" and extracted_chars < min_text_chars)
            if use_ocr:
                paragraphs = extract_ocr_paragraphs(page, lang=ocr_lang)
                method = "tesseract"
            else:
                method = "pymupdf" if paragraphs else "none"

            page_id = f"{document_id}:page{page_number}"
            _assign_ids(paragraphs, page_id)
            image_bboxes = [list(block["bbox"]) for block in raw_blocks if block["type"] == 1]
            pages.append({
                "page_id": page_id,
                "page_number": page_number,
                "bbox": list(page.rect),
                "extraction_method": method,
                "paragraphs": paragraphs,
                "image_bboxes": image_bboxes,
                "quality_flags": _quality_flags(paragraphs, method, min_ocr_confidence, page.rect.width),
            })

    return {
        "schema_version": SCHEMA_VERSION,
        "document_id": document_id,
        "source": {
            "filename": source.name,
            "path": source.as_posix(),
            "sha256": checksum,
            "page_count": len(pages),
        },
        "parser_config": {
            "ocr_mode": ocr_mode,
            "min_text_chars": min_text_chars,
            "ocr_lang": ocr_lang,
            "min_ocr_confidence": min_ocr_confidence,
        },
        "pages": pages,
    }


def write_parsed_document(document: dict, output: str | Path) -> None:
    """Write one document record in JSONL format for the next pipeline stage."""
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract structured text from a PDF")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ocr", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("--min-text-chars", type=int, default=40)
    args = parser.parse_args()
    document = parse_pdf(args.pdf, ocr_mode=args.ocr, min_text_chars=args.min_text_chars)
    write_parsed_document(document, args.output)
    print(f"Parsed {document['source']['page_count']} pages into {args.output}")


if __name__ == "__main__":
    main()
