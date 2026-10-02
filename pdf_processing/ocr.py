"""Tesseract page extraction adapted from the DSTA assignment OCR workflow."""

from collections import defaultdict
from statistics import mean

import pymupdf as fitz
from PIL import Image
import pytesseract


def _union_bboxes(boxes):
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def extract_ocr_paragraphs(page: fitz.Page, *, lang: str = "eng", scale: float = 2.0):
    """Return paragraph/line records with bounding boxes in PDF page coordinates."""
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    data = pytesseract.image_to_data(
        image,
        lang=lang,
        config="--oem 3 --psm 3",
        output_type=pytesseract.Output.DICT,
    )

    grouped = defaultdict(lambda: defaultdict(list))
    for index, raw_text in enumerate(data["text"]):
        text = raw_text.strip()
        if not text:
            continue
        confidence = float(data["conf"][index])
        if confidence < 0:
            continue
        x0 = data["left"][index] / scale
        y0 = data["top"][index] / scale
        x1 = (data["left"][index] + data["width"][index]) / scale
        y1 = (data["top"][index] + data["height"][index]) / scale
        para_key = (data["block_num"][index], data["par_num"][index])
        line_key = data["line_num"][index]
        grouped[para_key][line_key].append((text, [x0, y0, x1, y1], confidence))

    paragraphs = []
    for line_groups in grouped.values():
        lines = []
        for words in line_groups.values():
            text = " ".join(word[0] for word in words)
            lines.append({
                "text": text,
                "original_text": text,
                "bbox": _union_bboxes([word[1] for word in words]),
                "confidence": round(mean(word[2] for word in words) / 100, 4),
            })
        if lines:
            paragraphs.append({
                "text": "\n".join(line["text"] for line in lines),
                "original_text": "\n".join(line["original_text"] for line in lines),
                "bbox": _union_bboxes([line["bbox"] for line in lines]),
                "confidence": round(mean(line["confidence"] for line in lines), 4),
                "lines": lines,
            })
    return paragraphs
