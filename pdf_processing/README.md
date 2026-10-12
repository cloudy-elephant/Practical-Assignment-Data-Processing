# PDF preprocessing

[`extract.py`](extract.py) and [`ocr.py`](ocr.py) adapt the PyMuPDF and Tesseract
workflow from the earlier DSTA Assignment 1.1 exercise. This package provides
the first stage of [the pipeline design](../assignment-pipeline-design.md);
it does not perform chunking or indexing. Input samples are included in
[`data/`](../data/README.md).

Run from the repository root with the `assignment` Conda environment:

```bash
python -m pdf_processing.extract data/financial/ATSAR2023+bursa.pdf artifacts/parsed_documents.jsonl
```

`--ocr auto` (default) sends pages with fewer than 40 extracted characters to
Tesseract. `--ocr never` is useful for a quick text-only check. Tesseract's
executable and English language data must be installed for OCR.

The output is one JSONL document record containing the PDF checksum, stable
document/page/paragraph/line IDs, page numbers, text, original text, bounding
boxes, extraction method, image boxes, and quality flags. OCR boxes are scaled
back to PDF coordinates. The source PDF is retained in `data/` for citation.
