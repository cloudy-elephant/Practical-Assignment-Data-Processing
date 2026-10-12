# Reusable assignment data

This directory contains input samples and a legacy extraction result reused
unchanged from the earlier DSTA Assignment 1 exercises.

- `financial/ATSAR2023+bursa.pdf`: primary English financial report for the RAG pipeline.
- `pdf_samples/`: electronic, scanned, layout, and empty-page PDF examples from Assignment 1.1.
- `image_samples/`: JPEG input and EXIF-orientation examples from Assignment 1.1.
- `legacy_reference/ATSAR2023+bursa.json`: output of the old extraction pipeline, kept for comparison. It is not the new parsed-document schema.

Use the files under `data/` when running this repository's pipeline. All paths
above are relative to `data/`; commands in the module READMEs run from the
repository root.

The pipeline uses these local copies and does not require the original
assignment checkout.
