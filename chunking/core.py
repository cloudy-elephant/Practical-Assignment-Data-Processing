"""Traceable fixed-token, paragraph, and layout-aware chunking."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import re

import tiktoken
from tiktoken.load import load_tiktoken_bpe


CONFIG_VERSION = "chunking-v1"
COMPARISON_METHODS = ("fixed_token", "paragraph_recursive", "layout_paragraph")
ALL_METHODS = (*COMPARISON_METHODS, "page_control")


@dataclass(frozen=True)
class Unit:
    text: str
    paragraph_id: str
    page_number: int
    bbox: list[float]
    page_width: float
    reading_order: int
    char_start: int
    char_end: int


@lru_cache(maxsize=None)
def _encoding(name: str):
    if name == "cl100k_base":
        # Match tiktoken_ext.openai_public.cl100k_base, using the checked-in
        # vocabulary so the assignment also works without network access.
        vocabulary = Path(__file__).parent / "resources" / "cl100k_base.tiktoken"
        ranks = load_tiktoken_bpe(
            str(vocabulary),
            expected_hash="223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
        )
        return tiktoken.Encoding(
            name="cl100k_base",
            pat_str=r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s""",
            mergeable_ranks=ranks,
            special_tokens={
                "<|endoftext|>": 100257,
                "<|fim_prefix|>": 100258,
                "<|fim_middle|>": 100259,
                "<|fim_suffix|>": 100260,
                "<|endofprompt|>": 100276,
            },
        )
    return tiktoken.get_encoding(name)


def _count(text: str, encoding) -> int:
    return len(encoding.encode(text, disallowed_special=()))


def _token_windows(text: str, encoding, size: int, overlap: int = 0):
    """Return UTF-8-safe character windows with token-sized boundaries."""
    tokens = encoding.encode(text, disallowed_special=())
    if not tokens:
        return
    char_byte_offsets = [0]
    for character in text:
        char_byte_offsets.append(char_byte_offsets[-1] + len(character.encode("utf-8")))
    byte_to_char = {offset: index for index, offset in enumerate(char_byte_offsets)}
    token_byte_offsets = [0]
    for token in tokens:
        token_byte_offsets.append(token_byte_offsets[-1] + len(encoding.decode_single_token_bytes(token)))
    if token_byte_offsets[-1] != char_byte_offsets[-1]:
        raise ValueError("Tokenizer did not preserve the input text")
    safe = [index for index, offset in enumerate(token_byte_offsets) if offset in byte_to_char]
    start = 0
    total = len(tokens)
    while start < total:
        desired_end = min(start + size, total)
        end = safe[bisect_right(safe, desired_end) - 1]
        if end <= start:
            end = safe[bisect_right(safe, start)]
        char_start = byte_to_char[token_byte_offsets[start]]
        char_end = byte_to_char[token_byte_offsets[end]]
        while end > start and _count(text[char_start:char_end], encoding) > size:
            end = safe[bisect_left(safe, end) - 1]
            char_end = byte_to_char[token_byte_offsets[end]]
        if end <= start:
            raise ValueError("A single UTF-8 character exceeds the token window")
        yield char_start, char_end, text[char_start:char_end]
        if end == total:
            break
        desired_start = max(start + 1, end - overlap)
        next_start = safe[bisect_right(safe, desired_start) - 1]
        start = next_start if start < next_start < end else end


def _paragraph_units(document: dict) -> list[Unit]:
    units = []
    for page in document["pages"]:
        page_bbox = page.get("bbox")
        page_width = (
            page_bbox[2] - page_bbox[0]
            if page_bbox else max((p["bbox"][2] for p in page["paragraphs"]), default=0)
        )
        for index, paragraph in enumerate(page["paragraphs"], start=1):
            text = paragraph["text"]
            if not text.strip():
                continue
            units.append(Unit(
                text=text,
                paragraph_id=paragraph["paragraph_id"],
                page_number=page["page_number"],
                bbox=paragraph["bbox"],
                page_width=page_width,
                reading_order=paragraph.get("reading_order", index),
                char_start=0,
                char_end=len(text),
            ))
    return units


def _sentence_segments(text: str):
    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+", text):
        end = match.end()
        if text[start:end].strip():
            yield start, end
        start = end
    if text[start:].strip():
        yield start, len(text)


def _split_oversized(unit: Unit, encoding, max_tokens: int) -> list[Unit]:
    if _count(unit.text, encoding) <= max_tokens:
        return [unit]
    result = []
    for sentence_start, sentence_end in _sentence_segments(unit.text):
        sentence = unit.text[sentence_start:sentence_end]
        if _count(sentence, encoding) <= max_tokens:
            pieces = [(0, len(sentence), sentence)]
        else:
            pieces = list(_token_windows(sentence, encoding, max_tokens))
        for piece_start, piece_end, piece_text in pieces:
            result.append(Unit(
                text=piece_text,
                paragraph_id=unit.paragraph_id,
                page_number=unit.page_number,
                bbox=unit.bbox,
                page_width=unit.page_width,
                reading_order=unit.reading_order,
                char_start=sentence_start + piece_start,
                char_end=sentence_start + piece_end,
            ))
    return result


def _render(units: list[Unit]) -> str:
    pieces = [units[0].text]
    for previous, current in zip(units, units[1:]):
        pieces.append("" if previous.paragraph_id == current.paragraph_id else "\n\n")
        pieces.append(current.text)
    return "".join(pieces)


def _layout_break(previous: Unit, current: Unit, current_tokens: int, target: int) -> bool:
    if previous.page_number != current.page_number:
        return True
    if previous.paragraph_id == current.paragraph_id:
        return False
    if current.reading_order != previous.reading_order + 1:
        return True
    a, b = previous.bbox, current.bbox
    widths = (a[2] - a[0], b[2] - b[0])
    centers = ((a[0] + a[2]) / 2, (b[0] + b[2]) / 2)
    if (
        min(widths) >= current.page_width * 0.25
        and abs(centers[0] - centers[1]) >= current.page_width * 0.35
        and current_tokens >= target // 3
    ):
        return True
    gap = b[1] - a[3]
    return gap > max(24.0, 2 * min(a[3] - a[1], b[3] - b[1])) and current_tokens >= target // 2


def _pack_units(units: list[Unit], encoding, target: int, maximum: int, layout: bool):
    current = []
    for unit in units:
        if not current:
            current = [unit]
            continue
        current_tokens = _count(_render(current), encoding)
        proposed = current + [unit]
        proposed_tokens = _count(_render(proposed), encoding)
        should_break = proposed_tokens > maximum or (
            proposed_tokens > target and current_tokens >= target // 2
        )
        if layout and _layout_break(current[-1], unit, current_tokens, target):
            should_break = True
        if should_break:
            yield current
            current = [unit]
        else:
            current = proposed
    if current:
        yield current


def _spans_from_units(units: list[Unit]) -> list[dict]:
    spans = []
    for unit in units:
        if spans and spans[-1]["paragraph_id"] == unit.paragraph_id and spans[-1]["char_end"] == unit.char_start:
            spans[-1]["char_end"] = unit.char_end
            continue
        spans.append({
            "paragraph_id": unit.paragraph_id,
            "page_number": unit.page_number,
            "bbox": unit.bbox,
            "char_start": unit.char_start,
            "char_end": unit.char_end,
        })
    return spans


def _fixed_windows(units: list[Unit], encoding, target: int, overlap: int):
    if not units:
        return
    parts = []
    positions = []
    cursor = 0
    for index, unit in enumerate(units):
        if index:
            parts.append("\n\n")
            cursor += 2
        start = cursor
        parts.append(unit.text)
        cursor += len(unit.text)
        positions.append((start, cursor, unit))
    full_text = "".join(parts)
    for start, end, text in _token_windows(full_text, encoding, target, overlap):
        spans = []
        for para_start, para_end, unit in positions:
            left, right = max(start, para_start), min(end, para_end)
            if left < right:
                spans.append({
                    "paragraph_id": unit.paragraph_id,
                    "page_number": unit.page_number,
                    "bbox": unit.bbox,
                    "char_start": left - para_start,
                    "char_end": right - para_start,
                })
        if spans:
            yield text, spans


def _make_chunk(
    document: dict, method: str, config: dict, config_hash: str,
    parsed_checksum: str, index: int, text: str, spans: list[dict], encoding,
):
    text_hash = sha256(text.encode("utf-8")).hexdigest()
    identity = f"{document['document_id']}|{config_hash}|{index}|{text_hash}"
    pages = [span["page_number"] for span in spans]
    return {
        "chunk_id": f"chunk:{sha256(identity.encode('utf-8')).hexdigest()}",
        "document_id": document["document_id"],
        "source": document["source"],
        "text": text,
        "embedding_text": text,
        "embedding_context": None,
        "page_range": [min(pages), max(pages)],
        "source_paragraph_ids": list(dict.fromkeys(span["paragraph_id"] for span in spans)),
        "source_spans": spans,
        "token_counts": {config["tokenizer"]: _count(text, encoding)},
        "method": method,
        "config_version": CONFIG_VERSION,
        "config": config,
        "config_hash": config_hash,
        "content_hash": text_hash,
        "source_checksum": document["source"]["sha256"],
        "parsed_document_checksum": parsed_checksum,
    }


def chunk_document(
    document: dict,
    *,
    method: str,
    target_tokens: int = 512,
    max_tokens: int = 768,
    overlap_tokens: int = 64,
    tokenizer: str = "cl100k_base",
) -> list[dict]:
    """Chunk a parsed document with stable IDs and source paragraph offsets."""
    if method not in ALL_METHODS:
        raise ValueError(f"Unknown method: {method}")
    if target_tokens <= 0 or max_tokens < target_tokens:
        raise ValueError("Require 0 < target_tokens <= max_tokens")
    if overlap_tokens < 0 or overlap_tokens >= target_tokens:
        raise ValueError("Require 0 <= overlap_tokens < target_tokens")
    if document.get("schema_version") != "parsed-document-v1":
        raise ValueError("Expected parsed-document-v1 input")
    encoding = _encoding(tokenizer)
    config = {
        "config_version": CONFIG_VERSION,
        "target_tokens": target_tokens,
        "max_tokens": max_tokens,
        "overlap_tokens": overlap_tokens if method == "fixed_token" else 0,
        "tokenizer": tokenizer,
        "boundary_policy": method,
    }
    config_hash = sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
    parsed_checksum = sha256(
        json.dumps(document, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    units = _paragraph_units(document)
    if method == "fixed_token":
        candidates = _fixed_windows(units, encoding, target_tokens, overlap_tokens)
    elif method == "page_control":
        candidates = (
            (_render(page_units), _spans_from_units(page_units))
            for page in document["pages"]
            if (page_units := [unit for unit in units if unit.page_number == page["page_number"]])
        )
    else:
        split_units = [piece for unit in units for piece in _split_oversized(unit, encoding, max_tokens)]
        candidates = (
            (_render(group), _spans_from_units(group))
            for group in _pack_units(split_units, encoding, target_tokens, max_tokens, method == "layout_paragraph")
        )
    chunks = []
    for index, (text, spans) in enumerate(candidates, start=1):
        chunk = _make_chunk(document, method, config, config_hash, parsed_checksum, index, text, spans, encoding)
        if method != "page_control" and chunk["token_counts"][tokenizer] > max_tokens:
            raise ValueError(f"Chunk {index} exceeds max_tokens; inspect tokenizer boundaries")
        chunks.append(chunk)
    return chunks
