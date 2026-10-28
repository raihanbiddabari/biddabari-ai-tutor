"""
Step 2: split cleaned page text into retrieval-sized chunks.

Strategy (matches the PRD):
  - Split on markdown headings ('## Heading') first, so each chunk stays on
    one topic.
  - If a section is still bigger than CHUNK_MAX_WORDS, slide a window over
    it with CHUNK_OVERLAP_WORDS overlap so a sentence never gets cut without
    any surrounding context.
  - If a section is smaller than CHUNK_MIN_WORDS, merge it with the next
    section so we don't end up with tiny, context-free chunks.
  - Every chunk carries metadata (url, page_title, section_heading,
    last_updated) that the API uses for citations and the nightly refresh
    uses to know what changed.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Optional

from . import config
from .crawler import CleanPage


@dataclass
class Chunk:
    id: str
    url: str
    page_title: str
    section_heading: str
    text: str
    last_updated: Optional[str] = None
    word_count: int = field(default=0)

    def __post_init__(self):
        self.word_count = len(self.text.split())


HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)


def _split_into_sections(text: str) -> List[tuple[str, str]]:
    """Returns [(heading, section_text), ...]. Text before the first heading
    gets heading '' (page intro)."""
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [("", text.strip())]

    sections = []
    # text before first heading
    if matches[0].start() > 0:
        intro = text[: matches[0].start()].strip()
        if intro:
            sections.append(("", intro))

    for i, m in enumerate(matches):
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append((heading, body))

    return sections


def _merge_small_sections(sections: List[tuple[str, str]], min_words: int) -> List[tuple[str, str]]:
    merged: List[tuple[str, str]] = []
    buffer_heading, buffer_text = None, ""

    for heading, body in sections:
        combined = f"{buffer_text}\n\n{body}".strip() if buffer_text else body
        if len(combined.split()) < min_words:
            buffer_heading = buffer_heading or heading
            buffer_text = combined
            continue
        if buffer_text:
            merged.append((buffer_heading or heading, combined))
            buffer_heading, buffer_text = None, ""
        else:
            merged.append((heading, body))

    if buffer_text:
        merged.append((buffer_heading or "", buffer_text))

    return merged


def _sliding_window(words: List[str], max_words: int, overlap: int) -> List[List[str]]:
    if len(words) <= max_words:
        return [words]
    windows = []
    step = max(max_words - overlap, 1)
    for start in range(0, len(words), step):
        window = words[start : start + max_words]
        if window:
            windows.append(window)
        if start + max_words >= len(words):
            break
    return windows


def chunk_page(page: CleanPage) -> List[Chunk]:
    sections = _split_into_sections(page.text)
    sections = _merge_small_sections(sections, config.CHUNK_MIN_WORDS)

    chunks: List[Chunk] = []
    for heading, body in sections:
        words = body.split()
        windows = _sliding_window(words, config.CHUNK_MAX_WORDS, config.CHUNK_OVERLAP_WORDS)
        for idx, window_words in enumerate(windows):
            window_text = " ".join(window_words)
            raw_id = f"{page.url}::{heading}::{idx}"
            chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:16]
            chunks.append(
                Chunk(
                    id=chunk_id,
                    url=page.url,
                    page_title=page.title,
                    section_heading=heading or page.title,
                    text=window_text,
                    last_updated=page.lastmod,
                )
            )
    return chunks


def chunk_pages(pages: List[CleanPage]) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for page in pages:
        all_chunks.extend(chunk_page(page))
    return all_chunks


if __name__ == "__main__":
    from .crawler import crawl_and_clean

    pages = crawl_and_clean()
    chunks = chunk_pages(pages)
    print(f"{len(pages)} pages -> {len(chunks)} chunks")
    for c in chunks[:5]:
        print(f"  [{c.id}] {c.page_title} / {c.section_heading} ({c.word_count}w)")
        print(f"      {c.text[:120]}...")
