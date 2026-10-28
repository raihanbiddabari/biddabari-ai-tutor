"""
Step 1 of the pipeline: pull pages from the website and clean them down to
the real article text (nav/footer/ads stripped out).

Two modes, same output shape, controlled by config.CRAWL_MODE:
  - "local": reads sitemap.xml + .html files from a local folder (demo / dev)
  - "http":  fetches a real sitemap and real pages over HTTPS (production)

Output: a list of RawPage(url, title, html, lastmod)
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import trafilatura

from . import config


@dataclass
class RawPage:
    url: str
    title: str
    html: str
    lastmod: Optional[str] = None


@dataclass
class CleanPage:
    url: str
    title: str
    text: str          # cleaned main-content text, headings kept as markdown-ish "## "
    lastmod: Optional[str] = None


SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


def _first(el, tag_ns, tag_plain):
    found = el.find(tag_ns, SITEMAP_NS)
    if found is None:
        found = el.find(tag_plain)
    return found


def _parse_sitemap_xml(xml_text: str) -> List[dict]:
    """Returns [{'loc': ..., 'lastmod': ...}, ...]"""
    root = ET.fromstring(xml_text)
    url_els = root.findall("sm:url", SITEMAP_NS)
    if not url_els:
        url_els = root.findall("url")

    entries = []
    for url_el in url_els:
        loc_el = _first(url_el, "sm:loc", "loc")
        lastmod_el = _first(url_el, "sm:lastmod", "lastmod")
        if loc_el is None or not loc_el.text:
            continue
        entries.append({
            "loc": loc_el.text.strip(),
            "lastmod": lastmod_el.text.strip() if lastmod_el is not None and lastmod_el.text else None,
        })
    return entries


def fetch_sitemap_local(local_dir: str) -> List[dict]:
    sitemap_path = Path(local_dir) / "sitemap.xml"
    xml_text = sitemap_path.read_text(encoding="utf-8")
    return _parse_sitemap_xml(xml_text)


def fetch_sitemap_http(sitemap_url: str) -> List[dict]:
    import requests
    resp = requests.get(sitemap_url, timeout=20)
    resp.raise_for_status()
    entries = _parse_sitemap_xml(resp.text)
    # Handle sitemap index files (a sitemap that just lists other sitemaps)
    if not entries:
        return entries
    return entries


def fetch_page_local(local_dir: str, loc: str) -> str:
    # loc looks like "file://course-bcs-preli.html" in our sample sitemap
    filename = loc.replace("file://", "")
    path = Path(local_dir) / filename
    return path.read_text(encoding="utf-8")


def fetch_page_http(url: str) -> str:
    import requests
    resp = requests.get(url, timeout=20, headers={"User-Agent": "Biddabari-AI-Tutor-Indexer/1.0"})
    resp.raise_for_status()
    return resp.text


def crawl_all() -> List[RawPage]:
    """Reads the sitemap and fetches every page listed in it."""
    if config.CRAWL_MODE == "local":
        entries = fetch_sitemap_local(config.SITE_LOCAL_DIR)
    else:
        entries = fetch_sitemap_http(config.SITE_SITEMAP_URL)

    pages: List[RawPage] = []
    for entry in entries:
        loc = entry["loc"]
        try:
            if config.CRAWL_MODE == "local":
                html = fetch_page_local(config.SITE_LOCAL_DIR, loc)
            else:
                html = fetch_page_http(loc)
        except Exception as e:  # noqa: BLE001 - we want to keep going on single-page failure
            print(f"  [crawler] WARN: could not fetch {loc}: {e}")
            continue

        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else loc

        pages.append(RawPage(url=loc, title=title, html=html, lastmod=entry.get("lastmod")))

    return pages


def clean_page(page: RawPage) -> Optional[CleanPage]:
    """
    Strip nav/footer/ads/sidebar boilerplate and keep the real article text.
    Uses trafilatura, which is built for exactly this (main-content extraction).
    We ask it to keep headings as markdown-style '#' markers so the chunker
    can split on them later.
    """
    extracted = trafilatura.extract(
        page.html,
        include_comments=False,
        include_tables=True,
        output_format="markdown",
        favor_recall=True,
    )
    if not extracted or len(extracted.strip()) < 20:
        return None

    return CleanPage(url=page.url, title=page.title, text=extracted.strip(), lastmod=page.lastmod)


def crawl_and_clean() -> List[CleanPage]:
    raw_pages = crawl_all()
    cleaned = []
    for p in raw_pages:
        c = clean_page(p)
        if c:
            cleaned.append(c)
        else:
            print(f"  [crawler] WARN: no extractable content on {p.url}")
    return cleaned


if __name__ == "__main__":
    pages = crawl_and_clean()
    print(f"Crawled & cleaned {len(pages)} pages:")
    for p in pages:
        print(f"  - {p.url} ({len(p.text.split())} words)")
