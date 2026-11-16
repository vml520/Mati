# Part of Mati's field — the phase-field substrate its memory is built on.
#
# ⚠️ THIS IS THE AUTHORITATIVE COPY. Nothing outside Mati imports these
# modules, so the field can change without anything else moving under it.
#
# ⚠️ KEEP THIS A COMMENT, NOT A DOCSTRING. A docstring here would sit
# before `from __future__`, which is a SyntaxError that `ast.parse`
# does not report.
"""
Phase 5 data utilities: download, cache, and preprocess real text corpora.

Supports:
    - Project Gutenberg plain-text downloads (strips header/footer)
    - Local .txt file loading
    - Automatic caching to .cache/ directory
"""

import os
import urllib.request
import urllib.error

# Cache directory next to the project root
# Vendoring moved this file a directory deeper, so computing the cache path
# from __file__ silently pointed at `Mati/.cache` — which does not exist. The
# path now comes from ONE place; see Mati/config.py.
from ..config import CACHE_DIR                                     # noqa: E402


def _ensure_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)


def download(url: str, filename: str = None, timeout: int = 60) -> str:
    """
    Download a text URL; return file path.  Uses local cache if already present.

    Args:
        url      : HTTP/HTTPS URL to fetch
        filename : cache filename (derived from URL if not provided)
        timeout  : socket timeout in seconds

    Returns:
        Path to the cached local file.
    """
    _ensure_cache()
    if filename is None:
        filename = url.split('/')[-1].split('?')[0] or 'corpus.txt'
    path = os.path.join(CACHE_DIR, filename)

    if os.path.exists(path):
        print(f"  Cache hit  → {path}  ({os.path.getsize(path)//1024} KB)")
        return path

    print(f"  Downloading {url} ...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mati/1.0'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        text = raw.decode('utf-8', errors='replace')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"  Saved      → {path}  ({len(text)//1024} KB)")
        return path
    except (urllib.error.URLError, OSError) as e:
        print(f"  Download failed: {e}")
        return None


def strip_gutenberg(text: str) -> str:
    """
    Remove Project Gutenberg header and footer boilerplate.
    Returns only the literary content between the *** START *** and *** END *** markers.
    """
    lines = text.split('\n')
    start, end = 0, len(lines)

    for i, line in enumerate(lines):
        if '*** START OF' in line.upper() or '***START OF' in line.upper():
            start = i + 1
            break

    for i, line in enumerate(reversed(lines)):
        if '*** END OF' in line.upper() or '***END OF' in line.upper():
            end = len(lines) - i - 1
            break

    content = '\n'.join(lines[start:end]).strip()
    return content


def load_text(path: str, max_chars: int = None) -> str:
    """Load text from a local file, optionally truncated."""
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    if max_chars:
        text = text[:max_chars]
    return text


def load_corpus(url: str = None, local_path: str = None,
                filename: str = None, strip_pg: bool = True,
                max_chars: int = None) -> str:
    """
    Load a text corpus from a URL or local file.

    Priority: local_path > cached URL > download.

    Args:
        url        : URL to download (Project Gutenberg, etc.)
        local_path : path to a local .txt file (overrides url)
        filename   : cache filename for url download
        strip_pg   : strip Project Gutenberg header/footer
        max_chars  : truncate text to this many characters

    Returns:
        Cleaned text string, or None if all sources failed.
    """
    text = None

    # 1. Local file
    if local_path and os.path.exists(local_path):
        print(f"  Loading local corpus: {local_path}")
        text = load_text(local_path)

    # 2. URL (with cache)
    elif url:
        cached = download(url, filename)
        if cached:
            text = load_text(cached)

    if text is None:
        return None

    # Strip Gutenberg boilerplate
    if strip_pg and ('gutenberg' in (url or '').lower()
                     or 'PROJECT GUTENBERG' in text[:3000].upper()):
        text = strip_gutenberg(text)
        print(f"  Stripped PG boilerplate → {len(text)//1024} KB content")

    if max_chars:
        text = text[:max_chars]

    print(f"  Corpus ready: {len(text):,} chars")
    return text


# ── Bundled sources ────────────────────────────────────────────────────────────

SOURCES = {
    'pride_and_prejudice': {
        'url':      'https://www.gutenberg.org/files/1342/1342-0.txt',
        'filename': 'pride_and_prejudice.txt',
        'author':   'Jane Austen',
    },
    'moby_dick': {
        'url':      'https://www.gutenberg.org/files/2701/2701-0.txt',
        'filename': 'moby_dick.txt',
        'author':   'Herman Melville',
    },
    'tale_of_two_cities': {
        'url':      'https://www.gutenberg.org/files/98/98-0.txt',
        'filename': 'tale_of_two_cities.txt',
        'author':   'Charles Dickens',
    },
    'origin_of_species': {
        'url':      'https://www.gutenberg.org/files/2009/2009-0.txt',
        'filename': 'origin_of_species.txt',
        'author':   'Charles Darwin',
    },
    'einstein_relativity': {
        'url':      'https://www.gutenberg.org/files/5001/5001-0.txt',
        'filename': 'einstein_relativity.txt',
        'author':   'Albert Einstein',
    },
    'chemical_history_candle': {
        'url':      'https://www.gutenberg.org/files/14474/14474-0.txt',
        'filename': 'chemical_history_candle.txt',
        'author':   'Michael Faraday',
    },
    'descent_of_man': {
        'url':      'https://www.gutenberg.org/files/2300/2300-0.txt',
        'filename': 'descent_of_man.txt',
        'author':   'Charles Darwin',
    },
    # Edgar Allan Poe — 5 volumes of collected works (stories, poems, criticism)
    'poe_vol1': {
        'url':      'https://www.gutenberg.org/files/2147/2147-0.txt',
        'filename': 'poe_vol1.txt',
        'author':   'Edgar Allan Poe',
    },
    'poe_vol2': {
        'url':      'https://www.gutenberg.org/files/2148/2148-0.txt',
        'filename': 'poe_vol2.txt',
        'author':   'Edgar Allan Poe',
    },
    'poe_vol3': {
        'url':      'https://www.gutenberg.org/files/2149/2149-0.txt',
        'filename': 'poe_vol3.txt',
        'author':   'Edgar Allan Poe',
    },
    'poe_vol4': {
        'url':      'https://www.gutenberg.org/files/2150/2150-0.txt',
        'filename': 'poe_vol4.txt',
        'author':   'Edgar Allan Poe',
    },
    'poe_vol5': {
        'url':      'https://www.gutenberg.org/files/2151/2151-0.txt',
        'filename': 'poe_vol5.txt',
        'author':   'Edgar Allan Poe',
    },
    # Henry David Thoreau
    'thoreau_walden': {
        'url':      'https://www.gutenberg.org/files/205/205-0.txt',
        'filename': 'thoreau_walden.txt',
        'author':   'Henry David Thoreau',
    },
    'thoreau_civil_disobedience': {
        'url':      'https://www.gutenberg.org/files/71/71-0.txt',
        'filename': 'thoreau_civil_disobedience.txt',
        'author':   'Henry David Thoreau',
    },
    'thoreau_week': {
        'url':      'https://www.gutenberg.org/files/4280/4280-0.txt',
        'filename': 'thoreau_week.txt',
        'author':   'Henry David Thoreau',
    },
    # Baruch Spinoza
    'spinoza_ethics': {
        'url':      'https://www.gutenberg.org/cache/epub/3800/pg3800.txt',
        'filename': 'spinoza_ethics.txt',
        'author':   'Baruch Spinoza',
    },
    'spinoza_tractatus': {
        'url':      'https://www.gutenberg.org/files/989/989-0.txt',
        'filename': 'spinoza_tractatus.txt',
        'author':   'Baruch Spinoza',
    },
}
