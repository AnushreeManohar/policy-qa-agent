"""Safe PDF text extraction.

PDFs are untrusted input: we verify the magic bytes, refuse encrypted files, cap page count and
per-page text, and convert any parser failure into a typed error instead of a 500.
"""

from __future__ import annotations

import io
import logging
import re
import unicodedata

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from policyqa.models import PageText

logger = logging.getLogger(__name__)


class PdfValidationError(ValueError):
    """Raised for files we refuse to process; message is safe to show to a client."""


_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_MULTISPACE = re.compile(r"[ \t]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL.sub(" ", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)  # re-join words hyphenated across line breaks
    text = _MULTISPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return _MANY_NEWLINES.sub("\n\n", text).strip()


def extract_pages(data: bytes, *, max_pages: int, max_chars_per_page: int) -> list[PageText]:
    return extract_pdf(data, max_pages=max_pages, max_chars_per_page=max_chars_per_page)[0]


def extract_pdf(data: bytes, *, max_pages: int, max_chars_per_page: int) -> tuple[list[PageText], str | None]:
    """Return (pages, metadata title)."""
    if not data:
        raise PdfValidationError("File is empty.")
    if not data[:1024].lstrip().startswith(b"%PDF-"):
        raise PdfValidationError("File is not a PDF.")
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            raise PdfValidationError("Encrypted PDFs are not supported.")
        n = len(reader.pages)
        meta_title = (reader.metadata.title if reader.metadata else None) or None
    except PdfValidationError:
        raise
    except (PdfReadError, ValueError, KeyError, TypeError, OSError) as exc:
        logger.warning("pdf parse failure: %s", type(exc).__name__)
        raise PdfValidationError("PDF could not be parsed.") from exc
    if n == 0:
        raise PdfValidationError("PDF has no pages.")
    if n > max_pages:
        raise PdfValidationError(f"PDF has {n} pages; the limit is {max_pages}.")

    pages: list[PageText] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - a single bad page must not kill the document
            logger.warning("page %d extraction failed: %s", i, type(exc).__name__)
            raw = ""
        pages.append(PageText(page=i, text=clean_text(raw)[:max_chars_per_page]))

    if not any(p.text for p in pages):
        raise PdfValidationError("PDF contains no extractable text (it may be scanned images).")
    if meta_title:
        meta_title = clean_text(str(meta_title))[:200] or None
    return pages, meta_title


def guess_title(pages: list[PageText], fallback: str, meta_title: str | None = None) -> str:
    if meta_title and len(meta_title) >= 6 and meta_title.lower() not in ("untitled", "document", "microsoft word"):
        return meta_title
    for line in pages[0].text.splitlines():
        line = line.strip()
        if 6 <= len(line) <= 120 and not line.lower().startswith(("page ", "policy number")):
            return line
    return fallback
