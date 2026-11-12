"""Section-aware chunking that preserves page provenance.

Insurance policies are organised by headed sections (COVERAGES, EXCLUSIONS, CONDITIONS...). Splitting
on those boundaries keeps each chunk about one topic, which makes both retrieval and citations
("Exclusions, p. 4") more precise than fixed-size windows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from policyqa.models import Chunk, PageText
from policyqa.security import scan_for_injection

_HEADING = re.compile(
    r"^(?:"
    r"(?:SECTION|PART|ARTICLE|SCHEDULE|ENDORSEMENT)\s+[A-Z0-9IVX]+\b.*"  # SECTION II - LIABILITY
    r"|\d{1,2}(?:\.\d{1,2})*\.?\s+[A-Z][A-Za-z ,&'/()\-]{2,80}"  # 4.2 Waiting Periods
    r"|[A-Z][A-Z0-9 ,&'/()\-]{3,80}"  # EXCLUSIONS
    r")$"
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z(\"'])")


def is_heading(line: str) -> bool:
    line = line.strip()
    if not (3 < len(line) <= 90) or line.endswith((".", ",", ";", ":")):
        return False
    if not _HEADING.match(line):
        return False
    letters = [c for c in line if c.isalpha()]
    # Numbered headings may be title-case; bare headings must be mostly uppercase.
    if re.match(r"^\d", line):
        return len(line.split()) <= 10
    return bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.8


_ROMAN = re.compile(r"^(?:[IVX]{1,4}|[A-Z])$")
_SMALL = {"a", "an", "and", "of", "or", "the", "to", "for", "in", "on", "by", "your"}


def heading_case(line: str) -> str:
    """'SECTION II - LIABILITY COVERAGES' -> 'Section II - Liability Coverages' (str.title() mangles numerals)."""
    if not line.isupper():
        return line
    words = []
    for i, w in enumerate(line.split()):
        if _ROMAN.match(w) or any(ch.isdigit() for ch in w):
            words.append(w)
        elif i > 0 and w.lower() in _SMALL:
            words.append(w.lower())
        else:
            words.append(w.capitalize())
    return " ".join(words)


@dataclass
class _Para:
    text: str
    page: int
    section: str | None


def _paragraphs(pages: list[PageText]) -> list[_Para]:
    out: list[_Para] = []
    section: str | None = None
    for p in pages:
        buf: list[str] = []
        for line in [*p.text.splitlines(), ""]:  # trailing "" flushes the last paragraph of the page
            stripped = line.strip()
            if re.fullmatch(r"(?i)page \d+( of \d+)?", stripped):
                continue  # running footers add noise to retrieval
            heading = bool(stripped) and is_heading(stripped)
            if (not stripped or heading) and buf:
                out.append(_Para(" ".join(buf).strip(), p.page, section))
                buf = []
            if heading:
                section = heading_case(stripped)
            elif stripped:
                buf.append(stripped)
    # Text before the first heading that is only a title line carries no answerable content.
    return [para for para in out if para.text and (para.section is not None or len(para.text) >= 80)]


def _split_long(para: _Para, size: int, overlap: int) -> list[_Para]:
    """Split an over-long paragraph on sentence boundaries, seeding each piece with the previous
    piece's last sentence (when it fits in `overlap`) so facts straddling a boundary stay retrievable."""
    if len(para.text) <= size:
        return [para]
    sents = _SENTENCE_SPLIT.split(para.text)
    pieces: list[str] = []
    cur: list[str] = []
    seeded = 0  # how many leading sentences of `cur` are overlap copied from the previous piece
    for sent in sents:
        if len(cur) > seeded and sum(len(x) + 1 for x in cur) + len(sent) > size:
            pieces.append(" ".join(cur))
            cur, seeded = ([cur[-1]], 1) if len(cur[-1]) <= overlap else ([], 0)
        cur.append(sent)
    fresh = cur[seeded:]
    if fresh:
        tail_new = " ".join(fresh)
        # Avoid a tiny trailing fragment: fold its new sentences into the previous piece if that stays reasonable.
        if pieces and len(tail_new) < size * 0.3 and len(pieces[-1]) + len(tail_new) <= size * 1.3:
            pieces[-1] = f"{pieces[-1]} {tail_new}"
        else:
            pieces.append(" ".join(cur))
    # Safety net for a single "sentence" far over size (e.g. a table extracted as one line):
    # wrap at whitespace so words are never cut.
    final: list[str] = []
    for piece in pieces:
        while len(piece) > size * 1.5:
            cut = piece.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            final.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        final.append(piece)
    return [_Para(t, para.page, para.section) for t in final if t]


def chunk_pages(doc_id: str, pages: list[PageText], *, size: int, overlap: int) -> list[Chunk]:
    paras: list[_Para] = []
    for para in _paragraphs(pages):
        paras.extend(_split_long(para, size, overlap))

    chunks: list[Chunk] = []
    cur: list[_Para] = []

    def emit() -> None:
        if not cur:
            return
        text = "\n".join(p.text for p in cur)
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id[:16]}:{len(chunks):04d}",
                doc_id=doc_id,
                ordinal=len(chunks),
                text=text,
                page_start=cur[0].page,
                page_end=cur[-1].page,
                section=cur[0].section,
                flags=["possible_injection"] if scan_for_injection(text) else [],
            )
        )

    for para in paras:
        cur_len = sum(len(p.text) + 1 for p in cur)
        section_changed = bool(cur) and para.section != cur[-1].section
        if cur and (section_changed or cur_len + len(para.text) > size):
            emit()
            # Overlap only within a section: carry trailing paragraphs for context continuity.
            carry: list[_Para] = []
            if not section_changed:
                for prev in reversed(cur):
                    if sum(len(p.text) for p in carry) + len(prev.text) > overlap:
                        break
                    carry.insert(0, prev)
            cur = carry
        cur.append(para)
    emit()
    return chunks
