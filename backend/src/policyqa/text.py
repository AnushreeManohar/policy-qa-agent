"""Shared lexical utilities: tokenisation, light stemming and sentence splitting."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an and are as at be been but by can could do does did for from had has have how i if in into
    is it its me my no not of on or our please so such than that the their them then there these they
    this those to under up was we were what when where which while who whom why will with would you
    your yours am any all also about after before being both each few more most other same some very
    just should shall may might must""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])|\n+")


def stem(tok: str) -> str:
    """Deliberately conservative suffix stripping, applied identically to questions and documents.

    deductibles -> deductibl, covered -> cover, enrollments -> enroll, ridesharing/rideshare -> rideshar,
    driving/drive -> driv. A trailing "e" is dropped last so "drive" and "driving" meet in the middle.
    """
    for suf, rep in (("ies", "y"), ("sses", "ss"), ("ments", ""), ("ment", ""), ("ing", ""), ("ed", ""),
                     ("es", ""), ("s", "")):
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            if suf == "s" and tok.endswith(("ss", "us", "is")):
                continue
            tok = tok[: len(tok) - len(suf)] + rep
            break
    if len(tok) >= 4 and tok.endswith("e") and not tok.endswith("ee"):
        tok = tok[:-1]
    return tok


def tokens(text: str, *, keep_stopwords: bool = False) -> list[str]:
    # "$1,000" -> "1000" so amounts stay one token ("$1,000" and "$1,500" no longer share "000").
    toks = _TOKEN.findall(_THOUSANDS.sub("", text.lower().replace("’", "'")))
    toks = [t.split("'")[0] for t in toks]
    if not keep_stopwords:
        toks = [t for t in toks if t not in STOPWORDS and len(t) > 1]
    return [stem(t) for t in toks]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if len(s.strip()) > 2]
