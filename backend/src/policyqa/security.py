"""Trust-boundary helpers.

Policy documents and user questions are *data*. Nothing in them may be treated as an instruction.
Defence in depth:
  1. ingestion flags chunks that look like instructions aimed at an AI (stored as chunk flags);
  2. retrieved text is neutralised (tag-like markup stripped) and wrapped in <source> data blocks;
  3. the system prompt states that sources are untrusted data;
  4. flagged sentences are never used as answer evidence by the extractive path, and the
     verification node rejects LLM answers that are not supported by cited, unflagged text.
Pattern matching is a heuristic layer, not a guarantee — documented as such in the README.
"""

from __future__ import annotations

import re

_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ignore_instructions", re.compile(
        r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|all|earlier|system|your)\b"
        r".{0,30}\b(instructions?|prompts?|rules?|directions?)", re.I | re.S)),
    ("role_hijack", re.compile(r"\b(you are now|act as|pretend to be|from now on,? you)\b", re.I)),
    ("system_prompt_probe", re.compile(
        r"\b(system prompt|developer message|hidden instructions?|reveal your (prompt|instructions))\b", re.I)),
    ("ai_directed", re.compile(
        r"\b(AI|assistant|language model|LLM|chatbot)s?\b.{0,40}\b(must|should|will|shall)\b.{0,20}"
        r"\b(say|tell|respond|answer|state|reply|output)\b", re.I | re.S)),
    ("fake_role_tag", re.compile(r"(</?\s*(system|assistant|user|instructions?)\s*>|\[/?INST\]|<\|im_start\|>)", re.I)),
    # Added in the audit (H3) after a document injection without trigger words got through. Case A41 was
    # used to design these two patterns, so A41 passing afterwards is not independent evidence.
    ("addressed_to_ai", re.compile(
        r"\b(note|message|instructions?|notice)\s+(to|for)\s+(the\s+)?(reader'?s\s+)?"
        r"(ai|assistant|software|model|llm|chatbot|bot|summari[sz]er)\b", re.I)),
    ("summarizer_directive", re.compile(
        r"\bwhen\s+(summari[sz]ing|answering|responding|processing)\b.{0,60}\b(state|say|tell|claim|respond|"
        r"answer|report)\b", re.I | re.S)),
    ("exfiltration", re.compile(
        r"\b(send|post|email|upload)\b.{0,40}\b(api key|password|credentials|secrets?)\b", re.I | re.S)),
]

_TAGLIKE = re.compile(r"</?\s*[a-zA-Z_|][^<>]{0,40}>|\[/?INST\]|<\|[^|]{0,20}\|>")


def scan_for_injection(text: str) -> list[str]:
    """Return the names of injection heuristics that match `text`."""
    return [name for name, pat in _INJECTION_PATTERNS if pat.search(text)]


def neutralize(text: str) -> str:
    """Strip markup that could be confused with prompt structure before text enters a prompt."""
    return _TAGLIKE.sub(" ", text)


def is_flagged_sentence(sentence: str) -> bool:
    return bool(scan_for_injection(sentence))
