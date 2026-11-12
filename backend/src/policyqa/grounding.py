"""Lexical grounding signals used by the agent (evidence gating, verification) and by the evaluator.

These are deliberately simple and transparent: they can be explained in an interview and audited in
code. They measure *lexical* support, so they under-credit good paraphrase and can be fooled by
answers that reuse document words in a wrong way — the README states both limitations.
"""

from __future__ import annotations

import math
import re

from policyqa.semantics import word_vectors, words_in
from policyqa.text import STOPWORDS, sentences, stem, tokens

# Words that appear in nearly every policy question/document and therefore say nothing about whether
# the documents address *this* question.
GENERIC_TERMS = frozenset(
    stem(w)
    for w in """policy policies insurance insured insurer cover covered coverage covers plan claim claims
    benefit benefits pay paid payable get need much many long happen happens does mean means include
    included includes apply applies under amount company provide provided term terms condition
    conditions type kind would like know tell explain what which problem problems issue issues question
    situation thing things anything something happen happens drive drives driving driver drivers""".split()
)

# Conversational filler: common English words that carry no information about *which* policy provision
# a question is about. They are excluded from key terms (but still help rank sentences, see below).
FILLER_TERMS = frozenset(
    stem(w)
    for w in """go goes going went gone get gets got getting because since quickly quick fast soon old new
    ordinary regular normal usual usually typical really actually exactly just still ever even ok okay want
    wanted wonder think guess say said someone somebody person people lot lots bit little big good bad best
    worst right wrong sure able help use using dollar dollars money amount exact certain possible
    fully completely totally entirely absolutely definitely
    """.split()
) - {"money"}

THESAURUS: dict[str, list[str]] = {
    "car": ["auto", "vehicle"], "automobile": ["auto", "vehicle"], "truck": ["vehicle"],
    "house": ["dwelling", "home"], "home": ["dwelling"], "apartment": ["dwelling"],
    "dog": ["pet"], "cat": ["pet"], "puppy": ["pet"], "kitten": ["pet"], "vet": ["veterinary"],
    "stolen": ["theft"], "steal": ["theft"], "robbery": ["theft"], "burglary": ["theft"],
    "quake": ["earthquake"], "hurt": ["injury"], "injured": ["injury"], "cancel": ["cancellation"],
    "cost": ["premium"], "price": ["premium"], "sick": ["illness"], "surgery": ["treatment"],
    "crash": ["collision", "accident"], "wreck": ["collision"], "towing": ["tow"],
    "leak": ["discharge", "overflow"], "leaking": ["discharge"], "pipe": ["plumbing"],
    "son": ["family", "resident"], "daughter": ["family", "resident"], "child": ["family", "resident"],
    "kid": ["family", "resident"], "teenage": ["family", "resident"], "teenager": ["family", "resident"],
    "wife": ["spouse"], "husband": ["spouse"], "gun": ["firearm"], "guns": ["firearm"],
    "lawsuit": ["suit"], "sued": ["suit"], "lawyer": ["defense", "counsel"], "paid": ["pay"],
}
# Keyed by *stemmed* word so lookups agree with tokens() output (bug fix: "house" stems to "hous").
_THES: dict[str, set[str]] = {stem(k): {stem(x) for x in v} for k, v in THESAURUS.items()}

# Words that identify WHICH policy a question is about. They route retrieval (policy-type filter) but say
# nothing about which provision answers the question, so sentence selection ignores them.
POLICY_TYPE_TERMS = frozenset(stem(w) for w in """homeowner homeowners home house dwelling auto car vehicle
    pet pets dog cat""".split())

_CITE = re.compile(r"\[\d+\]")


def key_terms(question: str) -> list[str]:
    """Specific, informative question terms (generic insurance words and conversational filler removed)."""
    seen: list[str] = []
    for t in tokens(question):
        if t not in GENERIC_TERMS and t not in FILLER_TERMS and t not in seen:
            seen.append(t)
    return seen


def generic_terms(question: str) -> list[str]:
    """Generic insurance words in the question ("claim", "paid"). Too common to decide *whether* the
    documents address a question, but useful to rank sentences once they do."""
    seen: list[str] = []
    for t in tokens(question):
        if t in GENERIC_TERMS and t not in seen:
            seen.append(t)
    return seen


def expand_terms(terms: list[str]) -> list[str]:
    out = list(terms)
    for t in terms:
        for syn in sorted(_THES.get(t, ())):
            if syn not in out:
                out.append(syn)
    return out


def term_groups(question: str) -> list[tuple[str, set[str]]]:
    """Each specific question term with its synonyms; a group is satisfied if any member appears."""
    return [(t, {t, *_THES.get(t, ())}) for t in key_terms(question)]


_RAW_WORD = re.compile(r"[a-z0-9]+")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")


def raw_words(question: str) -> list[str]:
    """Lower-cased, unstemmed, non-stopword words of the question."""
    return [w for w in _RAW_WORD.findall(_THOUSANDS.sub("", question.lower())) if w not in STOPWORDS]


def query_words(question: str) -> list[str]:
    """UNSTEMMED keyword-search words: the question's specific words plus thesaurus synonyms.

    Bug fix (audit H1): stores used to receive our own stems ("deductibl"), and PostgreSQL then stemmed them
    again with its English stemmer ("deductibl" vs the indexed "deduct"), so core terms never matched. Each
    store now applies its own stemmer to raw words, consistently with how it indexed the documents."""
    keys = set(key_terms(question))
    out: list[str] = []
    for w in raw_words(question):
        if stem(w) in keys:
            for x in [w, *THESAURUS.get(w, [])]:
                if x not in out:
                    out.append(x)
    return out or raw_words(question)


def query_terms(question: str) -> list[str]:
    """Lexical search terms: specific terms plus synonyms; all tokens only if nothing specific remains.
    Generic words ("cover", "claim") match almost every chunk and drown out the decisive term ("Mexico")."""
    return expand_terms(key_terms(question)) or tokens(question)


def generic_groups(question: str) -> list[tuple[str, set[str]]]:
    return [(t, {t, *_THES.get(t, ())}) for t in generic_terms(question)]


def soften(question: str, groups: list[tuple[str, set[str]]],
           evidence_texts: list[str] | None) -> list[tuple[str, set[str]]]:
    """Add evidence words that are near-synonyms of each question term (pretrained word vectors).
    No-op when the semantic layer is unavailable."""
    wv = word_vectors()
    if wv is None or not evidence_texts or not groups:
        return groups
    vocab = words_in(evidence_texts)
    vocab_stems = {stem(w) for w in vocab}
    surface = dict(zip([g[0] for g in groups], surface_forms(question, [g[0] for g in groups]), strict=True))
    # Only for words the evidence does NOT contain. Word vectors put co-hyponyms close together ("mexico" ~
    # "canada"): letting a present word also match its neighbours made the agent quote the Canada clause for
    # a Mexico question (caught by test_multi_part_question_answers_each_part during the audit).
    # Numbers are never soft-matched: their vectors are close to unrelated units ("16" ~ "days", "month").
    return [(name, members if (members & vocab_stems or name.isdigit())
             else members | wv.neighbors(surface[name], vocab))
            for name, members in groups]


def content_groups(question: str, evidence_texts: list[str] | None = None) -> list[tuple[str, set[str]]]:
    """Term groups for *sentence selection*: key terms minus policy-type words; if nothing remains
    ("How quickly are pet claims paid?") fall back to the question's generic insurance terms."""
    groups = [g for g in term_groups(question) if g[0] not in POLICY_TYPE_TERMS] or generic_groups(question)
    return soften(question, groups, evidence_texts)


def topic_stems(question: str) -> set[str] | None:
    """Stems of the question's nouns/proper nouns/adjectives, or None without the semantic layer."""
    wv = word_vectors()
    return None if wv is None else {stem(w) for w in wv.topic_words(question)}


def idf_weights(groups: list[tuple[str, set[str]]], token_sets: list[set[str]]) -> dict[str, float]:
    """IDF of each group over the given texts. A term absent everywhere gets the maximum weight: an
    unmatched specific word ("cryptocurrency") is the strongest signal that the documents don't cover it."""
    n = max(len(token_sets), 1)
    return {name: math.log(1 + n / (1 + sum(1 for ts in token_sets if members & ts)))
            for name, members in groups}


def term_coverage(question: str, texts: list[str]) -> tuple[float, list[str]]:
    """IDF-weighted share of the question's terms present in `texts`, plus the missing topic terms.

    Present words always count as evidence. Absent words count AGAINST the question only if they are topics
    (nouns/adjectives): a missing "roof" or "cryptocurrency" matters, a missing "start" or "newly" does not.
    (First audit fix used topics only, which dropped present verbs like "damages" and over-abstained.)"""
    groups = [g for g in term_groups(question) if not g[0].isdigit()]
    if not groups:
        return 1.0, []
    token_sets = [set(tokens(t)) for t in texts]
    vocab = set().union(*token_sets) if token_sets else set()
    topics = topic_stems(question)
    if topics is not None:
        kept = [g for g in groups if g[1] & vocab or g[0] in topics]
        groups = kept or groups
    w = idf_weights(groups, token_sets)
    missing = [name for name, members in groups if not members & vocab]
    total = sum(w.values())
    covered = sum(w[name] for name, members in groups if members & vocab)
    return (covered / total if total else 1.0), missing


def surface_forms(question: str, stems: list[str]) -> list[str]:
    """Map internal stems back to the words the user typed ("exchang" -> "exchange")."""
    words = re.findall(r"[A-Za-z0-9]+", question)
    by_stem: dict[str, str] = {}
    for w in words:
        by_stem.setdefault(stem(w.lower()), w.lower())
    return [by_stem.get(s, s) for s in stems]


def sentence_support(sentence: str, evidence_texts: list[str]) -> float:
    """Share of a sentence's content tokens found in the evidence (0..1)."""
    toks = tokens(_CITE.sub(" ", sentence))
    if not toks:
        return 1.0
    vocab: set[str] = set()
    for t in evidence_texts:
        vocab.update(tokens(t))
    return sum(1 for t in toks if t in vocab) / len(toks)


def answer_support(answer: str, evidence_texts: list[str]) -> tuple[float, list[tuple[str, float]]]:
    """Mean sentence support plus per-sentence detail (bullets/markup ignored)."""
    sents = [s.lstrip("-*• ").strip() for s in sentences(answer)]
    sents = [s for s in sents if len(tokens(s)) >= 3]
    if not sents:
        return 0.0, []
    detail = [(s, sentence_support(s, evidence_texts)) for s in sents]
    return sum(v for _, v in detail) / len(detail), detail


def cited_ids(answer: str) -> list[int]:
    return sorted({int(m[1:-1]) for m in _CITE.findall(answer)})
