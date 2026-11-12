"""The LangGraph agent.

    guard_input ──refuse──────────────────────────────────────────────▶ refuse ─▶ END
        │
        ▼
    plan_query ─▶ retrieve ─▶ grade_evidence ──sufficient──▶ generate ─▶ verify ──pass──▶ END
                     ▲              │                          ▲          │
                     └──broaden─────┘ (drop policy filter)     └─retry────┘ (once, with feedback)
                                    │                                     │
                                    └──insufficient──▶ abstain ◀──────────┘ (nothing verifiable left)

Why a graph rather than a single prompt: abstention, query broadening and answer verification are
explicit, testable control-flow decisions with their own state, not hopes buried in a prompt.
"""

from __future__ import annotations

import logging
import operator
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from policyqa import grounding
from policyqa.agent.llm import LLMError, build_chat_model, message_text, parse_json_object
from policyqa.agent.prompts import PLANNER_PROMPT, SYSTEM_PROMPT, answer_prompt
from policyqa.config import Settings
from policyqa.models import Citation, RetrievedChunk
from policyqa.retrieval import HybridRetriever
from policyqa.security import is_flagged_sentence, scan_for_injection
from policyqa.text import sentences, tokens

logger = logging.getLogger(__name__)

REFUSE_PATTERNS = {"ignore_instructions", "system_prompt_probe", "role_hijack", "exfiltration", "fake_role_tag"}
_TYPE_HINTS = {
    "homeowners": {"homeowner", "dwelling", "house", "home"},
    "auto": {"auto", "car", "vehicle", "driver", "collision"},
    "pet": {"pet", "dog", "cat", "veterinary", "vet"},
}
_SPLIT = re.compile(r"\?\s+|;\s+|\b(?:and|also)\s+(?=(?:what|how|is|are|does|do|can|when|which|who|will)\b)", re.I)
_QUANTITY = re.compile(r"\b(how (much|long|many|soon|quickly|fast|old)|"
                       r"what is the (deductible|limit|premium|maximum|minimum)|"
                       r"deductible|limit|days?|percent|premium|reimburse)\b", re.I)


class AgentState(TypedDict, total=False):
    question: str
    policy_type: str | None
    routed_type: str | None  # policy type the question was routed to (user-given or detected)
    user_scoped: bool  # True when the CALLER chose the policy type: never broaden beyond it (audit M1)
    degraded: bool  # True when the LLM failed and the answer fell back to extractive mode
    queries: list[str]
    subquestions: list[str]
    attempts: int
    retrieved: list[RetrievedChunk]
    retrieval_ms: float
    coverage: float
    missing_terms: list[str]
    answer: str
    sufficient: bool
    regenerations: int
    feedback: str | None
    faithfulness: float | None
    status: str
    trace: Annotated[list[dict[str, Any]], operator.add]


@dataclass
class AgentResult:
    question: str
    answer: str
    status: str  # answered | partial | insufficient_information | refused
    citations: list[Citation]
    sources: list[dict[str, Any]]
    policy_type: str | None
    coverage: float | None
    missing_terms: list[str]
    faithfulness: float | None
    mode: str
    degraded: bool
    latency_ms: float
    retrieval_ms: float | None
    trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _timed(name: str):  # type: ignore[no-untyped-def]
    def deco(fn):  # type: ignore[no-untyped-def]
        def wrapper(self: PolicyQAAgent, state: AgentState) -> dict:
            t0 = time.perf_counter()
            out = fn(self, state)
            info = out.pop("_info", {})
            out["trace"] = [{"node": name, "ms": round((time.perf_counter() - t0) * 1000, 2), **info}]
            return out
        return wrapper
    return deco


class PolicyQAAgent:
    def __init__(self, retriever: HybridRetriever, settings: Settings, chat_model: Any | None = None) -> None:
        self.retriever = retriever
        self.settings = settings
        self.llm = chat_model if chat_model is not None else build_chat_model(settings)
        self.mode = "llm" if self.llm is not None else "extractive"
        self.graph = self._build()

    # ------------------------------------------------------------------ graph wiring
    def _build(self):  # type: ignore[no-untyped-def]
        g = StateGraph(AgentState)
        for name in ("guard_input", "plan_query", "retrieve", "grade_evidence", "generate", "verify",
                     "abstain", "refuse"):
            g.add_node(name, getattr(self, name))
        g.add_edge(START, "guard_input")
        g.add_conditional_edges("guard_input", lambda s: "refuse" if s.get("status") == "refused" else "plan_query",
                                ["refuse", "plan_query"])
        g.add_edge("plan_query", "retrieve")
        g.add_edge("retrieve", "grade_evidence")
        g.add_conditional_edges("grade_evidence", self._route_after_grade, ["generate", "retrieve", "abstain"])
        g.add_edge("generate", "verify")
        g.add_conditional_edges("verify", self._route_after_verify, ["generate", "abstain", END])
        g.add_edge("abstain", END)
        g.add_edge("refuse", END)
        return g.compile()

    def _route_after_grade(self, s: AgentState) -> str:
        if s["coverage"] >= self.settings.min_term_coverage and s["retrieved"]:
            return "generate"
        if s.get("policy_type") and not s.get("user_scoped") and s["attempts"] < self.settings.max_retrieval_attempts:
            return "retrieve"  # broaden: retry without the policy-type filter
        return "abstain"

    def _route_after_verify(self, s: AgentState) -> str:
        if s.get("status") in ("answered", "partial"):
            return END
        if s.get("feedback") and s.get("regenerations", 0) <= self.settings.max_regenerations:
            return "generate"
        return "abstain"

    # ------------------------------------------------------------------ nodes
    @_timed("guard_input")
    def guard_input(self, s: AgentState) -> dict:
        hits = set(scan_for_injection(s["question"]))
        if hits & REFUSE_PATTERNS:
            return {"status": "refused", "_info": {"flags": sorted(hits)}}
        return {"attempts": 0, "regenerations": 0, "_info": {"flags": sorted(hits)}}

    @_timed("plan_query")
    def plan_query(self, s: AgentState) -> dict:
        q = s["question"]
        ptype = s.get("policy_type")
        user_scoped = ptype is not None
        if not ptype:
            toks = set(tokens(q)) | set(re.findall(r"[a-z]+", q.lower()))
            found = [t for t, hints in _TYPE_HINTS.items() if toks & hints]
            ptype = found[0] if len(found) == 1 else None

        queries: list[str] = []
        if self.llm is not None:
            try:
                msg = self.llm.invoke(PLANNER_PROMPT.replace("__QUESTION__", q))
                obj = parse_json_object(message_text(msg)) or {}
                queries = [str(x)[:300] for x in obj.get("queries", []) if str(x).strip()][:3]
            except Exception as exc:  # noqa: BLE001 - planning is an optimisation; fall back to heuristics
                logger.warning("planner failed, using heuristic plan: %s", type(exc).__name__)
        if not queries:
            parts = [p.strip(" ?.") for p in _SPLIT.split(q) if p and len(tokens(p)) >= 1]
            queries = parts[:3] or [q]
        # Thesaurus expansion bridges everyday words to policy vocabulary (car -> auto/vehicle).
        expanded = []
        for sub in queries:
            extra = [t for t in grounding.expand_terms(grounding.key_terms(sub)) if t not in tokens(sub)]
            expanded.append(f"{sub} {' '.join(extra)}".strip())
        return {"queries": expanded, "subquestions": queries, "policy_type": ptype, "routed_type": ptype,
                "user_scoped": user_scoped,
                "_info": {"queries": expanded, "policy_type": ptype}}

    @_timed("retrieve")
    def retrieve(self, s: AgentState) -> dict:
        attempts = s.get("attempts", 0)
        ptype = s.get("policy_type") if attempts == 0 else None  # second attempt broadens
        per_query = [self.retriever.retrieve(q, policy_type=ptype) for q in s["queries"]]
        # Each extra sub-question adds half a budget (capped) so no part of a multi-part question starves.
        k = min(self.settings.retrieval_top_k + (len(per_query) - 1) * (self.settings.retrieval_top_k // 2), 12)
        # When broadening, keep the on-policy evidence from the first attempt and only *add* to it
        # (bug fix: previously a broadened search replaced it, so a pet question got answered from the
        # auto policy). On-policy chunks stay first, so they also win ranking ties downstream.
        merged: list[RetrievedChunk] = list(s.get("retrieved") or [])[: k // 2] if attempts else []
        seen: set[str] = {c.chunk.chunk_id for c in merged}
        kept = len(merged)
        flagged: list[RetrievedChunk] = []
        # Round-robin across sub-queries so every part of a multi-part question gets evidence.
        for rank in range(max((len(r.chunks) for r in per_query), default=0)):
            for r in per_query:
                c = r.chunks[rank] if rank < len(r.chunks) else None
                if c is None or c.chunk.chunk_id in seen:
                    continue
                if c.chunk.flags:  # shown to the user as ignored; never counts toward the evidence budget
                    seen.add(c.chunk.chunk_id)
                    flagged.append(c)
                elif sum(1 for m in merged if not m.chunk.flags) < k:
                    seen.add(c.chunk.chunk_id)
                    merged.append(c)
        merged += flagged[:2]
        ms = sum(r.latency_ms for r in per_query)
        return {
            "retrieved": merged, "attempts": attempts + 1, "retrieval_ms": ms,
            "_info": {"attempt": attempts + 1, "filter": ptype, "hits": len(merged),
                      "kept_from_previous": kept,
                      "top": [c.chunk.chunk_id for c in merged[:3]], "retrieval_ms": round(ms, 2)},
        }

    @_timed("grade_evidence")
    def grade_evidence(self, s: AgentState) -> dict:
        usable = [c for c in s["retrieved"] if not c.chunk.flags]
        texts = [f"{c.title} {c.chunk.section or ''} {c.chunk.text}" for c in usable]
        coverage, missing = grounding.term_coverage(s["question"], texts) if texts else (0.0, [])
        missing = grounding.surface_forms(s["question"], missing)
        return {"coverage": coverage, "missing_terms": missing,
                "_info": {"coverage": round(coverage, 3), "missing": missing, "usable_chunks": len(usable)}}

    @_timed("generate")
    def generate(self, s: AgentState) -> dict:
        if self.llm is None:
            answer = self._extractive_answer(s)
            return {"answer": answer, "sufficient": bool(answer), "_info": {"mode": "extractive"}}
        try:
            msg = self.llm.invoke([("system", SYSTEM_PROMPT),
                                   ("human", answer_prompt(s["question"], s["retrieved"], s.get("feedback")))])
        except Exception as exc:
            if not self.settings.llm_fallback_extractive:
                raise LLMError("language model call failed") from exc
            # Degrade instead of failing (audit M3): quote the evidence directly and say so.
            logger.warning("llm failed, falling back to extractive answer: %s", type(exc).__name__)
            answer = self._extractive_answer(s)
            return {"answer": answer, "sufficient": bool(answer), "degraded": True,
                    "_info": {"mode": "extractive_fallback", "error": type(exc).__name__}}
        text = message_text(msg)
        obj = parse_json_object(text)
        answer = str(obj.get("answer", "")) if obj else text
        sufficient = bool(obj.get("sufficient", True)) if obj else True
        return {"answer": answer.strip(), "sufficient": sufficient,
                "_info": {"mode": "llm", "sufficient": sufficient, "retry": bool(s.get("feedback"))}}

    @_timed("verify")
    def verify(self, s: AgentState) -> dict:
        retrieved, answer = s["retrieved"], s.get("answer", "")
        if not s.get("sufficient") or not answer:
            return {"status": "insufficient", "feedback": None, "_info": {"result": "model_reported_insufficient"}}
        ids = grounding.cited_ids(answer)
        problems = []
        if not ids:
            problems.append("no citations")
        bad = [i for i in ids if i < 1 or i > len(retrieved)]
        if bad:
            problems.append(f"citations {bad} do not exist")
        flagged = [i for i in ids if 1 <= i <= len(retrieved) and retrieved[i - 1].chunk.flags]
        if flagged:
            problems.append(f"sources {flagged} contain instruction-like text and cannot be evidence")
        valid = [i for i in ids if 1 <= i <= len(retrieved) and i not in flagged]
        evidence = [retrieved[i - 1].chunk.text for i in valid]
        faith, detail = grounding.answer_support(answer, evidence)
        weak = [sent for sent, v in detail if v < self.settings.min_sentence_support]
        if weak:
            problems.append(f"{len(weak)} sentence(s) not supported by the cited sources")

        if not problems:
            status = "partial" if s.get("missing_terms") else "answered"
            return {"status": status, "faithfulness": faith, "feedback": None,
                    "_info": {"result": "pass", "faithfulness": round(faith, 3)}}

        if self.llm is not None and s.get("regenerations", 0) < self.settings.max_regenerations:
            return {"feedback": "; ".join(problems) + ".", "faithfulness": faith,
                    "regenerations": s.get("regenerations", 0) + 1,
                    "_info": {"result": "retry", "problems": problems}}

        # Last resort: keep only cited, supported sentences; abstain if nothing survives.
        kept = [sent for sent, v in detail if v >= self.settings.min_sentence_support
                and grounding.cited_ids(sent) and not set(grounding.cited_ids(sent)) - set(valid)]
        if kept:
            return {"answer": " ".join(kept), "status": "partial", "faithfulness":
                    grounding.answer_support(" ".join(kept), evidence)[0], "feedback": None,
                    "_info": {"result": "pruned", "problems": problems, "kept": len(kept)}}
        return {"status": "insufficient", "feedback": None, "_info": {"result": "fail", "problems": problems}}

    @_timed("abstain")
    def abstain(self, s: AgentState) -> dict:
        missing = s.get("missing_terms") or []
        msg = "I couldn't find enough information in the policy documents to answer that reliably."
        if missing:
            msg += " The documents don't appear to address: " + ", ".join(missing[:5]) + "."
        msg += " Please check the full policy wording or contact the insurer."
        return {"answer": msg, "status": "insufficient_information", "_info": {}}

    @_timed("refuse")
    def refuse(self, s: AgentState) -> dict:
        return {"answer": "I can only answer questions about the content of the loaded insurance policy "
                          "documents. I can't follow instructions to change how I operate or reveal my configuration.",
                "status": "refused", "retrieved": [], "_info": {}}

    # ------------------------------------------------------------------ extractive generator
    def _extractive_answer(self, s: AgentState) -> str:
        """Deterministic, no-LLM answer built from verbatim retrieved sentences, each cited.

        For every sub-question, sentences are scored by IDF-weighted coverage of the sub-question's term
        groups (rare, specific terms dominate), plus a small bonus for numeric sentences when a quantity is
        asked for and a small prior for higher-ranked chunks. The best sentence per sub-question is always
        considered; near-ties (>= 85% of the best) are added, capped at 2 per part and 4 overall.
        """
        pool: list[tuple[int, int, str, set[str]]] = []  # (source idx, position, sentence, tokens)
        routed = s.get("routed_type")
        on_policy = routed is not None and any(rc.policy_type == routed for rc in s["retrieved"])
        for idx, rc in enumerate(s["retrieved"], start=1):
            if rc.chunk.flags or (on_policy and rc.policy_type != routed):
                continue  # never answer a pet question with an auto clause when pet evidence exists
            carry = ""
            for pos, sent in enumerate(sentences(rc.chunk.text)):
                if len(sent.split()) < 5:  # run-in heading ("Free look period.") -> prefix of the next sentence
                    carry = f"{carry} {sent}".strip()
                    continue
                sent = f"{carry} {sent}".strip() if carry else sent
                carry = ""
                toks = set(tokens(sent))
                if len(toks) >= 3 and not is_flagged_sentence(sent):
                    pool.append((idx, pos, sent, toks))
        if not pool:
            return ""
        token_sets = [p[3] for p in pool]
        pool_texts = [p[2] for p in pool]
        by_chunk: dict[int, list[tuple[int, str, set[str]]]] = {}
        for idx, pos, sent, toks in pool:
            by_chunk.setdefault(idx, []).append((pos, sent, toks))
        picked: dict[tuple[int, int], str] = {}
        part: dict[tuple[int, int], int] = {}  # which sub-question a sentence answers
        for part_no, sub in enumerate(s.get("subquestions") or [s["question"]]):
            groups = grounding.content_groups(sub, pool_texts) or [(t, {t}) for t in tokens(sub)]
            weights = grounding.idf_weights(groups, token_sets)
            # Normalise only over terms that occur somewhere in the evidence: a term found nowhere ("fully")
            # cannot be matched by any sentence; whether that matters was already decided by grade_evidence.
            present = [name for name, members in groups if any(members & ts for ts in token_sets)]
            total = sum(weights[n] for n in present) or 1.0
            generic = grounding.generic_groups(sub)
            gweights = grounding.idf_weights(generic, token_sets)
            wants_quantity = bool(_QUANTITY.search(sub))
            scored = []
            for idx, pos, sent, toks in pool:
                matched = frozenset(name for name, members in groups if members & toks)
                cov = sum(weights[name] for name in matched) / total
                if cov == 0:
                    continue
                generic_bonus = 0.15 * sum(gweights[g] for g, m in generic if m & toks) / (sum(gweights.values()) or 1)
                bonus = (0.1 if wants_quantity and re.search(r"\$?\d", sent) else 0.0) + 0.05 / idx + generic_bonus
                scored.append((cov + bonus, cov, idx, pos, sent, matched))
            scored.sort(key=lambda x: (-x[0], x[2], x[3]))
            if not scored or scored[0][1] < 0.34:
                continue  # nothing in the evidence addresses this part
            best, best_terms = scored[0][0], scored[0][5]
            taken = 0
            for score, _cov, idx, pos, sent, matched in scored:
                if taken == 2 or len(picked) == 4 or score < 0.85 * best:
                    break
                # A supporting sentence must address everything the best sentence addressed; a tie on a
                # *different* term ("driving record" vs "Mexico") is a different topic, not support.
                if taken and not best_terms <= matched:
                    continue
                if (idx, pos) not in picked:
                    picked[(idx, pos)] = sent
                    part[(idx, pos)] = part_no
                    taken += 1
            # Quantity questions: the figure often sits in the sentence right after a topic sentence
            # ("Coverage D pays additional living expenses ... . Payment is ... not more than 12 months.").
            if wants_quantity:
                group_terms = {t for _, m in groups for t in m}
                for (p_idx, p_pos) in list(picked):
                    follow = next((x for x in by_chunk.get(p_idx, []) if x[0] == p_pos + 1), None)
                    if (follow and len(picked) < 5 and (p_idx, p_pos + 1) not in picked
                            and re.search(r"\$?\d", follow[1])
                            and (follow[2] & group_terms or follow[1].lower().startswith(("this ", "payment")))):
                        picked[(p_idx, p_pos + 1)] = follow[1]
                        part[(p_idx, p_pos + 1)] = part[(p_idx, p_pos)]
        # Answer the parts in the order they were asked; within a part, follow document order.
        ordered = sorted(picked.items(), key=lambda kv: (part[kv[0]], kv[0]))
        return " ".join(f"{sent.rstrip()} [{idx}]" for (idx, _), sent in ordered)

    # ------------------------------------------------------------------ public API
    def ask(self, question: str, policy_type: str | None = None) -> AgentResult:
        t0 = time.perf_counter()
        final: AgentState = self.graph.invoke({"question": question, "policy_type": policy_type, "trace": [],
                                               "degraded": False})
        status = final.get("status", "insufficient_information")
        if status == "insufficient":
            status = "insufficient_information"
        retrieved = final.get("retrieved", []) or []
        answer = final.get("answer", "")
        if status == "partial" and final.get("missing_terms"):
            answer += ("\n\nNote: the documents don't mention " + ", ".join(final["missing_terms"][:5])
                       + ", so this answer may not fully address the question.")
        cited = set(grounding.cited_ids(answer)) if status in ("answered", "partial") else set()
        citations = [self._citation(i, retrieved[i - 1]) for i in sorted(cited) if 1 <= i <= len(retrieved)]
        sources = [{"id": i, "chunk_id": rc.chunk.chunk_id, "title": rc.title, "filename": rc.filename,
                    "section": rc.chunk.section, "page_start": rc.chunk.page_start, "page_end": rc.chunk.page_end,
                    "score": round(rc.score, 4), "flags": rc.chunk.flags, "cited": i in cited,
                    "snippet": rc.chunk.text[:400]} for i, rc in enumerate(retrieved, start=1)]
        return AgentResult(
            question=question, answer=answer, status=status, citations=citations, sources=sources,
            policy_type=final.get("policy_type"), coverage=final.get("coverage"),
            missing_terms=final.get("missing_terms", []) or [], faithfulness=final.get("faithfulness"),
            mode=self.mode, degraded=bool(final.get("degraded")),
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            retrieval_ms=round(final["retrieval_ms"], 2) if final.get("retrieval_ms") is not None else None,
            trace=final.get("trace", []),
        )

    @staticmethod
    def _citation(i: int, rc: RetrievedChunk) -> Citation:
        c = rc.chunk
        # Full clause text (chunks are <= ~1.8k chars) so the UI can show and highlight the cited sentence.
        return Citation(i, c.chunk_id, c.doc_id, rc.filename, rc.title, c.page_start, c.page_end, c.section, c.text)
