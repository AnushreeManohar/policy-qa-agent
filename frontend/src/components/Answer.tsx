import { useState } from "react";
import type { AskResponse } from "../api";
import { pages, segment, sentencesByCitation } from "../citations";
import { Clause } from "./Clause";

interface Props {
  turnId: number;
  result: AskResponse;
}

export function Answer({ turnId, result }: Props) {
  const [flash, setFlash] = useState<number | null>(null);
  const bySentence = sentencesByCitation(result.answer);
  const others = result.sources.filter((s) => !s.cited);

  const focusNote = (id: number) => {
    document.getElementById(`note-${turnId}-${id}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    setFlash(id);
    window.setTimeout(() => setFlash(null), 1400);
  };

  if (result.status === "refused") {
    return (
      <div className="answer answer--refused" role="status">
        <p className="answer__lead">Not something this assistant can do</p>
        <p>{result.answer}</p>
      </div>
    );
  }

  if (result.status === "insufficient_information") {
    return (
      <div className="answer answer--notfound" role="status">
        <p className="answer__lead">Not found in these policies</p>
        <p>
          The loaded policy documents don't contain enough to answer this reliably, so no answer was given.
          {result.missing_terms.length > 0 && (
            <> Nothing in them addresses <strong>{result.missing_terms.slice(0, 5).join(", ")}</strong>.</>
          )}
        </p>
        <p className="answer__hint">Try naming the policy and the provision, for example "homeowners flood exclusion".</p>
      </div>
    );
  }

  const [body, ...notes] = result.answer.split(/\n\n+/);
  return (
    <div className="answer">
      <p className="answer__text">
        {segment(body).map((s, i) =>
          s.kind === "text" ? (
            <span key={i}>{s.text}</span>
          ) : (
            <button key={i} type="button" className="cite" onClick={() => focusNote(s.id)}
                    aria-label={`Show source ${s.id}`}>
              {s.id}
            </button>
          ),
        )}
      </p>
      {notes.map((n, i) => (
        <p key={i} className="answer__caveat">{n.replace(/^Note:\s*/, "")}</p>
      ))}

      <ol className="notes" aria-label="Sources quoted in this answer">
        {result.citations.map((c) => (
          <li key={c.source_id} id={`note-${turnId}-${c.source_id}`}
              className={`note${flash === c.source_id ? " note--flash" : ""}`}>
            <span className="note__id" aria-hidden="true">{c.source_id}</span>
            <div className="note__body">
              <p className="note__where">
                {c.title}, <span className="note__section">{c.section ?? "untitled section"}</span>, {pages(c.page_start, c.page_end)}
              </p>
              <Clause text={c.snippet} sentences={bySentence.get(c.source_id) ?? []} />
            </div>
          </li>
        ))}
      </ol>

      {others.length > 0 && (
        <details className="searched">
          <summary>Other passages that were searched ({others.length})</summary>
          <ul>
            {others.map((s) => (
              <li key={s.chunk_id}>
                <span className="searched__where">
                  {s.title}, {s.section ?? "untitled section"}, {pages(s.page_start, s.page_end)}
                </span>
                {s.flags.includes("possible_injection") && (
                  <span className="flag" title="This passage contains text addressed to an AI assistant. It was not used as evidence.">
                    Contains instructions, ignored
                  </span>
                )}
              </li>
            ))}
          </ul>
        </details>
      )}

      <p className="answer__meta">
        {result.degraded
          ? "The language model was unavailable, so these are direct quotes from the policy text."
          : result.mode === "extractive"
          ? "Quoted directly from the policy text, with no language model involved."
          : "Written by the language model from the clauses above, then checked against them."}{" "}
        Took {Math.round(result.latency_ms)} ms.
      </p>
    </div>
  );
}
