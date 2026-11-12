import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api, type AskResponse, type Health, type PolicyDoc } from "./api";
import { Answer } from "./components/Answer";
import { Library, typeName } from "./components/Library";

type Turn =
  | { id: number; question: string; scope: string | null; state: "loading" }
  | { id: number; question: string; scope: string | null; state: "done"; result: AskResponse }
  | { id: number; question: string; scope: string | null; state: "error"; error: string };

const EXAMPLES = [
  "Is earthquake damage covered under the homeowners policy?",
  "What is the rental reimbursement limit and is driving in Mexico covered?",
  "How long is the waiting period for hip dysplasia?",
  "Does the homeowners policy cover cryptocurrency theft?",
];
const MAX_CHARS = 1000;

export default function App() {
  const [docs, setDocs] = useState<PolicyDoc[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [filter, setFilter] = useState<string | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [draft, setDraft] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const nextId = useRef(1);
  const endRef = useRef<HTMLDivElement>(null);
  const busy = turns.some((t) => t.state === "loading");

  const refresh = useCallback(async () => {
    try {
      const [h, d] = await Promise.all([api.health(), api.documents().catch(() => [])]);
      setHealth(h);
      setDocs(d);
    } catch {
      setHealth({ status: "degraded", database: false, llm_mode: "?", embedding_model: "?", documents: null,
                  uploads_enabled: false, auth_required_for_ask: false });
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [turns]);

  const ask = async (question: string, scope: string | null, replaceId?: number) => {
    const q = question.trim();
    if (q.length < 3 || busy) return;
    const id = replaceId ?? nextId.current++;
    const pending: Turn = { id, question: q, scope, state: "loading" };
    setTurns((ts) => (replaceId ? ts.map((t) => (t.id === id ? pending : t)) : [...ts, pending]));
    setDraft("");
    try {
      const result = await api.ask(q, scope, apiKey || undefined);
      setTurns((ts) => ts.map((t) => (t.id === id ? { ...pending, state: "done", result } : t)));
    } catch (e) {
      const err = e as ApiError;
      setTurns((ts) => ts.map((t) => (t.id === id ? { ...pending, state: "error", error: err.message } : t)));
    }
  };

  const onKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void ask(draft, filter);
    }
  };

  return (
    <div className="shell">
      <Library docs={docs} health={health} filter={filter} onFilter={setFilter} onUploaded={refresh}
               apiKey={apiKey} onApiKey={setApiKey} />

      <main className="desk">
        <div className="thread" aria-live="polite">
          {turns.length === 0 ? (
            <section className="intro">
              <h1>Ask the policy, see the clause.</h1>
              <p>
                Every answer is taken from the policy documents in the library and shows the exact passage it came
                from, with the page. When the documents don't say, you'll be told that instead of getting a guess.
              </p>
              <ul className="examples" aria-label="Example questions">
                {EXAMPLES.map((q) => (
                  <li key={q}>
                    <button type="button" onClick={() => void ask(q, null)} disabled={busy || !health?.database}>{q}</button>
                  </li>
                ))}
              </ul>
            </section>
          ) : (
            turns.map((t) => (
              <article key={t.id} className="turn">
                <h2 className="turn__q">
                  {t.question}
                  {t.scope && <span className="turn__scope">in {typeName(t.scope)}</span>}
                </h2>
                {t.state === "loading" && (
                  <p className="pending" role="status"><span className="pending__bar" />Searching the policies</p>
                )}
                {t.state === "error" && (
                  <div className="answer answer--error" role="alert">
                    <p>{t.error}</p>
                    <button type="button" className="btn btn--quiet" onClick={() => void ask(t.question, t.scope, t.id)}>
                      Ask again
                    </button>
                  </div>
                )}
                {t.state === "done" && <Answer turnId={t.id} result={t.result} />}
              </article>
            ))
          )}
          <div ref={endRef} />
        </div>

        <form className="composer" onSubmit={(e) => { e.preventDefault(); void ask(draft, filter); }}>
          <label htmlFor="question" className="sr-only">Your question</label>
          <textarea id="question" rows={2} value={draft} maxLength={MAX_CHARS} onKeyDown={onKey}
                    onChange={(e) => setDraft(e.target.value)}
                    placeholder={filter ? `Ask about the ${typeName(filter).toLowerCase()} policy` : "Ask about deductibles, exclusions, limits, claims…"} />
          <div className="composer__row">
            <span className="composer__hint">
              {draft.length > MAX_CHARS * 0.8 ? `${MAX_CHARS - draft.length} characters left` : "Enter to ask, Shift+Enter for a new line"}
            </span>
            <button type="submit" className="btn" disabled={busy || draft.trim().length < 3}>
              {busy ? "Searching…" : "Ask"}
            </button>
          </div>
        </form>
      </main>
    </div>
  );
}
