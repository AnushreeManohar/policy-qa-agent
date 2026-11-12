import { useState } from "react";
import { highlight } from "../citations";

const SENTENCE_END = /[.;:](?=\s+[A-Z(])/g;

/** Index just after the sentence boundary `n` sentences before/after `pos` (bounded by the text). */
function boundary(text: string, pos: number, dir: -1 | 1, n: number): number {
  const ends = [...text.matchAll(SENTENCE_END)].map((m) => (m.index ?? 0) + 1);
  if (dir < 0) {
    const before = ends.filter((e) => e <= pos);
    return before.length > n ? before[before.length - 1 - n] : 0;
  }
  const after = ends.filter((e) => e >= pos);
  return after.length > n ? after[n] : text.length;
}

/** A cited passage in the policy's own typeface. Shows the highlighted sentence with one sentence of
 *  context on each side; the full passage is one click away. */
export function Clause({ text, sentences }: { text: string; sentences: string[] }) {
  const [full, setFull] = useState(false);
  const parts = highlight(text, sentences);
  const flat = parts.map((p) => p.text).join("");
  let start = 0;
  let end = flat.length;
  const firstMark = parts.findIndex((p) => p.marked);
  if (firstMark >= 0 && !full) {
    const offset = (i: number) => parts.slice(0, i).reduce((n, p) => n + p.text.length, 0);
    const lastMark = parts.length - 1 - [...parts].reverse().findIndex((p) => p.marked);
    start = boundary(flat, offset(firstMark), -1, 1);
    end = boundary(flat, offset(lastMark) + parts[lastMark].text.length, 1, 1);
  }
  const trimmed = start > 0 || end < flat.length;

  // Re-slice the highlighted parts to the visible window.
  const visible: { text: string; marked: boolean }[] = [];
  let pos = 0;
  for (const p of parts) {
    const a = Math.max(pos, start);
    const b = Math.min(pos + p.text.length, end);
    if (b > a) visible.push({ text: p.text.slice(a - pos, b - pos), marked: p.marked });
    pos += p.text.length;
  }

  return (
    <>
      <blockquote className={`clause${full || firstMark < 0 ? " clause--full" : ""}`}>
        {start > 0 && "… "}
        {visible.map((p, i) => (p.marked ? <mark key={i}>{p.text.trim()}</mark> : <span key={i}>{p.text}</span>))}
        {end < flat.length && " …"}
      </blockquote>
      {(trimmed || full) && firstMark >= 0 && (
        <button type="button" className="clause__toggle" onClick={() => setFull((f) => !f)} aria-expanded={full}>
          {full ? "Show only the quoted part" : "Show the whole passage"}
        </button>
      )}
    </>
  );
}
