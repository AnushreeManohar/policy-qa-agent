// Helpers that connect answer text to the clauses it cites.

export type Segment = { kind: "text"; text: string } | { kind: "cite"; id: number };

/** Split "Sentence one [1] Sentence two [2][3]" into text and citation segments. */
export function segment(answer: string): Segment[] {
  const out: Segment[] = [];
  const re = /\[(\d+)\]/g;
  let last = 0;
  for (let m = re.exec(answer); m; m = re.exec(answer)) {
    if (m.index > last) out.push({ kind: "text", text: answer.slice(last, m.index) });
    out.push({ kind: "cite", id: Number(m[1]) });
    last = m.index + m[0].length;
  }
  if (last < answer.length) out.push({ kind: "text", text: answer.slice(last) });
  return out;
}

/** Sentences of the answer attributed to each citation id (the text right before its marker). */
export function sentencesByCitation(answer: string): Map<number, string[]> {
  const map = new Map<number, string[]>();
  const segs = segment(answer);
  segs.forEach((s, i) => {
    if (s.kind !== "cite") return;
    let j = i - 1;
    while (j >= 0 && segs[j].kind === "cite") j--;
    const prev = segs[j];
    if (prev?.kind === "text" && prev.text.trim()) {
      map.set(s.id, [...(map.get(s.id) ?? []), prev.text.trim()]);
    }
  });
  return map;
}

const norm = (s: string) => s.replace(/\s+/g, " ");

/** Split a clause into plain/highlighted parts where it contains one of the cited sentences verbatim. */
export function highlight(clause: string, sentences: string[]): { text: string; marked: boolean }[] {
  const text = norm(clause);
  const ranges: [number, number][] = [];
  for (const s of sentences) {
    const needle = norm(s).replace(/[.;:,]$/, "");
    if (needle.length < 12) continue;
    const at = text.indexOf(needle);
    if (at >= 0) ranges.push([at, at + needle.length]);
  }
  ranges.sort((a, b) => a[0] - b[0]);
  const parts: { text: string; marked: boolean }[] = [];
  let pos = 0;
  for (const [a, b] of ranges) {
    if (a < pos) continue;
    if (a > pos) parts.push({ text: text.slice(pos, a), marked: false });
    parts.push({ text: text.slice(a, b), marked: true });
    pos = b;
  }
  if (pos < text.length) parts.push({ text: text.slice(pos), marked: false });
  return parts;
}

export function pages(start: number, end: number): string {
  return start === end ? `page ${start}` : `pages ${start}–${end}`;
}
