import { useRef, useState } from "react";
import { ApiError, api, type Health, type PolicyDoc } from "../api";

const TYPE_NAMES: Record<string, string> = { homeowners: "Homeowners", auto: "Auto", pet: "Pet health", general: "Other" };
export const typeName = (t: string) => TYPE_NAMES[t] ?? t.charAt(0).toUpperCase() + t.slice(1);

interface Props {
  docs: PolicyDoc[];
  health: Health | null;
  filter: string | null;
  onFilter: (t: string | null) => void;
  onUploaded: () => void;
  apiKey: string;
  onApiKey: (k: string) => void;
}

export function Library({ docs, health, filter, onFilter, onUploaded, apiKey, onApiKey }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [upload, setUpload] = useState<{ state: "idle" | "busy" | "done" | "error"; message?: string }>({ state: "idle" });
  const [needsKey, setNeedsKey] = useState(false);
  const showKey = needsKey || Boolean(health?.upload_requires_key);
  const types = [...new Set(docs.map((d) => d.policy_type))].sort();

  const onFile = async (file: File | undefined) => {
    if (!file) return;
    setUpload({ state: "busy", message: `Reading ${file.name}…` });
    try {
      const r = await api.upload(file, apiKey || undefined);
      setUpload({
        state: "done",
        message: r.status === "duplicate"
          ? `${r.title} is already in the library.`
          : `Added ${r.title}: ${r.num_pages} pages, ${r.num_chunks} passages.` +
            (r.flagged_chunks ? ` ${r.flagged_chunks} passage(s) contain instructions and will be ignored as evidence.` : ""),
      });
      setNeedsKey(false);
      onUploaded();
    } catch (e) {
      const err = e as ApiError;
      if (err.status === 401) setNeedsKey(true);
      setUpload({ state: "error", message: err.message });
    } finally {
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  return (
    <aside className="library" aria-label="Policy library">
      <div className="brand">
        <span className="brand__mark" aria-hidden="true" />
        <span className="brand__name">Policy Q&amp;A</span>
      </div>

      <p className={`service service--${health?.database === false ? "down" : "up"}`} role="status">
        {health === null ? "Connecting…" : health.database
          ? `Searching ${docs.length} ${docs.length === 1 ? "policy" : "policies"}`
          : "Policy database unavailable"}
      </p>

      <fieldset className="scope">
        <legend>Ask about</legend>
        <label className={filter === null ? "is-on" : ""}>
          <input type="radio" name="scope" checked={filter === null} onChange={() => onFilter(null)} />
          Any policy
        </label>
        {types.map((t) => (
          <label key={t} className={filter === t ? "is-on" : ""}>
            <input type="radio" name="scope" checked={filter === t} onChange={() => onFilter(t)} />
            {typeName(t)}
          </label>
        ))}
      </fieldset>

      <ul className="docs">
        {docs.map((d) => (
          <li key={d.doc_id} className={filter && filter !== d.policy_type ? "is-dim" : ""}>
            <span className="docs__title">{d.title}</span>
            <span className="docs__meta">{typeName(d.policy_type)}, {d.num_pages} {d.num_pages === 1 ? "page" : "pages"}</span>
          </li>
        ))}
      </ul>

      {health?.uploads_enabled && (
        <div className="upload">
          <input ref={fileRef} id="pdf" type="file" accept="application/pdf,.pdf" hidden
                 onChange={(e) => onFile(e.target.files?.[0])} />
          {showKey && (
            <label className="upload__key">
              API key required to add policies
              <input type="password" value={apiKey} autoComplete="off" onChange={(e) => onApiKey(e.target.value)} />
            </label>
          )}
          <button type="button" className="btn btn--quiet" disabled={upload.state === "busy"}
                  onClick={() => fileRef.current?.click()}>
            {upload.state === "busy" ? "Adding policy…" : "Add a policy PDF"}
          </button>
          {upload.message && <p className={`upload__msg upload__msg--${upload.state}`} role="status">{upload.message}</p>}
        </div>
      )}
    </aside>
  );
}
