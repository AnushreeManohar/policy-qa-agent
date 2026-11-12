// Typed client for the FastAPI backend. Every call surfaces the server's error message and request id.

export type Status = "answered" | "partial" | "insufficient_information" | "refused";

export interface Citation {
  source_id: number;
  chunk_id: string;
  doc_id: string;
  filename: string;
  title: string;
  page_start: number;
  page_end: number;
  section: string | null;
  snippet: string;
}

export interface Source {
  id: number;
  chunk_id: string;
  title: string;
  filename: string;
  section: string | null;
  page_start: number;
  page_end: number;
  score: number;
  flags: string[];
  cited: boolean;
  snippet: string;
}

export interface AskResponse {
  question: string;
  answer: string;
  status: Status;
  citations: Citation[];
  sources: Source[];
  policy_type: string | null;
  coverage: number | null;
  missing_terms: string[];
  faithfulness: number | null;
  mode: string;
  degraded: boolean;
  latency_ms: number;
  retrieval_ms: number | null;
}

export interface PolicyDoc {
  doc_id: string;
  filename: string;
  title: string;
  policy_type: string;
  num_pages: number;
  num_chunks: number;
}

export interface Health {
  status: "ok" | "degraded";
  database: boolean;
  llm_mode: string;
  embedding_model: string;
  documents: number | null;
  uploads_enabled: boolean;
  upload_requires_key: boolean;
  auth_required_for_ask: boolean;
}

export interface IngestResult {
  status: "ingested" | "duplicate";
  title: string;
  num_pages: number;
  num_chunks: number;
  flagged_chunks: number;
}

export class ApiError extends Error {
  constructor(message: string, public status: number, public requestId?: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    throw new ApiError("Can't reach the server. Check that the API is running, then try again.", 0);
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const fallback = res.status === 429 ? "Too many questions in a minute. Wait a moment and ask again." : `Request failed (${res.status}).`;
    throw new ApiError(body?.error ?? fallback, res.status, body?.request_id);
  }
  return body as T;
}

const keyHeader = (apiKey?: string): Record<string, string> => (apiKey ? { "X-API-Key": apiKey } : {});

export const api = {
  health: () => request<Health>("/api/health"),
  documents: () => request<PolicyDoc[]>("/api/documents"),
  ask: (question: string, policyType: string | null, apiKey?: string) =>
    request<AskResponse>("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...keyHeader(apiKey) },
      body: JSON.stringify({ question, policy_type: policyType ?? undefined }),
    }),
  upload: (file: File, apiKey?: string) => {
    const form = new FormData();
    form.append("file", file);
    return request<IngestResult>("/api/documents", { method: "POST", body: form, headers: keyHeader(apiKey) });
  },
};
