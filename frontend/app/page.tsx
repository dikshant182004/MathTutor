"use client";

import Script from "next/script";
import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";

type Thread = {
  thread_id: string;
  created_at?: number | null;
  last_active?: number | null;
  problem_summary?: string;
  topic?: string;
  outcome?: string;
};
type ChatMessage = {
  id?: string;
  role: "user" | "assistant" | "system";
  content: string;
  created_at?: number | null;
};
type PendingInterrupt = {
  thread_id: string;
  interrupt_id: string;
  hitl_type?: string;
  type?: string;
  prompt?: string;
  question?: string;
  [key: string]: unknown;
};
type DocumentItem = {
  document_id: string;
  filename: string;
  pages: number;
  chunks: number;
  uploaded_at: number;
};
type SseEvent = { event: string; data: Record<string, unknown> };

declare global {
  interface Window {
    google?: {
      accounts?: {
        id?: {
          initialize: (options: {
            client_id: string;
            callback: (response: { credential?: string }) => void;
            auto_select?: boolean;
          }) => void;
          renderButton: (
            element: HTMLElement,
            options: { theme: string; size: string; shape: string; text: string; width: number },
          ) => void;
          disableAutoSelect: () => void;
        };
      };
    };
  }
}

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const GOOGLE_CLIENT_ID = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID || "";

function parseEventBlock(block: string): SseEvent | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (!dataLines.length) return null;
  try {
    return { event, data: JSON.parse(dataLines.join("\n")) as Record<string, unknown> };
  } catch {
    return null;
  }
}

async function readSse(
  response: Response,
  onEvent: (event: SseEvent) => void,
): Promise<void> {
  if (!response.body) throw new Error("The server did not open a streaming response.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done }).replace(/\r\n/g, "\n");
    let boundary = buffer.indexOf("\n\n");
    while (boundary >= 0) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const parsed = parseEventBlock(block);
      if (parsed) onEvent(parsed);
      boundary = buffer.indexOf("\n\n");
    }
    if (done) break;
  }
  if (buffer.trim()) {
    const parsed = parseEventBlock(buffer);
    if (parsed) onEvent(parsed);
  }
}

function messageFromEvent(event: SseEvent): string {
  const value = event.data.message;
  return typeof value === "string" ? value : "The request failed. Please try again.";
}

export default function HomePage() {
  const [token, setToken] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [threads, setThreads] = useState<Thread[]>([]);
  const [selectedThread, setSelectedThread] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [authBusy, setAuthBusy] = useState(false);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [pending, setPending] = useState<PendingInterrupt | null>(null);
  const [hitlResponse, setHitlResponse] = useState("");
  const [notice, setNotice] = useState("");
  const [googleReady, setGoogleReady] = useState(false);

  const selected = useMemo(
    () => threads.find((thread) => thread.thread_id === selectedThread),
    [threads, selectedThread],
  );

  const apiFetch = useCallback(
    async (path: string, init: RequestInit = {}) => {
      const headers = new Headers(init.headers);
      if (token) headers.set("Authorization", `Bearer ${token}`);
      const response = await fetch(`${API_BASE}${path}`, { ...init, headers });
      if (!response.ok) {
        let message = `Request failed (${response.status}).`;
        try {
          const payload = (await response.json()) as { detail?: { message?: string } | string };
          if (typeof payload.detail === "string") message = payload.detail;
          else if (payload.detail?.message) message = payload.detail.message;
        } catch {
          // Keep the status-based fallback for non-JSON proxy errors.
        }
        throw new Error(message);
      }
      return response;
    },
    [token],
  );

  const loadThreads = useCallback(async () => {
    const response = await apiFetch("/threads");
    const data = (await response.json()) as { threads: Thread[] };
    setThreads(data.threads || []);
    return data.threads || [];
  }, [apiFetch]);

  const loadDocuments = useCallback(async () => {
    const response = await apiFetch("/documents");
    const data = (await response.json()) as { documents: DocumentItem[] };
    setDocuments(data.documents || []);
  }, [apiFetch]);

  const loadMessages = useCallback(
    async (threadId: string) => {
      const response = await apiFetch(`/threads/${encodeURIComponent(threadId)}/messages`);
      const data = (await response.json()) as {
        messages: ChatMessage[];
        pending_interrupt?: PendingInterrupt | null;
      };
      setMessages(data.messages || []);
      setPending(data.pending_interrupt || null);
      setSelectedThread(threadId);
    },
    [apiFetch],
  );

  useEffect(() => {
    const saved = window.sessionStorage.getItem("mathtutor_access_token");
    const name = window.sessionStorage.getItem("mathtutor_display_name") || "";
    if (saved) {
      setToken(saved);
      setDisplayName(name);
    }
  }, []);

  useEffect(() => {
    if (!token) return;
    Promise.all([loadThreads(), loadDocuments()])
      .then(([threadList]) => {
        if (!selectedThread && threadList.length) {
          void loadMessages(threadList[0].thread_id);
        }
      })
      .catch((error: unknown) => setNotice(error instanceof Error ? error.message : "Could not load your workspace."));
  }, [token, loadThreads, loadDocuments, loadMessages, selectedThread]);

  const initializeGoogle = useCallback(() => {
    const identity = window.google?.accounts?.id;
    if (!identity || !GOOGLE_CLIENT_ID) return;
    identity.initialize({
      client_id: GOOGLE_CLIENT_ID,
      callback: async (response) => {
        if (!response.credential) {
          setNotice("Google did not return a sign-in credential.");
          return;
        }
        setAuthBusy(true);
        setNotice("");
        try {
          const result = await fetch(`${API_BASE}/auth/google`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ id_token: response.credential }),
          });
          const payload = (await result.json()) as {
            access_token?: string;
            display_name?: string;
            detail?: { message?: string };
          };
          if (!result.ok || !payload.access_token) {
            throw new Error(payload.detail?.message || "Google sign-in failed.");
          }
          window.sessionStorage.setItem("mathtutor_access_token", payload.access_token);
          window.sessionStorage.setItem("mathtutor_display_name", payload.display_name || "Student");
          setDisplayName(payload.display_name || "Student");
          setToken(payload.access_token);
        } catch (error) {
          setNotice(error instanceof Error ? error.message : "Google sign-in failed.");
        } finally {
          setAuthBusy(false);
        }
      },
      auto_select: false,
    });
    const mount = document.getElementById("google-sign-in");
    if (mount) {
      mount.replaceChildren();
      identity.renderButton(mount, {
        theme: "outline",
        size: "large",
        shape: "pill",
        text: "signin_with",
        width: 260,
      });
      setGoogleReady(true);
    }
  }, []);

  function logout() {
    window.sessionStorage.removeItem("mathtutor_access_token");
    window.sessionStorage.removeItem("mathtutor_display_name");
    setToken("");
    setDisplayName("");
    setThreads([]);
    setMessages([]);
    setDocuments([]);
    setSelectedThread("");
    setPending(null);
    window.google?.accounts?.id?.disableAutoSelect();
  }

  async function createThread() {
    setBusy(true);
    setNotice("");
    try {
      const response = await apiFetch("/threads", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: "New math question" }),
      });
      const thread = (await response.json()) as Thread;
      setThreads((current) => [thread, ...current.filter((item) => item.thread_id !== thread.thread_id)]);
      setMessages([]);
      setPending(null);
      setSelectedThread(thread.thread_id);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not create a conversation.");
    } finally {
      setBusy(false);
    }
  }

  async function consumeRun(path: string, body: Record<string, unknown>, idempotencyKey?: string) {
    const headers = new Headers({ "Content-Type": "application/json" });
    if (token) headers.set("Authorization", `Bearer ${token}`);
    if (idempotencyKey) headers.set("Idempotency-Key", idempotencyKey);
    const response = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      let message = `Request failed (${response.status}).`;
      try {
        const payload = (await response.json()) as { detail?: { message?: string } | string };
        if (typeof payload.detail === "string") message = payload.detail;
        else if (payload.detail?.message) message = payload.detail.message;
      } catch {
        // Keep fallback.
      }
      throw new Error(message);
    }
    let assistantText = "";
    let interrupt: PendingInterrupt | null = null;
    await readSse(response, (event) => {
      if (event.event === "token" && typeof event.data.text === "string") {
        assistantText += event.data.text;
        setMessages((current) => {
          const next = [...current];
          const last = next[next.length - 1];
          if (last?.role === "assistant") next[next.length - 1] = { ...last, content: assistantText };
          else next.push({ role: "assistant", content: assistantText });
          return next;
        });
      } else if (event.event === "solution" && typeof event.data.text === "string") {
        assistantText = event.data.text;
        setMessages((current) => {
          const next = [...current];
          const last = next[next.length - 1];
          if (last?.role === "assistant") next[next.length - 1] = { ...last, content: assistantText };
          else next.push({ role: "assistant", content: assistantText });
          return next;
        });
      } else if (event.event === "interrupt") {
        interrupt = event.data as PendingInterrupt;
        setPending(interrupt);
        setHitlResponse("");
      } else if (event.event === "error") {
        setNotice(messageFromEvent(event));
      }
    });
    setPending(interrupt);
    return assistantText;
  }

  async function submitMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const content = draft.trim();
    if (!content || !selectedThread || busy || pending) return;
    setDraft("");
    setNotice("");
    setBusy(true);
    setMessages((current) => [...current, { role: "user", content }, { role: "assistant", content: "" }]);
    try {
      await consumeRun(`/threads/${encodeURIComponent(selectedThread)}/messages`, { content });
      await loadThreads();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The message could not be sent.");
      setMessages((current) => current.filter((message) => !(message.role === "assistant" && !message.content)));
    } finally {
      setBusy(false);
    }
  }

  async function resumeThread(decision: "submit" | "retry" | "correct" | "incorrect") {
    if (!selectedThread || !pending || busy) return;
    setBusy(true);
    setNotice("");
    try {
      const response: Record<string, unknown> =
        (pending.hitl_type || pending.type) === "verification"
          ? { is_correct: decision === "correct", fix_hint: hitlResponse.trim() }
          : { answer: hitlResponse.trim(), corrected_text: hitlResponse.trim(), raw_text: hitlResponse.trim() };
      await consumeRun(
        `/threads/${encodeURIComponent(selectedThread)}/resume`,
        { interrupt_id: pending.interrupt_id, decision, response },
        crypto.randomUUID(),
      );
      await loadMessages(selectedThread);
      await loadThreads();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not resume this conversation.");
    } finally {
      setBusy(false);
    }
  }

  async function uploadPdf(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    const input = form.elements.namedItem("pdf") as HTMLInputElement | null;
    const file = input?.files?.[0];
    if (!file) return;
    setUploadBusy(true);
    setNotice("");
    const body = new FormData();
    body.append("file", file);
    try {
      await apiFetch("/documents", { method: "POST", body });
      form.reset();
      await loadDocuments();
      setNotice("Your notes are indexed and will be available across conversations.");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "PDF upload failed.");
    } finally {
      setUploadBusy(false);
    }
  }

  async function removePdf(documentId: string) {
    if (!window.confirm("Delete this document and its indexed chunks?")) return;
    try {
      await apiFetch(`/documents/${encodeURIComponent(documentId)}`, { method: "DELETE" });
      await loadDocuments();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not delete the document.");
    }
  }

  if (!token) {
    return (
      <main className="auth-shell">
        <Script
          src="https://accounts.google.com/gsi/client"
          strategy="afterInteractive"
          onLoad={initializeGoogle}
        />
        <section className="auth-card">
          <div className="brand-mark">∑</div>
          <p className="eyebrow">PERSONAL AI MATH COACH</p>
          <h1>MathTutor</h1>
          <p className="muted auth-copy">
            Work through tough problems step by step. Keep your notes, conversations, and learning history in one private workspace.
          </p>
          {GOOGLE_CLIENT_ID ? (
            <>
              <div id="google-sign-in" className="google-button" />
              {!googleReady && <p className="muted small">Preparing secure sign-in…</p>}
            </>
          ) : (
            <div className="configuration-note">
              Set <code>NEXT_PUBLIC_GOOGLE_CLIENT_ID</code> in <code>frontend/.env.local</code> to enable Google sign-in.
            </div>
          )}
          {authBusy && <p className="muted">Signing you in…</p>}
          {notice && <p className="notice error">{notice}</p>}
          <p className="small muted auth-footnote">Your API access token stays in this browser tab session.</p>
        </section>
      </main>
    );
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand-row">
          <div className="brand-mark small-mark">∑</div>
          <div><strong>MathTutor</strong><span className="muted tiny">LEARN WITH REASONING</span></div>
        </div>
        <button className="primary-button full-width" onClick={() => void createThread()} disabled={busy}>
          <span>＋</span> New conversation
        </button>
        <div className="sidebar-section-title">RECENT CONVERSATIONS</div>
        <div className="thread-list">
          {threads.map((thread) => (
            <button
              key={thread.thread_id}
              className={`thread-item ${selectedThread === thread.thread_id ? "selected" : ""}`}
              onClick={() => void loadMessages(thread.thread_id).catch((error: unknown) => setNotice(error instanceof Error ? error.message : "Could not open conversation."))}
            >
              <span className="thread-icon">◷</span>
              <span className="thread-copy">
                <strong>{thread.problem_summary || "Math conversation"}</strong>
                <small>{thread.topic || thread.outcome || "Open conversation"}</small>
              </span>
            </button>
          ))}
          {!threads.length && <p className="muted small sidebar-empty">Your conversations will appear here.</p>}
        </div>
        <div className="sidebar-footer">
          <div className="avatar">{(displayName || "S").slice(0, 1).toUpperCase()}</div>
          <div className="user-label"><strong>{displayName || "Student"}</strong><span className="muted tiny">Personal workspace</span></div>
          <button className="icon-button" aria-label="Sign out" title="Sign out" onClick={logout}>↪</button>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">YOUR LEARNING SPACE</p>
            <h2>{selected?.problem_summary || "Let's solve something."}</h2>
          </div>
          <div className="status-pill"><span className="status-dot" /> Private session</div>
        </header>

        {notice && <div className="notice" role="status">{notice}<button onClick={() => setNotice("")} aria-label="Dismiss message">×</button></div>}

        <div className="workspace-grid">
          <section className="chat-panel">
            {!selectedThread ? (
              <div className="empty-state">
                <div className="empty-symbol">∫</div>
                <h3>Make a difficult problem feel manageable.</h3>
                <p>Start a conversation, ask for a hint, or work through a full solution.</p>
                <button className="primary-button" onClick={() => void createThread()} disabled={busy}>Start learning <span>→</span></button>
              </div>
            ) : (
              <>
                <div className="message-list">
                  {!messages.length && (
                    <div className="welcome-note">
                      <div className="assistant-avatar">∑</div>
                      <div><strong>Ready when you are.</strong><p>Send a question. I’ll explain the reasoning, not just the answer.</p></div>
                    </div>
                  )}
                  {messages.map((message, index) => (
                    <article className={`message-row ${message.role === "user" ? "user-message" : "assistant-message"}`} key={message.id || `${message.role}-${index}`}>
                      {message.role !== "user" && <div className="assistant-avatar">∑</div>}
                      <div className="message-bubble">
                        <div className="message-label">{message.role === "user" ? "YOU" : "MATHTUTOR"}</div>
                        <div className="message-content">{message.content || (busy ? <span className="typing"><i /><i /><i /></span> : "No response was returned.")}</div>
                      </div>
                    </article>
                  ))}
                  {pending && (
                    <section className="hitl-card">
                      <div className="hitl-heading"><span>✦</span><div><strong>Let’s clarify this together</strong><p>{String(pending.prompt || pending.question || "I need one more detail before continuing.")}</p></div></div>
                      {(pending.hitl_type || pending.type) === "verification" ? (
                        <div className="hitl-actions">
                          <button className="secondary-button" disabled={busy} onClick={() => void resumeThread("correct")}>The solution looks correct</button>
                          <button className="primary-button" disabled={busy} onClick={() => void resumeThread("incorrect")}>Needs correction</button>
                        </div>
                      ) : (
                        <form onSubmit={(event) => { event.preventDefault(); void resumeThread("submit"); }} className="hitl-form">
                          <textarea value={hitlResponse} onChange={(event) => setHitlResponse(event.target.value)} placeholder="Add the missing detail or correction…" required rows={2} />
                          <button className="primary-button" type="submit" disabled={busy || !hitlResponse.trim()}>{busy ? "Continuing…" : "Continue solution"} <span>→</span></button>
                        </form>
                      )}
                    </section>
                  )}
                </div>
                <form className="composer" onSubmit={(event) => void submitMessage(event)}>
                  <textarea
                    value={draft}
                    onChange={(event) => setDraft(event.target.value)}
                    placeholder={pending ? "Respond to the question above to continue…" : "Ask a math question, request a hint, or paste a problem…"}
                    rows={2}
                    disabled={busy || !!pending}
                  />
                  <div className="composer-footer">
                    <span className="muted tiny">Shift + Enter for a new line · AI can make mistakes</span>
                    <button className="primary-button send-button" type="submit" disabled={busy || !!pending || !draft.trim()}>
                      {busy ? "Working…" : "Send"} <span>↑</span>
                    </button>
                  </div>
                </form>
              </>
            )}
          </section>

          <aside className="notes-panel">
            <div className="panel-heading"><div><p className="eyebrow">YOUR MATERIALS</p><h3>Study notes</h3></div><span className="count-badge">{documents.length}</span></div>
            <p className="muted small">Upload PDFs once. Your notes remain available across your conversations.</p>
            <form className="upload-box" onSubmit={(event) => void uploadPdf(event)}>
              <label htmlFor="pdf-upload" className="upload-icon">↑</label>
              <strong>Add your notes</strong>
              <span className="muted tiny">PDF only · up to 10 MB</span>
              <input id="pdf-upload" name="pdf" type="file" accept="application/pdf,.pdf" required />
              <button className="secondary-button full-width" type="submit" disabled={uploadBusy}>{uploadBusy ? "Indexing notes…" : "Upload PDF"}</button>
            </form>
            <div className="document-list">
              {documents.map((document) => (
                <article className="document-item" key={document.document_id}>
                  <div className="pdf-icon">PDF</div>
                  <div className="document-copy"><strong title={document.filename}>{document.filename}</strong><span className="muted tiny">{document.pages} pages · {document.chunks} chunks</span></div>
                  <button className="icon-button delete-button" title="Delete document" aria-label={`Delete ${document.filename}`} onClick={() => void removePdf(document.document_id)}>×</button>
                </article>
              ))}
              {!documents.length && <p className="muted small empty-docs">No notes yet. Upload a PDF to ground explanations in your material.</p>}
            </div>
            <div className="privacy-note"><span>◈</span><p><strong>Your notes stay yours.</strong><br />Documents are scoped to your account and are not visible to other students.</p></div>
          </aside>
        </div>
      </section>
    </main>
  );
}
