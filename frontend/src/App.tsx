import { useEffect, useRef, useState } from "react";
import type { AskResponse, ChatMessage, Stats } from "./types";

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "bot",
      text:
        "Hi! I can answer questions grounded in the venue's documents. Try: \"How long do refunds take if an event is cancelled?\" or \"Is parking available?\"",
    },
  ]);
  const [query, setQuery] = useState("");
  const [alpha, setAlpha] = useState(50);
  const [stats, setStats] = useState<Stats | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  const refreshStats = async () => {
    try {
      setStats(await fetch("/api/stats").then((r) => r.json() as Promise<Stats>));
    } catch {
      /* ignore */
    }
  };

  useEffect(() => {
    refreshStats();
  }, []);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const ask = async (e: React.FormEvent) => {
    e.preventDefault();
    const q = query.trim();
    if (!q) return;
    setMessages((m) => [...m, { role: "user", text: q }]);
    setQuery("");
    const res = (await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q, alpha: alpha / 100 }),
    }).then((r) => r.json())) as AskResponse;
    setMessages((m) => [
      ...m,
      { role: "bot", text: res.answer, citations: res.citations, cached: res.cached, abstained: res.abstained },
    ]);
    refreshStats();
  };

  return (
    <div className="max-w-4xl mx-auto px-6 py-8">
      <header className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-violet-400 to-fuchsia-500 grid place-items-center font-bold text-slate-900">
            DS
          </div>
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">DocSage</h1>
            <p className="text-sm text-slate-300/70">
              Grounded document Q&A · hybrid search · citations · never hallucinates
            </p>
          </div>
        </div>
        <div className="text-xs text-slate-400 text-right font-mono">
          {stats && (
            <>
              {stats.documents} docs · {stats.chunks} chunks · cache {stats.cache_hits} hits
              <br />
              {stats.embedder} + {stats.generator}
            </>
          )}
        </div>
      </header>

      <div className="flex items-center gap-3 mb-4 text-sm">
        <span className="text-slate-400">Retrieval:</span>
        <input
          type="range"
          min={0}
          max={100}
          value={alpha}
          onChange={(e) => setAlpha(Number(e.target.value))}
          className="w-40 accent-violet-500"
        />
        <span className="text-slate-300 font-mono w-44">
          lexical {100 - alpha}% / semantic {alpha}%
        </span>
      </div>

      <div className="space-y-4 mb-4 min-h-[300px]">
        {messages.map((m, i) => (
          <div key={i} className={`msg-enter ${m.role === "user" ? "text-right" : ""}`}>
            {m.role === "user" ? (
              <div className="inline-block bg-violet-500/20 border border-violet-500/30 rounded-2xl rounded-br-sm px-4 py-2 text-sm">
                {m.text}
              </div>
            ) : (
              <div className="inline-block bg-slate-900/70 border border-slate-800 rounded-2xl rounded-bl-sm px-4 py-3 text-sm max-w-2xl text-left">
                <div className={m.abstained ? "text-amber-300" : "text-slate-100"}>
                  {m.text}
                  {m.cached && (
                    <span className="ml-2 text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300">
                      cached
                    </span>
                  )}
                </div>
                {m.citations && m.citations.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-slate-800 space-y-2">
                    <div className="text-[11px] uppercase tracking-wide text-slate-500">Sources</div>
                    {m.citations.map((c, j) => (
                      <div key={j} className="text-xs text-slate-400">
                        <span className="text-violet-300 font-mono">
                          [{j + 1}] {c.doc_id}
                        </span>
                        <span className="text-slate-500"> · score {c.score}</span>
                        <div className="text-slate-500 italic mt-0.5">"{c.snippet}…"</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
        <div ref={endRef} />
      </div>

      <form onSubmit={ask} className="sticky bottom-6">
        <div className="flex gap-2 bg-slate-900/80 border border-slate-700 rounded-2xl p-2 backdrop-blur">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask about the refund policy, parking, accessibility…"
            className="flex-1 bg-transparent px-3 py-2 text-sm focus:outline-none"
          />
          <button className="bg-violet-500 hover:bg-violet-400 text-slate-900 font-medium rounded-xl px-5 py-2 text-sm">
            Ask
          </button>
        </div>
      </form>
    </div>
  );
}
