"use client";

import { useMemo, useState } from "react";

type Message = { role: "user" | "assistant"; content: string };

export default function HomePage() {
  const [documentText, setDocumentText] = useState("");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", content: "Paste your document text below, ask a question, and the assistant will answer based on the content." }
  ]);

  const summary = useMemo(() => {
    if (!documentText.trim()) return "No document loaded yet.";
    return `${documentText.trim().split(/\s+/).length} words loaded`;
  }, [documentText]);

  const askQuestion = async () => {
    if (!question.trim() || !documentText.trim()) return;
    const next = [...messages, { role: "user", content: question } as Message];
    setMessages(next);
    const answer = `Demo answer: Based on the uploaded content, the best response to "${question}" should reference the provided document. Replace this with a real OpenAI API call in production.`;
    setMessages([...next, { role: "assistant", content: answer }]);
    setQuestion("");
  };

  return (
    <main className="min-h-screen bg-slate-950 px-6 py-10">
      <div className="mx-auto grid max-w-6xl gap-6 lg:grid-cols-2">
        <section className="rounded-3xl border border-slate-800 bg-slate-900 p-6 shadow-2xl">
          <p className="text-sm uppercase tracking-[0.3em] text-cyan-400">Portfolio Project</p>
          <h1 className="mt-2 text-4xl font-bold">AI Document Assistant</h1>
          <p className="mt-3 text-slate-300">A simple starter app for document Q&amp;A. Users paste or upload content, then ask natural-language questions.</p>
          <div className="mt-6 rounded-2xl bg-slate-950 p-4">
            <p className="text-sm text-slate-400">Document status</p>
            <p className="mt-1 text-lg font-semibold">{summary}</p>
          </div>
          <textarea className="mt-6 h-72 w-full rounded-2xl border border-slate-700 bg-slate-950 p-4 outline-none" placeholder="Paste document text here..." value={documentText} onChange={(e) => setDocumentText(e.target.value)} />
        </section>
        <section className="rounded-3xl border border-slate-800 bg-slate-900 p-6 shadow-2xl">
          <div className="space-y-4 rounded-2xl bg-slate-950 p-4">
            {messages.map((message, index) => (
              <div key={index} className="rounded-2xl border border-slate-800 p-4">
                <p className="text-xs uppercase tracking-[0.25em] text-cyan-400">{message.role}</p>
                <p className="mt-2 leading-7 text-slate-200">{message.content}</p>
              </div>
            ))}
          </div>
          <div className="mt-6 flex flex-col gap-3">
            <input className="w-full rounded-2xl border border-slate-700 bg-slate-950 px-4 py-3 outline-none" placeholder="Ask a question about the document" value={question} onChange={(e) => setQuestion(e.target.value)} />
            <button onClick={askQuestion} className="rounded-2xl bg-cyan-500 px-5 py-3 font-semibold text-slate-950">Ask Question</button>
          </div>
        </section>
      </div>
    </main>
  );
}
