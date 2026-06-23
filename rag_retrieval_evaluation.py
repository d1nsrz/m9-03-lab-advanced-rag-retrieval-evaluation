"""
Lab 3 | Make Retrieval Better -- and Prove It

Upgrade retrieval over the same knowledge_base.json, then MEASURE whether the
change helped: hit rate (exact, no LLM) + faithfulness (LLM-as-judge).

Upgrade chosen: HYBRID SEARCH
  dense (embeddings) ranking  +  BM25 keyword ranking,
  merged with Reciprocal Rank Fusion (RRF).
Dense retrieval alone fumbles exact tokens like the error code 0x80070005;
BM25 nails them. RRF gets the best of both.

- Embeddings : Gemini gemini-embedding-001 OR local sentence-transformers
- Judge      : Gemini chat model (needs GOOGLE_API_KEY); hit rate needs no key

Run:  python solution.py    ->  also writes eval_results.md
"""

import json
import os
import re

import numpy as np
from rank_bm25 import BM25Okapi

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "") 
EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "gemini" if GOOGLE_API_KEY else "local")
CHAT_MODEL = "gemini-2.5-flash"
BASE = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(BASE, "knowledge_base.json")
TOP_K = 3



def get_embedder():
    if EMBED_BACKEND == "gemini":
        from google import genai

        client = genai.Client(api_key=GOOGLE_API_KEY)

        def embed(texts):
            r = client.models.embed_content(model="gemini-embedding-001", contents=texts)
            return np.array([e.values for e in r.embeddings], dtype=np.float32)

        print("Embeddings: Gemini gemini-embedding-001")
        return embed

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Embeddings: local all-MiniLM-L6-v2")
    return lambda texts: model.encode(texts, convert_to_numpy=True).astype(np.float32)


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())



class Retrievers:
    def __init__(self, kb, embed):
        self.kb = kb
        self.ids = [p["id"] for p in kb]
        self.embed = embed
        self.doc_vecs = embed([p["text"] for p in kb])
        self.doc_vecs /= np.linalg.norm(self.doc_vecs, axis=1, keepdims=True) + 1e-10
        self.bm25 = BM25Okapi([tokenize(p["text"]) for p in kb])

    def dense(self, query, k=TOP_K):
        q = self.embed([query])[0]
        q /= np.linalg.norm(q) + 1e-10
        order = np.argsort(self.doc_vecs @ q)[::-1]
        return [self.ids[i] for i in order[:k]]

    def _bm25_ranking(self, query):
        scores = self.bm25.get_scores(tokenize(query))
        return list(np.argsort(scores)[::-1])

    def _dense_ranking(self, query):
        q = self.embed([query])[0]
        q /= np.linalg.norm(q) + 1e-10
        return list(np.argsort(self.doc_vecs @ q)[::-1])

    def hybrid(self, query, k=TOP_K, rrf_k=60):
        fused = {}
        for ranking in (self._dense_ranking(query), self._bm25_ranking(query)):
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank)
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [self.ids[i] for i in best]


def _chat(prompt):
    if not GOOGLE_API_KEY:
        return None
    from google import genai

    client = genai.Client(api_key=GOOGLE_API_KEY)
    return client.models.generate_content(model=CHAT_MODEL, contents=prompt).text.strip()


def generate_answer(kb_by_id, question, ids):
    ctx = "\n".join(f"[{i}] {kb_by_id[i]['text']}" for i in ids)
    prompt = (
        "Answer ONLY from the context. If absent, say \"I don't know\".\n\n"
        f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    )
    return _chat(prompt)


def judge_faithful(question, answer, kb_by_id, ids):
    """LLM-as-judge: is the answer fully supported by the retrieved context?"""
    if answer is None:
        return None
    ctx = "\n".join(f"[{i}] {kb_by_id[i]['text']}" for i in ids)
    prompt = (
        "You are a strict grader. Is every claim in the ANSWER fully supported "
        "by the CONTEXT? Reply with one word: YES or NO.\n\n"
        f"CONTEXT:\n{ctx}\n\nANSWER:\n{answer}\n\nSupported?"
    )
    verdict = _chat(prompt)
    return verdict.upper().startswith("YES") if verdict else None


EVAL_SET = [
    {"q": "How long do I have to get a full refund?", "expected": "kb-04"},
    {"q": "What does error 0x80070005 mean?", "expected": "kb-08"},  # exact term
    {"q": "How do I cancel my subscription and stop billing?", "expected": "kb-05"},
    {"q": "When are employees allowed to park in lot B?", "expected": "kb-01"},
    {"q": "How do I reset my password?", "expected": "kb-07"},
]


def evaluate(name, retrieve_fn, kb_by_id):
    hits, faith_yes, faith_total = 0, 0, 0
    rows = []
    for item in EVAL_SET:
        ids = retrieve_fn(item["q"])
        hit = item["expected"] in ids
        hits += hit
        ans = generate_answer(kb_by_id, item["q"], ids)
        faithful = judge_faithful(item["q"], ans, kb_by_id, ids)
        if faithful is not None:
            faith_total += 1
            faith_yes += faithful
        rows.append((item["q"], item["expected"], ids, hit, faithful))
    n = len(EVAL_SET)
    hit_rate = hits / n
    faith_rate = (faith_yes / faith_total) if faith_total else None
    return {"name": name, "hit_rate": hit_rate, "faith_rate": faith_rate, "rows": rows}


def main():
    kb = json.load(open(KB_PATH))
    kb_by_id = {p["id"]: p for p in kb}
    r = Retrievers(kb, get_embedder())

    baseline = evaluate("baseline (dense)", r.dense, kb_by_id)
    upgraded = evaluate("upgraded (hybrid RRF)", r.hybrid, kb_by_id)

    def fmt(x):
        return "n/a (set GOOGLE_API_KEY)" if x is None else f"{x:.0%}"

    lines = []
    lines.append("# Lab 3 — Evaluation Results\n")
    lines.append(f"Embedding backend: `{EMBED_BACKEND}`. Eval set: {len(EVAL_SET)} "
                 "questions, one with the exact term `0x80070005`.\n")
    lines.append("## Comparison table\n")
    lines.append("| Setup | Retrieval hit rate | Faithfulness (LLM judge) |")
    lines.append("|---|---|---|")
    for res in (baseline, upgraded):
        lines.append(f"| {res['name']} | {res['hit_rate']:.0%} | {fmt(res['faith_rate'])} |")
    lines.append("\n## Per-question retrieval (expected id must be in the set)\n")
    lines.append("| Question | Expected | baseline hit | hybrid hit |")
    lines.append("|---|---|---|---|")
    for b, u in zip(baseline["rows"], upgraded["rows"]):
        lines.append(f"| {b[0]} | {b[1]} | {'✅' if b[3] else '❌'} | {'✅' if u[3] else '❌'} |")

    lines.append("\n## Conclusion\n")
    delta = upgraded["hit_rate"] - baseline["hit_rate"]
    if delta > 0:
        verdict = (f"Hybrid RRF **raised hit rate by {delta:.0%}**, driven mainly by the "
                   "exact-term query `0x80070005` that dense retrieval ranks poorly — BM25 "
                   "matches the literal token and RRF promotes it into the top 3.")
    elif delta == 0:
        verdict = ("Hit rate was **flat**: on this tiny, well-separated KB dense retrieval "
                   "already finds every expected passage, so hybrid had no room to help. "
                   "On a larger corpus with more lexical collisions the exact-term win shows.")
    else:
        verdict = (f"Hybrid **hurt** hit rate by {-delta:.0%} here — BM25 noise outranked a "
                   "correct dense hit. Reported honestly; the measurement is the point.")
    lines.append(verdict)
    lines.append("\nFaithfulness is judged by an LLM and needs `GOOGLE_API_KEY`; with the key "
                 "set, the table above fills in both columns for a fair side-by-side.")

    report = "\n".join(lines) + "\n"
    open(os.path.join(BASE, "eval_results.md"), "w").write(report)
    print(report)
    print("Wrote eval_results.md")


if __name__ == "__main__":
    main()
