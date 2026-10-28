"""The `recall` (look-back) tool: exact passages from the session's own verbatim history.

Owner design (2026-10-02): exact long-range recall comes from a TOOL, not from memorizing in
the recurrent state. The model "generally knows where it is, all it has to do is look for it,
then say it again". The runtime keeps the full verbatim history of the session OUTSIDE the
model (conversation turns, documents it read, tool results: append-only, RAM/disk, no model
memory); the tool returns exact passages with their positions, and the model copies from the
result, which arrives as an ordinary tool message (a cheap prefill).

    history = SessionHistory()
    history.append("user", long_document)
    history.recall("pass key")            # {"ok", "query", "results": [{"n", "pos", "entry", "role", "text"}]}
    history.recall("locker", near="Alice") # results nearest the first occurrence of "Alice" rank first

Ranking: exact (case-insensitive) occurrences of the query first, then the passages with the
most query-word hits (BM25 over overlapping windows), optionally reordered by distance to a
`near` landmark (a string: its first occurrence; or a character position). Each result is a
window of `span` characters around the hit, expanded to word boundaries; positions are character
offsets into the concatenated history, so the model can ask again near a place it was told
about. Results are numbered from `start` and fit `max_bytes` of compact JSON (the tool-message
cap). Standard library only.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field

MAX_QUERY_CHARS = 128
_WORD = re.compile(r"[A-Za-z0-9_']+")
SCHEMA = {
    "name": "recall",
    "description": ("Look back in this conversation's own history (messages, documents read, tool "
                    "results) and return the exact passages that match the query, with their positions. "
                    "Use it to quote or copy something from earlier instead of relying on memory."),
    "parameters": {"type": "object",
                   "properties": {"query": {"type": "string", "maxLength": MAX_QUERY_CHARS},
                                  "near": {"type": "string", "maxLength": MAX_QUERY_CHARS}},
                   "required": ["query"], "additionalProperties": False},
}


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


@dataclass
class Entry:
    role: str
    start: int
    text: str


@dataclass
class SessionHistory:
    """Append-only verbatim history of one session."""
    entries: list = field(default_factory=list)
    length: int = 0

    def append(self, role: str, text: str) -> tuple[int, int]:
        """Record a message / document / tool result; returns its (start, end) positions."""
        start = self.length
        self.entries.append(Entry(role, start, text))
        self.length += len(text) + 1  # a separator between entries
        return start, start + len(text)

    def text(self) -> str:
        return "\n".join(e.text for e in self.entries)

    def _locate(self, pos: int) -> int:
        lo, hi = 0, len(self.entries) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.entries[mid].start <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _window(self, full: str, a: int, b: int, span: int) -> tuple[int, int]:
        pad = max(0, (span - (b - a)) // 2)
        lo, hi = max(0, a - pad), min(len(full), b + pad)
        while lo > 0 and full[lo - 1].isalnum():
            lo -= 1
        while hi < len(full) and full[hi].isalnum():
            hi += 1
        return lo, hi

    def recall(self, query: str, near: str | int | None = None, k: int = 3, span: int = 400,
               start: int = 1, max_bytes: int = 4096) -> dict:
        query = (query or "").strip()
        if not query or len(query) > MAX_QUERY_CHARS:
            return {"ok": False, "error": "invalid_query", "query": query[:MAX_QUERY_CHARS]}
        full = self.text()
        if not full:
            return {"ok": False, "error": "not_found", "query": query}
        low = full.lower()
        anchor = None
        if isinstance(near, int):
            anchor = near
        elif isinstance(near, str) and near.strip():
            at = low.find(near.strip().lower())
            anchor = at if at >= 0 else None
        hits: list[tuple[float, int, int]] = []  # (score, start, end)
        q = query.lower()
        at = low.find(q)
        while at >= 0:
            hits.append((1e6, at, at + len(q)))
            at = low.find(q, at + 1)
        if len(hits) < k:
            hits += self._bm25(full, low, query, span)
        if not hits:
            return {"ok": False, "error": "not_found", "query": query}
        if anchor is not None:
            hits = [(s - abs(a - anchor) * 1e-3, a, b) for s, a, b in hits]
        hits.sort(key=lambda h: (-h[0], h[1]))
        chosen: list[tuple[int, int]] = []
        for _, a, b in hits:
            lo, hi = self._window(full, a, b, span)
            if any(lo < c_hi and c_lo < hi for c_lo, c_hi in chosen):
                continue
            chosen.append((lo, hi))
            if len(chosen) >= k:
                break
        results = []
        for n, (lo, hi) in enumerate(chosen):
            e = self.entries[self._locate(lo)]
            results.append({"n": start + n, "pos": lo, "entry": self.entries.index(e), "role": e.role,
                            "text": full[lo:hi]})
        out = {"ok": True, "query": query, "results": results}
        while len(json.dumps(out, separators=(",", ":"), ensure_ascii=False).encode()) > max_bytes and results:
            last = results[-1]
            if len(last["text"]) > 80:
                last["text"] = last["text"][:int(len(last["text"]) * 0.8)]
            else:
                results.pop()
        return out

    def _bm25(self, full: str, low: str, query: str, span: int) -> list[tuple[float, int, int]]:
        terms = set(_words(query))
        if not terms:
            return []
        stride = max(50, span // 2)
        windows = [(a, min(len(full), a + span)) for a in range(0, max(1, len(full) - span // 2), stride)]
        docs = [Counter(_words(low[a:b])) for a, b in windows]
        n = len(docs)
        avg = sum(sum(d.values()) for d in docs) / max(1, n)
        df = {t: sum(1 for d in docs if t in d) for t in terms}
        scored = []
        for (a, b), d in zip(windows, docs):
            length = sum(d.values())
            s = 0.0
            for t in terms:
                f = d.get(t, 0)
                if f:
                    idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                    s += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * length / max(1.0, avg)))
            if s > 0:
                scored.append((s, a, b))
        scored.sort(key=lambda x: -x[0])
        return scored[:8]
