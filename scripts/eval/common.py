"""Shared pieces of the skill-transplant harness (docs/SKILL_TRANSPLANT.md).

Runs in ~/evalenv (the training venv plus tokenizers/safetensors/transformers):
* ``teacher_dir``: a Hugging Face snapshot directory from the local cache (no download);
* ``TeacherTokenizer``: the teacher's byte-level BPE with the exact UTF-8 bytes of every
  token, so token spans can be placed on PseudoBrain's byte stream;
* ``corpus``/``Windows``: corpus_v4 documents through the Stage 3 mixture, cut to the
  first ``max_bytes`` bytes at a character boundary (the same text for every arm);
* ``STAGE3_MIXTURE``: the live Stage 3 run's mixture (ps args, 2026-10-02).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

STAGE3_MIXTURE = ("fineweb_edu=0.394729,cosmopedia=0.171246,books=0.0125457,wikipedia=0.152436,"
                  "tinystories=0.0383541,writingprompts=0.0372739,conversation=0.0477321,"
                  "reading_comp=0.046534,code=0.0347239,finemath=0.0231721,numina_cot=0.0232074,"
                  "openthoughts=0.00248447,orca_math=0.0122455,openr1_math=0.00331546")
HF_CACHE = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def token_rows_sha256(rows: list[list[int]]) -> str:
    """Fingerprint ordered token rows, including BOS and row boundaries, for paired evaluation."""
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode("ascii")).hexdigest()


def teacher_dir(name: str) -> Path:
    """'HuggingFaceTB/SmolLM2-135M' (cached snapshot) or a local directory."""
    path = Path(name).expanduser()
    if path.is_dir():
        return path
    repo = HF_CACHE / ("models--" + name.replace("/", "--")) / "snapshots"
    snaps = sorted(repo.iterdir()) if repo.is_dir() else []
    if not snaps:
        raise FileNotFoundError(f"{name} is not in the local Hugging Face cache ({repo})")
    return snaps[-1]


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte -> printable-unicode table used by byte-level BPE vocabularies."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class TeacherTokenizer:
    """The teacher's tokenizer (tokenizers library) with exact per-token bytes."""

    def __init__(self, directory: Path):
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(Path(directory) / "tokenizer.json"))
        config = json.loads((Path(directory) / "config.json").read_text(encoding="utf-8"))
        text = config.get("text_config") or {}
        # Document separator: BOS when the model has one, else its end-of-text token
        # (Qwen has no BOS).
        for key in ("bos_token_id", "eos_token_id"):
            value = text.get(key, config.get(key))
            if isinstance(value, list):
                value = value[0]
            if value is not None:
                self.bos = int(value)
                break
        else:
            raise ValueError("teacher config has neither bos_token_id nor eos_token_id")
        decoder = {ch: b for b, ch in _bytes_to_unicode().items()}
        vocab = self.tok.get_vocab(with_added_tokens=True)
        size = max(vocab.values()) + 1
        special = {t.content for t in self.tok.get_added_tokens_decoder().values()}
        self.token_bytes: list[bytes] = [b""] * size
        for text, index in vocab.items():
            if text in special or any(ch not in decoder for ch in text):
                self.token_bytes[index] = text.encode("utf-8")
            else:
                self.token_bytes[index] = bytes(decoder[ch] for ch in text)
        self.lengths = [len(b) for b in self.token_bytes]

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self.tok.encode_batch(texts, add_special_tokens=False)]

    def byte_lengths(self, ids: list[int]) -> list[int]:
        return [self.lengths[i] for i in ids]


def load_byte_shards():
    path = ROOT / "learning" / "byte_shards.py"
    spec = importlib.util.spec_from_file_location("transplant_byte_shards", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def utf8_prefix(data: bytes, n: int) -> str:
    return data[:n].decode("utf-8", errors="ignore")


class Windows:
    """Training texts: corpus documents (mixture sampler, seeded) cut to max_bytes."""

    def __init__(self, corpus_dir: str | Path, mixture: str | None, seed: int, max_bytes: int,
                 min_bytes: int = 64):
        bs = load_byte_shards()
        self.corpus = bs.ShardedByteCorpus(Path(corpus_dir).expanduser(), "train",
                                           mixture=bs.parse_mixture(mixture), seed=seed)
        self.max_bytes, self.min_bytes = max_bytes, min_bytes

    def next_texts(self, n: int) -> list[str]:
        return [text for text, _ in self.next_docs(n)]

    def next_docs(self, n: int) -> list[tuple[str, bool]]:
        """(text, complete): the document's first max_bytes bytes and whether that is all of it."""
        out = []
        while len(out) < n:
            data = self.corpus.doc_bytes(self.corpus.sampler.next())
            text = utf8_prefix(data, self.max_bytes)
            if len(text.encode("utf-8")) >= self.min_bytes:
                out.append((text, len(data) <= self.max_bytes))
        return out

    def state(self) -> dict:
        return self.corpus.sampler_state()

    def load(self, state: dict) -> None:
        self.corpus.load_sampler_state(state)


def parse_window(text: str) -> tuple[int, int]:
    """'W' or 'W+S' -> (attention window W, S attention-sink positions)."""
    window, _, sinks = text.partition("+")
    return int(window), int(sinks or 0)


def holdout_texts(corpus_dir: str | Path, n: int, max_bytes: int) -> list[str]:
    """Validation texts: the first n documents of the corpus holdout split, interleaved
    round-robin over every source (arch_eval --val-corpus-holdout)."""
    bs = load_byte_shards()
    corpus = bs.ShardedByteCorpus(Path(corpus_dir).expanduser(), "holdout")
    return [d["text"] for d in corpus.holdout_documents(n, max_bytes=max_bytes)]


def pack_tokens(token_rows: list[list[int]], bos: int, length: int) -> tuple[list[list[int]], int]:
    """Concatenate [bos] + tokens of each document into rows of exactly `length` tokens
    (documents split across rows; the tail that does not fill a row is dropped).
    Returns (rows, dropped tokens)."""
    stream: list[int] = []
    for row in token_rows:
        stream.append(bos)
        stream.extend(row)
    usable = len(stream) // length * length
    return [stream[i:i + length] for i in range(0, usable, length)], len(stream) - usable


def far_copy_mask(ids: list[int], n: int = 8, distance: int = 1024) -> list[bool]:
    """mask[p] for the prediction of ids[p + 1]: the n + 1 tokens ids[p - n + 1 .. p + 1] are a
    verbatim repeat whose most recent earlier occurrence ends at least `distance` positions back,
    so only exact recall from beyond a `distance`-token window can predict that token from a copy."""
    last: dict[tuple, int] = {}
    out = [False] * (len(ids) - 1)
    for e in range(n, len(ids)):  # (n+1)-gram ids[e-n .. e], predicted at position e - 1
        key = tuple(ids[e - n:e + 1])
        j = last.get(key)
        if j is not None and e - j >= distance:
            out[e - 1] = True
        last[key] = e
    return out


_TURN = re.compile(r"(?:^|\n\n)(User|Assistant): ")


def conversation_messages(text: str) -> list[dict] | None:
    """corpus_v4 'conversation' documents ("User: ...\n\nAssistant: ...") as chat messages, or
    None when the text does not start with a user turn and alternate."""
    parts = _TURN.split(text)
    msgs = [{"role": "user" if role == "User" else "assistant", "content": content.strip()}
            for role, content in zip(parts[1::2], parts[2::2])]
    if not msgs or parts[0].strip() or any(m["role"] != ("user", "assistant")[i % 2] for i, m in enumerate(msgs)):
        return None
    return msgs
