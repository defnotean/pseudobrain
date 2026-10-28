"""Token-level transplant: a teacher (Llama family or a Qwen3.5/3.8 hybrid) -> a PseudoBrain core.

    PYTHONPATH=~/pylibs/fla ~/evalenv/bin/python -m scripts.transplant.convert_token \\
        --teacher Qwen/Qwen3.5-0.8B-Base --mixer recurrent --out ~/x2/runs/q08_rec \\
        --stage1-tokens 30e6 --stage2-tokens 60e6

The teacher runs as an exact TokenCore (efficiency/transplant.py load_hf_core: the
teacher's own layers in PB modules; Qwen3.5's Gated DeltaNet layers ARE PB layers). The
student starts as a copy whose --mixer layers are replaced:
  recurrent   every attention layer -> a PB Gated DeltaNet mixer initialized from it (a
              hybrid teacher's own Gated DeltaNet layers are copied exactly): fully
              recurrent, constant state, no KV cache;
  mingru      every attention layer -> a PB minGRU mixer (Llama teachers);
  hybrid:<k>  Llama teachers: recurrent, but every k-th layer keeps its attention;
  swa:<W>     every attention layer kept, as a sliding window of W tokens (bounded cache,
              no conversion training needed; --stage1-tokens 0).
Only the converted layers' mixers train in stage 1.

Stage 1 (attention transfer; MOHAWK stage 2 / RADLADS step 1 / LoLCATs): each converted
mixer reads the teacher's own residual input of its layer (teacher forcing, all converted
layers in parallel) and is fit to that layer's attention output: mean over layers of
||mixer(x) - attn(x)||^2 / ||attn(x)||^2. Memory: the frozen teacher plus the new mixers.

Stage 2 (end-to-end distillation): the student runs on its own outputs; loss = KL(teacher
|| student) over the full vocabulary at --kd-fraction of the positions (both heads applied
chunk by chunk, so a 248k vocabulary never materializes [B,T,V]). --stage2-train:
'converted' (the new mixers; the scalable default), 'mixers' (every mixer), 'all'.

Data: corpus_v4 through the Stage 3 mixture, packed into --seq-token rows with the
teacher's BOS/end-of-text between documents. Validation: per-token KL and NLL of student
and teacher on holdout rows. Output: core.pt ({"config", "state_dict", "history"}),
log.jsonl, summary.json.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .common import (STAGE3_MIXTURE, TeacherTokenizer, Windows, holdout_texts, log, pack_tokens, parse_window,
                     teacher_dir)
from efficiency import transplant as tp
from efficiency.lora import packs_bypassed


def student_spec(text: str, c: tp.TeacherConfig) -> tp.CoreSpec:
    exact = tp.CoreSpec.exact(c)
    if text == "exact":
        return exact
    if text == "recurrent":
        return tp.CoreSpec.uniform("gated_deltanet", c.layers)
    if text == "mingru":
        return tp.CoreSpec(["mingru" if m == "attention" else m for m in exact.mixers])
    if text.startswith("hybrid:"):
        return tp.CoreSpec.hybrid(c.layers, int(text.split(":")[1]))
    if text.startswith("keep:"):  # fully recurrent except these attention layers (chosen by sensitivity)
        keep = {int(i) for i in text.split(":")[1].split(",")}
        if any(exact.mixers[i] != "attention" for i in keep):
            raise ValueError(f"keep:{sorted(keep)} must name teacher attention layers")
        return tp.CoreSpec(["attention" if i in keep else "gated_deltanet" for i in range(c.layers)])
    if text.startswith("swa:"):  # swa:W or swa:W+S (S attention-sink positions)
        window, sinks = parse_window(text.split(":")[1])
        return tp.CoreSpec(exact.mixers, attention_window=window, attention_sinks=sinks)
    if text.startswith("swa_gdn:"):  # windowed teacher attention + a GDN for beyond the window
        window, sinks = parse_window(text.split(":")[1])
        return tp.CoreSpec(["swa_gdn" if m == "attention" else m for m in exact.mixers],
                           attention_window=window, attention_sinks=sinks)
    raise ValueError(f"unknown --mixer {text!r}")


class JoinedChunks:
    """Long rows from CONSECUTIVE chunks of one corpus source joined by a blank line (corpus_v4
    stores each book as consecutive ~60 KB chunks, so the rows are continuous long text; a chunk
    boundary may start a new book, as in evaluate.py longnll32). Only ~6% of single book chunks
    reach 16k tokens; joined rows reach any length. Same state()/load() surface as Windows."""

    def __init__(self, corpus: str, tok: TeacherTokenizer, split: str = "train", source: str = "books",
                 seed: int = 0, start_fraction: float = 0.0):
        import random
        from .common import load_byte_shards
        self.corpus = load_byte_shards().ShardedByteCorpus(Path(corpus).expanduser(), split)
        lo, hi = self.corpus.source_range(source)
        self.start, self.stop = lo + int((hi - lo) * start_fraction), hi
        self.tok = tok
        self.rng = random.Random(seed)

    def row_from(self, i: int, seq: int) -> tuple[list[int] | None, int]:
        """(BOS + seq - 1 tokens joined from chunk i on, or None at the source's end; next index)."""
        ids, j = [], i
        while len(ids) < seq - 1 and j < self.stop:
            ids.extend(self.tok.encode_batch([("\n\n" if j > i else "") + self.corpus.text(j)])[0])
            j += 1
        return ([self.tok.bos] + ids[:seq - 1] if len(ids) >= seq - 1 else None), j

    def next_row(self, seq: int) -> list[int]:
        while True:
            row, _ = self.row_from(self.rng.randrange(self.start, self.stop), seq)
            if row is not None:
                return row

    def state(self) -> dict:
        return {"rng": self.rng.getstate()}

    def load(self, state: dict) -> None:
        self.rng.setstate(state["rng"])


class TokenBatches:
    """Rows of `seq` tokens: documents packed BOS-separated, and optionally a fraction of each
    batch from single long documents (BOS + the document's first seq-1 tokens), so mixers that
    replace attention see dependencies across the whole row (long_windows/long_fraction), and a
    fraction of packed conversations rendered with the teacher's chat template (chat_windows,
    render: text -> rendered string or None; post-trained teachers run in that format)."""

    def __init__(self, windows: Windows, tok: TeacherTokenizer, batch: int, seq: int,
                 long_windows: Windows | None = None, long_fraction: float = 0.0,
                 chat_windows: Windows | None = None, chat_fraction: float = 0.0, render=None,
                 long_single: Windows | None = None, join_fraction: float = 1.0):
        self.windows, self.tok, self.batch, self.seq = windows, tok, batch, seq
        self.long_windows = long_windows
        # with joined long rows (JoinedChunks): a share 1 - join_fraction of the long rows are single long
        # documents of long_single instead (e.g. reasoning traces and code next to joined books)
        self.long_single, self.join_fraction = long_single, join_fraction
        # Expected rows per batch. A shared random offset below gives stochastic rounding
        # without clipping one category when another fills a small batch.
        self.long_mean = batch * long_fraction if long_windows is not None else 0.0
        self.long_rows = int(round(self.long_mean))
        self.chat_windows, self.render = chat_windows, render
        self.chat_mean = batch * chat_fraction if chat_windows is not None else 0.0
        self.chat_rows = int(round(self.chat_mean))
        if self.long_mean + self.chat_mean > batch + 1e-9:
            raise ValueError("long + chat rows exceed the batch")
        self.buffer: list[list[int]] = []
        self.long_buffer: list[list[int]] = []
        self.chat_buffer: list[list[int]] = []
        self.dropped = 0
        self.chat_skipped = 0
        import random
        self.rng = random.Random(1234)

    def _chat(self, n: int) -> list[list[int]]:
        # Encoded conversations accumulate until they fill a row: 32 conversations are far fewer than
        # seq tokens at seq 65536, and packing each group alone dropped every token and never ended.
        pending, pending_tokens = [], 0
        while len(self.chat_buffer) < n:
            texts = []
            for text in self.chat_windows.next_texts(32):
                rendered = self.render(text)
                if rendered is None:
                    self.chat_skipped += 1
                else:
                    texts.append(rendered)
            for ids in self.tok.encode_batch(texts):
                pending.append(ids)
                pending_tokens += len(ids) + 1
            if pending_tokens < self.seq:
                continue
            rows, dropped = pack_tokens(pending, self.tok.bos, self.seq)
            pending, pending_tokens = [], 0
            self.dropped += dropped
            self.chat_buffer.extend(rows)
        rows, self.chat_buffer = self.chat_buffer[:n], self.chat_buffer[n:]
        return rows

    def _long(self, n: int) -> list[list[int]]:
        rows = []
        for _ in range(n):
            if isinstance(self.long_windows, JoinedChunks) and (
                    self.long_single is None or self.rng.random() < self.join_fraction):
                rows.append(self.long_windows.next_row(self.seq))
            else:
                rows.append(self._single_long())
        return rows

    def _single_long(self) -> list[int]:
        """BOS + the first seq-1 tokens of the next single document of at least seq tokens."""
        source = self.long_single if self.long_single is not None else self.long_windows
        while not self.long_buffer:
            for ids in self.tok.encode_batch(source.next_texts(8)):
                if len(ids) + 1 >= self.seq:
                    self.long_buffer.append([self.tok.bos] + ids[:self.seq - 1])
        return self.long_buffer.pop(0)

    def next(self) -> torch.Tensor:
        # Systematic categorical sampling: each row has the requested marginal mix,
        # including batch=1. Independent rounded counts followed by clipping made
        # 75% long + 15% chat become 75% long + 3.75% chat for batch=1.
        offset = self.rng.random()
        long_cut = self.long_mean
        chat_cut = long_cut + self.chat_mean
        n_long = n_chat = 0
        for i in range(self.batch):
            point = offset + i
            n_long += point < long_cut
            n_chat += long_cut <= point < chat_cut
        packed = self.batch - n_long - n_chat
        pending, pending_tokens = [], 0  # as in _chat: accumulate until a row is full
        while len(self.buffer) < packed:
            for ids in self.tok.encode_batch(self.windows.next_texts(64)):
                pending.append(ids)
                pending_tokens += len(ids) + 1
            if pending_tokens < self.seq:
                continue
            rows, dropped = pack_tokens(pending, self.tok.bos, self.seq)
            pending, pending_tokens = [], 0
            self.dropped += dropped
            self.buffer.extend(rows)
        rows, self.buffer = self.buffer[:packed], self.buffer[packed:]
        if n_long:
            rows = rows + self._long(n_long)
        if n_chat:
            rows = rows + self._chat(n_chat)
        return torch.tensor(rows, dtype=torch.long)


class FileBatches:
    """Pre-computed batches (dump_batches.py): a uint32 array [batches, batch, seq] read in order, so a run
    on another machine (Colab) sees exactly the token stream the corpus samplers produced on the Spark,
    without a corpus replica. `cursor` is the number of batches consumed; resume sets it from the step.
    A run longer than the file wraps around (a logged new pass), so stage 2 does not stop when stage 1 and
    stage 2 together need more batches than the file holds."""

    def __init__(self, path, batch: int, seq: int):
        import numpy as np
        self.data = np.load(path, mmap_mode="r")
        if self.data.shape[1:] != (batch, seq):
            raise ValueError(f"batch file {path} has shape {self.data.shape}, run wants (*, {batch}, {seq})")
        self.cursor, self.dropped, self.chat_skipped = 0, 0, 0

    def next(self) -> torch.Tensor:
        n = self.data.shape[0]
        if self.cursor and self.cursor % n == 0:
            log(f"batch file pass {self.cursor // n + 1}: wrapping after {self.cursor} batches")
        rows = torch.from_numpy(self.data[self.cursor % n].astype("int64"))
        self.cursor += 1
        return rows


def make_optimizer(params, lr: float, kind: str):
    """AdamW (fp32 states) or bitsandbytes 8-bit AdamW (2 bytes of state per parameter instead of 8: the 16
    converted layers of a 27B are ~1.85B trainable parameters)."""
    if kind == "adamw8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    if kind == "paged_adamw8bit":  # states page out to host memory under GPU memory pressure
        import bitsandbytes as bnb
        return bnb.optim.PagedAdamW8bit(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)


def stage2_parameters(student: tp.TokenCore, mode: str, converted: list[int]) -> list[torch.nn.Parameter]:
    """Set requires_grad for stage 2 and return the trainable parameters: 'converted' = the
    converted layers' mixers (empty when nothing was converted, e.g. --mixer exact or swa),
    'mixers' = every mixer, 'all' = everything."""
    if mode not in ("converted", "mixers", "all"):
        raise ValueError(f"unknown stage-2 parameter set {mode!r}")
    for p in student.parameters():
        p.requires_grad_(mode == "all")
    if mode == "converted":
        for p in student.mixer_parameters(converted):
            p.requires_grad_(True)
    elif mode == "mixers":
        for p in student.mixer_parameters():
            p.requires_grad_(True)
    return [p for p in student.parameters() if p.requires_grad]


def _state_leaves(state) -> list:
    """The tensors of a recurrent streaming state (lists of tensors per block), in a fixed order."""
    out = []
    for st in state:
        if not isinstance(st, (list, tuple)):
            raise TypeError("--full-bptt supports recurrent states only (no attention caches)")
        out.extend(t for t in st if t is not None)
    return out


def reverse_bptt_row(student, teacher, ids, ft_row, pos, args) -> torch.Tensor:
    """Exact gradient of the row's KD loss through the recurrent state across every chunk, at one
    chunk's activation memory: a no-grad pass stores each chunk's entry state, then the chunks are
    recomputed last to first, each backpropagating its own loss plus the gradient its exit state
    received from the later chunks; its entry-state gradient goes on to the chunk before it.
    The same function and gradient as the full-row forward/backward (up to recompute numerics)."""
    starts = list(range(0, args.seq, args.tbptt_chunk))
    entry, state = [], None
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for c0 in starts[:-1]:
            _, state = student.forward_state(ids[:, c0:c0 + args.tbptt_chunk], state)
            entry.append(state)
    entry = [None] + entry
    loss, carried = torch.zeros((), device=ids.device), None
    for c0, st in zip(reversed(starts), reversed(entry)):
        c1 = min(args.seq, c0 + args.tbptt_chunk)
        leaves = None
        if st is not None:
            st = [[None if t is None else t.detach().requires_grad_(True) for t in s] for s in st]
            leaves = _state_leaves(st)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fs_chunk, out_state = student.forward_state(ids[:, c0:c1], st)
        rows, cols = ((pos >= c0) & (pos < c1)).nonzero(as_tuple=True)
        outs, grads = [], []
        if len(rows):
            where = pos[rows, cols]
            part = tp.fused_head_kl(fs_chunk[rows, where - c0], ft_row[rows, where],
                                    student.head_weight().to(torch.bfloat16), teacher.head_weight(),
                                    chunk=args.kd_chunk, compiled=args.compile, temperature=args.kd_temperature,
                                    reverse=args.kd_reverse) * (len(rows) / pos.numel())
            outs.append(part)
            grads.append(None)
            loss = loss + part.detach()
        if carried is not None:
            for t, g in zip(_state_leaves(out_state), carried):
                if g is not None and t.requires_grad:
                    outs.append(t)
                    grads.append(g)
        if outs:
            torch.autograd.backward(outs, grads)
        carried = None if leaves is None else [t.grad for t in leaves]
        del fs_chunk, out_state, outs, grads, st
    return loss


def recall_sft_step(student_final, student, items, tok, rng, args):
    """Cross-entropy on one assistant turn (a recall call or the exact answer) of one verified
    recall trace (scripts/transplant/recall_data.py), prompted exactly as the runtime prompts it
    (turn_examples). An over-long prompt keeps its head (system prompt, tool schema) and its tail
    (the question and the tool turns); the needle itself comes back through the tool result."""
    from .recall_data import bounded_no_tool_example, sample_tool_example
    item = items[rng.randrange(len(items))]
    if getattr(args, "recall_no_tool", 0.0) and rng.random() < args.recall_no_tool:
        windows = [student.spec.window(i) for i, mixer in enumerate(student.spec.mixers)
                   if mixer in ("swa_gdn", "attention") and student.spec.window(i)]
        limit = min(args.recall_max_tokens, min(windows) if windows else args.recall_max_tokens)
        prompt, completion, p_ids, c_ids = bounded_no_tool_example(
            tok, item, rng, args.recall_no_tool_chars, limit,
            natural=getattr(args, "recall_reply", "bare") == "natural")
    else:
        natural = getattr(args, "recall_reply", "bare") == "natural"
        prompt, completion = sample_tool_example(
            tok, item, rng, natural=natural,
            history_dropout=getattr(args, "recall_history_dropout", 0.0))
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(completion, add_special_tokens=False)["input_ids"]
    budget = args.recall_max_tokens - len(c_ids)
    if len(p_ids) > budget:
        head = min(1024, budget // 4)
        p_ids = p_ids[:head] + p_ids[len(p_ids) - (budget - head):]
    ids = p_ids + c_ids
    target = torch.full((len(ids),), -100, dtype=torch.long)
    target[len(p_ids) - 1:len(ids) - 1] = torch.tensor(c_ids)  # position i-1 predicts token i
    x = torch.tensor([ids], device="cuda")
    chunk = getattr(args, "tbptt_chunk", 0)
    if getattr(args, "full_bptt", False) and chunk and len(ids) > chunk:  # long trace: exact chunked BPTT
        return reverse_bptt_ce(student, x, target.to("cuda"), chunk, args.recall_weight)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        final = student(x, logits=False)["final"][0]
    mask = target.to("cuda") >= 0
    logits = F.linear(final[mask].to(torch.bfloat16), student.head_weight().to(torch.bfloat16)).float()
    return F.cross_entropy(logits, target.to("cuda")[mask])


def thread_sft_step(student_final, student, items, tok, rng, args):
    """Cross-entropy on the final answer of one long working conversation (scripts/transplant/thread_data.py):
    a question about the thread (aim, current approach, owners, deadline, a running summary) after events
    scattered among pasted text. The whole prompt is kept (the events are in the middle), so items over
    --thread-max-tokens are resampled; long ones backpropagate through the state with chunked exact BPTT."""
    from .thread_data import examples
    for _ in range(16):
        prompt, completion = rng.choice(examples(tok, items[rng.randrange(len(items))]))
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(completion, add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(c_ids) <= args.thread_max_tokens:
            break
    else:
        return None
    ids = p_ids + c_ids
    target = torch.full((len(ids),), -100, dtype=torch.long)
    target[len(p_ids) - 1:len(ids) - 1] = torch.tensor(c_ids)
    x = torch.tensor([ids], device="cuda")
    chunk = getattr(args, "tbptt_chunk", 0)
    if getattr(args, "full_bptt", False) and chunk and len(ids) > chunk:
        return reverse_bptt_ce(student, x, target.to("cuda"), chunk, args.thread_weight)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        final = student(x, logits=False)["final"][0]
    mask = target.to("cuda") >= 0
    logits = F.linear(final[mask].to(torch.bfloat16), student.head_weight().to(torch.bfloat16)).float()
    return F.cross_entropy(logits, target.to("cuda")[mask])


def copy_sft_step(student, rows, tok, rng, args, watch=False):
    """Cross-entropy on the verbatim key of one long exact-copy item (scripts/transplant/copy_data.py), most of
    them asked from beyond the attention window. The context streams through the bounded caches without grad in
    --copy-prefill-chunk pieces (with --exact-memory the evicted blocks go to the archive, as in serving); the last
    --copy-tail context tokens plus the answer run as one chunk with grad, so retrieval is driven by the question
    and the readout learns to use retrieved far blocks and the carried GDN state."""
    from .copy_data import make_item
    length = rng.randrange(args.copy_min_tokens, args.copy_max_tokens + 1)
    far = args.copy_far if rng.random() < args.copy_far_frac else 0
    ctx, ans, _ = make_item(tok, rows, length, rng, min_far=far)
    x = torch.tensor([ctx + ans], device="cuda")
    cut = max(0, len(ctx) - args.copy_tail)
    state = None
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, cut, args.copy_prefill_chunk):
            _, state = student.forward_state(x[:, s:min(cut, s + args.copy_prefill_chunk)], state)
            if watch and (s // args.copy_prefill_chunk) % 32 == 31:
                log_memory(f"copy prefill at {s + args.copy_prefill_chunk}/{cut}")
    if state is not None:
        state = student.detach_state(state)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fs, _ = student.forward_state(x[:, cut:], state)
    first = len(ctx) - 1 - cut  # position len(ctx)-1 predicts the first answer token
    logits = F.linear(fs[0, first:first + len(ans)].to(torch.bfloat16),
                      student.head_weight().to(torch.bfloat16)).float()
    return F.cross_entropy(logits, torch.tensor(ans, device="cuda"))


def reverse_bptt_ce(student, x, target, chunk, weight) -> torch.Tensor:
    """reverse_bptt_row for a cross-entropy target (-100 = ignored) on one row of any length: backpropagates
    weight x the mean CE through the recurrent state across every chunk at one chunk's activation memory, and
    returns the detached loss (the caller must not call backward on it)."""
    n = x.shape[1]
    total = int((target >= 0).sum())
    starts = list(range(0, n, chunk))
    entry, state = [], None
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for c0 in starts[:-1]:
            _, state = student.forward_state(x[:, c0:c0 + chunk], state)
            entry.append(state)
    entry = [None] + entry
    loss, carried = torch.zeros((), device=x.device), None
    for c0, st in zip(reversed(starts), reversed(entry)):
        c1 = min(n, c0 + chunk)
        leaves = None
        if st is not None:
            st = [[None if t is None else t.detach().requires_grad_(True) for t in s] for s in st]
            leaves = _state_leaves(st)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fs_chunk, out_state = student.forward_state(x[:, c0:c1], st)
        tgt = target[c0:c1]
        mask = tgt >= 0
        outs, grads = [], []
        if mask.any():
            logits = F.linear(fs_chunk[0][mask].to(torch.bfloat16), student.head_weight().to(torch.bfloat16)).float()
            part = F.cross_entropy(logits, tgt[mask], reduction="sum") / total
            outs.append(part * weight)
            grads.append(None)
            loss = loss + part.detach()
        if carried is not None:
            for t, g in zip(_state_leaves(out_state), carried):
                if g is not None and t.requires_grad:
                    outs.append(t)
                    grads.append(g)
        if outs:
            torch.autograd.backward(outs, grads)
        carried = None if leaves is None else [t.grad for t in leaves]
        del fs_chunk, out_state, outs, grads, st
    return loss


def mc_sft_step(student_final, student, items, tok, rng, args):
    """Cross-entropy on the answer letter of `--mc-batch` multiple-choice items (scripts/transplant/mc_data.py),
    scored the way lm-eval scores MMLU: the letter token (with its leading space) after "Answer:"."""
    losses = []
    for _ in range(args.mc_batch):
        item = items[rng.randrange(len(items))]
        p_ids = tok(item["prompt"], add_special_tokens=False)["input_ids"][-args.mc_max_tokens:]
        c_id = tok(" " + item["label"], add_special_tokens=False)["input_ids"][-1]
        x = torch.tensor([p_ids], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            final = student(x, logits=False)["final"][0, -1:]
        logits = F.linear(final.to(torch.bfloat16), student.head_weight().to(torch.bfloat16)).float()
        losses.append(F.cross_entropy(logits, torch.tensor([c_id], device="cuda")))
    return torch.stack(losses).mean()


def mckl_step(student, teacher, items, tok, rng, args, converted, base):
    """Full-vocabulary distillation on `--mckl-batch` multiple-choice documents (scripts/transplant/mc_kl_data.py):
    KL(teacher || student) at every position, plus `--mckl-answer-weight` x the KL at the positions that predict an
    answer letter (after "Answer:"), plus optionally `--mckl-align` x the converted layers' normalized mixer MSE
    against the teacher's attention output (hidden-state matching). Soft targets only: the student learns the
    teacher's distribution over the letters, not the dataset's labels. Returns (kl_all, kl_answer, align)."""
    import re
    w_s, w_t = student.head_weight().to(torch.bfloat16), teacher.head_weight()
    sums = [torch.zeros((), device="cuda") for _ in range(3)]
    for _ in range(args.mckl_batch):
        text = items[rng.randrange(len(items))]["text"]
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"][:args.mckl_max_tokens], enc["offset_mapping"][:args.mckl_max_tokens]
        letters = {m.end() - 1 for m in re.finditer(r"Answer: [ABCD]", text)}
        answer = [j - 1 for j, (a, b) in enumerate(offs) if j and any(a <= c < b for c in letters)]
        x = torch.tensor([ids], device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), base():
            taps = teacher(x, collect=converted if args.mckl_align else False, logits=False)
        ft = taps["final"][0, :-1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fs = student(x, logits=False)["final"][0, :-1]
        kl_all = tp.fused_head_kl(fs, ft, w_s, w_t, chunk=args.kd_chunk, compiled=args.compile,
                                  temperature=args.kd_temperature, reverse=args.kd_reverse)
        loss = kl_all
        kl_ans = torch.zeros((), device="cuda")
        if answer:
            where = torch.tensor(answer, device="cuda")
            kl_ans = tp.fused_head_kl(fs[where], ft[where], w_s, w_t, chunk=args.kd_chunk, compiled=args.compile,
                                      temperature=args.kd_temperature, reverse=args.kd_reverse)
            loss = loss + args.mckl_answer_weight * kl_ans
        align = torch.zeros((), device="cuda")
        if args.mckl_align:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                align = torch.stack([tp.normalized_mse(tp.mixer_delta(student.blocks[i], taps["inputs"][i].float()),
                                                       taps["mix"][i]) for i in converted]).mean()
            loss = loss + args.mckl_align * align
        (args.mckl_weight * loss / args.mckl_batch).backward()
        for k, v in enumerate((kl_all, kl_ans, align)):
            sums[k] = sums[k] + v.detach() / args.mckl_batch
        del taps, ft, fs, loss
    return sums


@torch.no_grad()
def mc_validate(student, items, tok, args) -> dict:
    """Held-out multiple-choice accuracy: the argmax over the four answer-letter logits, per item."""
    ids = [tok(" " + c, add_special_tokens=False)["input_ids"][-1] for c in "ABCD"]
    hit = {}
    for item in items:
        p_ids = tok(item["prompt"], add_special_tokens=False)["input_ids"][-args.mc_max_tokens:]
        x = torch.tensor([p_ids], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            final = student(x, logits=False)["final"][0, -1:]
        logits = F.linear(final.to(torch.bfloat16), student.head_weight().to(torch.bfloat16)).float()[0, ids]
        hit.setdefault(item["task"], []).append(int("ABCD"[int(logits.argmax())] == item["label"]))
    out = {f"mc_val_{t}": sum(v) / len(v) for t, v in hit.items()}
    out["mc_val_acc"] = sum(sum(v) for v in hit.values()) / sum(len(v) for v in hit.values())
    return out


def log_memory(tag: str) -> None:
    """CUDA allocator state and the host's MemAvailable (on the GB10's unified memory the GPU's
    reserved blocks come out of it; the Spark's memory guard kills jobs below its floor)."""
    avail = float("nan")
    try:
        with open("/proc/meminfo") as f:
            avail = next(int(line.split()[1]) for line in f if line.startswith("MemAvailable")) / 2 ** 20
    except (OSError, StopIteration):
        pass
    gb = 2 ** 30
    log(f"mem {tag}: allocated {torch.cuda.memory_allocated() / gb:.1f} GB, peak {torch.cuda.max_memory_allocated() / gb:.1f}, "
        f"reserved {torch.cuda.memory_reserved() / gb:.1f}, host MemAvailable {avail:.1f} GB")


def cosine(step: int, total: int, lr: float, floor: float, warmup: int) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + 0.5 * (lr - floor) * (1 + math.cos(math.pi * t))


def _kl_piece(fs, ft, ws, wt):
    s = F.linear(fs, ws).float().log_softmax(-1)
    t = F.linear(ft, wt).float().log_softmax(-1)
    return (t.exp() * (t - s)).sum()


def head_kd(final_s, final_t, head_s, head_t, chunk: int = 2048) -> torch.Tensor:
    """sum over rows of KL(teacher || student), heads applied per chunk (recomputed in
    backward), divided by the row count."""
    n = final_s.shape[0]
    total = final_s.new_zeros((), dtype=torch.float32)
    for s in range(0, n, chunk):
        fs = final_s[s:s + chunk].to(torch.bfloat16)
        ft = final_t[s:s + chunk].to(torch.bfloat16)
        total = total + checkpoint(_kl_piece, fs, ft, head_s.to(torch.bfloat16), head_t.to(torch.bfloat16),
                                   use_reentrant=False)
    return total / n


@torch.no_grad()
def head_stats(final_s, final_t, head_s, head_t, targets, chunk: int = 2048) -> tuple[float, float, float]:
    """(sum KL, sum NLL student, sum NLL teacher) over rows."""
    kl = nll_s = nll_t = 0.0
    for s in range(0, final_s.shape[0], chunk):
        ls = F.linear(final_s[s:s + chunk].to(torch.bfloat16), head_s.to(torch.bfloat16)).float().log_softmax(-1)
        lt = F.linear(final_t[s:s + chunk].to(torch.bfloat16), head_t.to(torch.bfloat16)).float().log_softmax(-1)
        tg = targets[s:s + chunk]
        kl += float((lt.exp() * (lt - ls)).sum())
        nll_s -= float(ls.gather(-1, tg[:, None]).sum())
        nll_t -= float(lt.gather(-1, tg[:, None]).sum())
    return kl, nll_s, nll_t


def long_val_rows(tok, corpus: str, n: int, seq: int, skip: int = 16, joined: bool = False) -> torch.Tensor:
    """n held-out books of at least seq tokens (one row of BOS + seq-1 tokens each), after the
    first `skip` such books (evaluate.py longnll scores the first 8), so they are disjoint from
    the long-NLL eval. joined=True: rows of consecutive held-out book chunks (JoinedChunks) from
    the second half of the holdout books (longnll/longnll32 read from its start)."""
    if joined:
        source = JoinedChunks(corpus, tok, split="holdout", start_fraction=0.5)
        rows, i = [], source.start
        while len(rows) < n and i < source.stop:
            row, i = source.row_from(i, seq)
            if row is not None:
                rows.append(row)
        return torch.tensor(rows, dtype=torch.long)
    from .common import load_byte_shards
    holdout = load_byte_shards().ShardedByteCorpus(Path(corpus).expanduser(), "holdout")
    start, stop = holdout.source_range("books")
    rows, seen = [], 0
    for i in range(start, stop):
        ids = tok.encode_batch([holdout.text(i)[:seq * 8]])[0]
        if len(ids) + 1 < seq:
            continue
        seen += 1
        if seen > skip:
            rows.append([tok.bos] + ids[:seq - 1])
        if len(rows) >= n:
            break
    return torch.tensor(rows, dtype=torch.long)


@torch.no_grad()
def validate_long(student, teacher, rows, device, split: int = 2048) -> dict:
    """Long single-document rows (held-out books): KL and NLL of both models before and after
    position `split` (beyond a bounded window is where conversions lose what attention keeps)."""
    if rows is None or not len(rows):
        return {}
    out = {}
    student.eval()
    sums = {"near": [0.0, 0.0, 0.0, 0], "far": [0.0, 0.0, 0.0, 0]}
    for r in range(rows.shape[0]):
        ids = rows[r:r + 1].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16), packs_bypassed(teacher):
            ft = teacher(ids, logits=False)["final"][0, :-1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fs = student(ids, logits=False)["final"][0, :-1]
        target = ids[0, 1:]
        for name, sl in (("near", slice(0, split)), ("far", slice(split, None))):
            a, b, c = head_stats(fs[sl], ft[sl], student.head_weight(), teacher.head_weight(), target[sl])
            acc = sums[name]
            acc[0] += a
            acc[1] += b
            acc[2] += c
            acc[3] += fs[sl].shape[0]
    student.train()
    for name, (kl, ns, nt, n) in sums.items():
        if n:
            out[f"long_{name}_kl"] = kl / n
            out[f"long_{name}_nll_student"] = ns / n
            out[f"long_{name}_nll_teacher"] = nt / n
    return out


@torch.no_grad()
def validate(student, teacher, val_rows, device, rows_per_call=None) -> dict:
    """Per-token KL and NLL of student and teacher on packed held-out rows, about 16k tokens per call."""
    if rows_per_call is None:
        rows_per_call = max(1, 16384 // val_rows.shape[1])
    student.eval()
    kl = nll_s = nll_t = 0.0
    n = 0
    for s in range(0, val_rows.shape[0], rows_per_call):
        ids = val_rows[s:s + rows_per_call].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16), packs_bypassed(teacher):  # (no-op without packs)
            ft = teacher(ids, logits=False)["final"][:, :-1].reshape(-1, teacher.c.width)
            fs = student(ids, logits=False)["final"][:, :-1].reshape(-1, teacher.c.width)
        a, b, c = head_stats(fs, ft, student.head_weight(), teacher.head_weight(), ids[:, 1:].reshape(-1))
        kl, nll_s, nll_t = kl + a, nll_s + b, nll_t + c
        n += fs.shape[0]
    student.train()
    return {"val_kl": kl / n, "val_nll_student": nll_s / n, "val_nll_teacher": nll_t / n}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--mixer", default="recurrent",
                    help="recurrent | mingru | hybrid:<k> | keep:<i,j,..> | swa:<W>[+<sinks>] | swa_gdn:<W>[+<sinks>] | exact")
    ap.add_argument("--no-mixer-init", action="store_true", help="fresh mixers (ablation of attention init)")
    ap.add_argument("--converted-key-dim", type=int, default=0,
                    help="Gated DeltaNet key dim of the mixers replacing attention (state = value heads x dk x dv)")
    ap.add_argument("--converted-heads", type=int, default=0,
                    help="key/value heads of the mixers replacing attention (0 = one per query head); the "
                         "teacher's KV heads give a GQA-sized GDN initialized 1:1 from them")
    ap.add_argument("--layer-windows", default="", help="per-layer attention windows, e.g. 15=4096,11=4096")
    ap.add_argument("--converted-head-mult", type=int, default=1,
                    help="mixers replacing attention get this many times the GDN heads; --init from a narrower "
                         "core is widened by head copies (widen_gdn_state)")
    ap.add_argument("--layer-head-mults", default="",
                    help="per-layer head multipliers overriding --converted-head-mult, e.g. 15=8 (a larger state "
                         "for one converted layer; --init from a narrower core widens it by head copies)")
    ap.add_argument("--widen-noise", type=float, default=0.02,
                    help="relative noise on the copied q/k heads when widening, so the copies can diverge")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--corpus", default="~/corpus_v4")
    ap.add_argument("--mixture", default=STAGE3_MIXTURE)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--doc-bytes", type=int, default=32768, help="bytes read per document")
    ap.add_argument("--seq", type=int, default=2048, help="tokens per row")
    ap.add_argument("--val-seq", type=int, default=0,
                    help="tokens per validation row (val and val-long; 0 = --seq). Validation runs whole rows "
                         "unchunked, so very long training rows need a shorter validation length")
    ap.add_argument("--long-fraction", type=float, default=0.0,
                    help="fraction of each batch taken from single documents of at least seq tokens")
    ap.add_argument("--long-mixture", default="books=0.5,openthoughts=0.2,code=0.15,openr1_math=0.1,wikipedia=0.05",
                    help="sources for the long-document rows (books are ~58 KB, the others have long tails)")
    ap.add_argument("--long-join-fraction", type=float, default=1.0,
                    help="with --long-join: this share of the long rows is joined chunks, the rest single documents "
                         "of at least --seq tokens from --long-mixture")
    ap.add_argument("--long-join", default=None,
                    help="long rows from consecutive chunks of this source joined (e.g. books) instead of single "
                         "documents of --long-mixture; --val-long rows are then joined held-out book chunks too")
    ap.add_argument("--chat-fraction", type=float, default=0.0,
                    help="fraction of each batch from corpus conversations rendered with the teacher's chat "
                         "template (thinking off), packed")
    ap.add_argument("--chat-mixture", default="conversation=1.0")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--optim", default="adamw", choices=("adamw", "adamw8bit", "paged_adamw8bit"))
    ap.add_argument("--batch-file", type=Path, default=None,
                    help="read training batches from dump_batches.py output instead of the corpus samplers")
    ap.add_argument("--stage1-tokens", type=float, default=30e6)
    ap.add_argument("--stage2-tokens", type=float, default=60e6)
    ap.add_argument("--stage1-layer-group", type=int, default=0,
                    help="stage 1: backward this many converted layers at a time (0 = all; bounds activation memory)")
    ap.add_argument("--stage1-seq", type=int, default=0,
                    help="stage 1: crop each row to this many tokens (0 = full --seq; bounds teacher-tap and mixer "
                         "activation memory; step count and the logged token totals still use --seq)")
    ap.add_argument("--stage1-layers", default="",
                    help="stage 1 trains only these converted layers (comma list; default all converted)")
    ap.add_argument("--stage1-after-init", action="store_true",
                    help="run stage 1 even with --init (layer-local re-alignment of an already trained core; "
                         "the other weights stay as --init left them)")
    ap.add_argument("--lr1", type=float, default=1e-3)
    ap.add_argument("--lr2", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--stage2-train", default="converted", choices=("converted", "mixers", "all"))
    ap.add_argument("--lr2-base-scale", type=float, default=1.0,
                    help="with --stage2-train all: lr multiplier for the non-mixer weights (MLPs, norms, embeddings)")
    ap.add_argument("--kd-fraction", type=float, default=0.25, help="positions per row scored by the KD loss")
    ap.add_argument("--tbptt-chunk", type=int, default=0,
                    help="stage 2: the student trains each row in chunks of this many tokens from the streaming "
                         "state the previous chunks left (detached; truncated BPTT), so rows longer than fit in "
                         "memory reach positions past one chunk; the teacher still reads the whole row")
    ap.add_argument("--full-bptt", action="store_true",
                    help="with --tbptt-chunk: exact backpropagation through the recurrent state across the whole "
                         "row at one chunk's activation memory (a no-grad pass stores each chunk's entry state; "
                         "chunks are then recomputed in reverse and the state gradient is carried backward)")
    ap.add_argument("--kd-temperature", type=float, default=1.0)
    ap.add_argument("--kd-chunk", type=int, default=2048, help="positions per fused head-KL chunk (peak memory ~ chunk x vocab)")
    ap.add_argument("--moe-token-chunk", type=int, default=0,
                    help="memory: no-grad MoE passes (the teacher, validation) run the routed experts over at most "
                         "this many tokens at a time (each token's experts are independent of the others)")
    ap.add_argument("--empty-cache", action="store_true",
                    help="memory: return the allocator's cached blocks after the teacher pass and validation (on "
                         "the GB10's unified memory cached blocks count against the host's MemAvailable)")
    ap.add_argument("--kd-reverse", type=float, default=0.0, help="weight of the reverse KL(student||teacher) term")
    ap.add_argument("--exact-memory", default="",
                    help="train with the exact memory tier active (efficiency/exact_memory.py): blocks,block_size[,decode_blocks,neighbors,rerank]; with --tbptt-chunk the archive is carried across chunks")
    ap.add_argument("--recall-data", default=None,
                    help="jsonl of verified recall traces (recall_data.py): mixed into stage 2 as SFT")
    ap.add_argument("--mc-data", default=None,
                    help="jsonl of multiple-choice items (mc_data.py): answer-letter cross-entropy mixed into stage 2")
    ap.add_argument("--mc-val", default=None, help="held-out multiple-choice jsonl scored at every eval")
    ap.add_argument("--mc-every", type=int, default=1)
    ap.add_argument("--mc-batch", type=int, default=8)
    ap.add_argument("--mc-weight", type=float, default=0.5)
    ap.add_argument("--mc-max-tokens", type=int, default=4096)
    ap.add_argument("--mckl-data", default=None,
                    help="jsonl of multiple-choice documents (mc_kl_data.py): full-distribution KL mixed into stage 2")
    ap.add_argument("--mckl-every", type=int, default=1)
    ap.add_argument("--mckl-batch", type=int, default=4)
    ap.add_argument("--mckl-weight", type=float, default=1.0)
    ap.add_argument("--mckl-answer-weight", type=float, default=4.0,
                    help="extra weight of the KL at answer-letter positions")
    ap.add_argument("--mckl-align", type=float, default=0.0,
                    help="weight of converted-layer hidden-state matching on the same documents (0 = off)")
    ap.add_argument("--mckl-max-tokens", type=int, default=4096)
    ap.add_argument("--thread-data", default=None,
                    help="jsonl of long working conversations (thread_data.py): answer cross-entropy mixed into stage 2")
    ap.add_argument("--thread-every", type=int, default=2)
    ap.add_argument("--thread-weight", type=float, default=0.5)
    ap.add_argument("--thread-max-tokens", type=int, default=131072)
    ap.add_argument("--copy-every", type=int, default=0,
                    help="exact-copy SFT from long training filler (copy_data.py) every N stage-2 steps (0 = off; "
                         "needs --batch-file for the filler)")
    ap.add_argument("--copy-weight", type=float, default=1.0)
    ap.add_argument("--copy-min-tokens", type=int, default=24576)
    ap.add_argument("--copy-max-tokens", type=int, default=131072)
    ap.add_argument("--copy-far", type=int, default=16384,
                    help="the asked record lies at least this many tokens before the question (outside the window)")
    ap.add_argument("--copy-far-frac", type=float, default=0.85, help="fraction of copy items with --copy-far")
    ap.add_argument("--copy-prefill-chunk", type=int, default=8192)
    ap.add_argument("--copy-tail", type=int, default=64)
    ap.add_argument("--recall-every", type=int, default=4)
    ap.add_argument("--recall-weight", type=float, default=0.5)
    ap.add_argument("--recall-max-tokens", type=int, default=16384)
    ap.add_argument("--recall-reply", default="bare", choices=("bare", "natural"),
                    help="final reply of a recall trace: the bare answer, or a short sentence containing it")
    ap.add_argument("--recall-no-tool", type=float, default=0.0,
                    help="fraction of recall SFT examples offered NO tool, on a short context (the whole item "
                         "inside the attention window) with the direct answer as target: when no tool is offered "
                         "the model answers instead of calling one")
    ap.add_argument("--recall-no-tool-chars", type=int, default=10000)
    ap.add_argument("--recall-history-dropout", type=float, default=0.0,
                    help="opt-in fraction of tool-available earlier-conversation examples whose old fact is "
                         "omitted from the visible prompt; verified full-history tool results are retained")
    ap.add_argument("--chat-template-from", default=None, help="tokenizer with the chat template (base models)")
    ap.add_argument("--lora-adjacent", type=int, default=0,
                    help="stage 2: LoRA rank on the MLP projections of the converted layers and their neighbours")
    ap.add_argument("--lora-mixers", type=int, default=0,
                    help="stage 2: LoRA rank on the proj/out of every Gated DeltaNet layer kept from the teacher "
                         "(they may take over long-range work; with --shared the teacher's passes read the base)")
    ap.add_argument("--val-docs", type=int, default=96)
    ap.add_argument("--val-rows", type=int, default=16)
    ap.add_argument("--val-long", type=int, default=0,
                    help="also validate on this many held-out books (one row of --seq tokens each), split at 2048")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--init", type=Path, default=None, help="start from this core.pt (skips stage 1)")
    ap.add_argument("--compile", action="store_true", help="torch.compile the student core blocks")
    ap.add_argument("--shared", action="store_true",
                    help="the student shares all frozen weights with the bf16 teacher (27B memory path)")
    ap.add_argument("--checkpoint-blocks", action="store_true", help="activation checkpointing per block (stage 2)")
    ap.add_argument("--max-minutes", type=float, default=1e9)
    ap.add_argument("--resume", action="store_true",
                    help="write <out>/resume.pt at every eval (trainable tensors, optimizer, data samplers, "
                         "history) and continue from it when it exists")
    args = ap.parse_args(argv)
    if args.exact_memory:
        from efficiency.exact_memory import EXACT_MEMORY
        vals = [int(x) for x in args.exact_memory.split(",")]
        for key, val in zip(("blocks", "block_size", "decode_blocks", "neighbors", "rerank"), vals):
            EXACT_MEMORY[key] = val
        EXACT_MEMORY["enabled"] = True
        EXACT_MEMORY["keys_like_values"] = True  # bf16 archive keys, as served (fp32 keys OOMed the 131K copy prefill)
    if not 0.0 <= args.recall_history_dropout <= 1.0:
        ap.error("--recall-history-dropout must be between 0 and 1")
    if args.shared and args.stage2_train != "converted":
        ap.error("--shared students share every unconverted tensor with the teacher: train them through "
                 "--lora-mixers / --lora-adjacent, not --stage2-train mixers/all (that would train the teacher)")
    if args.lora_adjacent and args.lora_mixers and args.lora_adjacent != args.lora_mixers:
        ap.error("--lora-adjacent and --lora-mixers share one pack: give them the same rank")

    torch.manual_seed(args.seed)
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    tdir = teacher_dir(args.teacher)
    tok = TeacherTokenizer(tdir)
    teacher, _ = tp.load_hf_core(tdir, None, dtype=torch.bfloat16, device=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    c = teacher.c
    spec = student_spec(args.mixer, c)
    if (args.converted_key_dim or args.layer_windows or args.converted_head_mult > 1 or args.layer_head_mults
            or args.converted_heads):
        import dataclasses
        windows = tuple((int(a), int(b)) for a, b in (x.split("=") for x in args.layer_windows.split(",") if x))
        spec = dataclasses.replace(spec, converted_key_dim=args.converted_key_dim, layer_windows=windows,
                                   converted_head_mult=args.converted_head_mult, converted_heads=args.converted_heads,
                                   layer_head_mults=tuple((int(a), int(b)) for a, b in
                                                          (x.split("=") for x in args.layer_head_mults.split(",") if x)))
    teacher_kinds = tp.CoreSpec.exact(c).mixers
    if args.shared:  # 27B path: the student shares every frozen tensor with the teacher
        student, converted = tp.shared_student(teacher, spec, mixer_init=not args.no_mixer_init)
        copied = {"shared": True, "converted": converted}
    else:
        student, copied = tp.load_hf_core(tdir, spec, dtype=torch.float32, device=device,
                                          mixer_init=not args.no_mixer_init)
        converted = [i for i, (a, b) in enumerate(zip(spec.mixers, teacher_kinds)) if a != b]
    student.checkpoint_blocks = args.checkpoint_blocks
    tp.SparseMoE.token_chunk = args.moe_token_chunk
    history = {"args": {k: str(v) for k, v in vars(args).items()}, "copied": copied, "converted": converted,
               "stage1": [], "stage2": []}
    if args.init is not None:
        payload = torch.load(args.init.expanduser(), map_location="cpu", weights_only=False)
        if "mixers_only" in payload:  # a --shared checkpoint: take the mixers of layers converted here too
            own = student.state_dict()  # (references, no copies)
            state = {k: v for k, v in payload["mixers_only"].items()
                     if k in own and int(k.split(".")[1]) in converted}
            skipped = sorted({int(k.split(".")[1]) for k in payload["mixers_only"]} - {int(k.split(".")[1]) for k in state})
            student.load_state_dict(state, strict=False)
            log(f"init: {len(state)} mixer tensors from {args.init}; its layers not converted here: {skipped}")
        else:
            state = payload["state_dict"]
            widened = tp.widen_gdn_state(state, student, args.widen_noise)
            if widened:
                log(f"init: widened layers {widened} to {[spec.head_mult(i) for i in widened]}x heads from {args.init}")
            student.load_state_dict(state)
        history["init"] = str(args.init)
    def stage1_losses(layers, inputs, targets):
        return [tp.normalized_mse(tp.mixer_delta(student.blocks[i], x.float()), t)
                for i, x, t in zip(layers, inputs, targets)]

    def student_final(ids):
        return student(ids, logits=False)["final"]

    def teacher_taps(ids):
        return teacher(ids, collect=converted, logits=False)

    def teacher_final(ids):
        return teacher(ids, logits=False)["final"]

    if args.compile:  # fuses the core blocks' elementwise work (fla stays an opaque custom op)
        torch._dynamo.config.cache_size_limit = 64
        stage1_losses = torch.compile(stage1_losses, dynamic=False)
        student_final = torch.compile(student_final, dynamic=False)
        teacher_taps = torch.compile(teacher_taps, dynamic=False)
        teacher_final = torch.compile(teacher_final, dynamic=False)
    params = sum(p.numel() for p in student.parameters())
    log(f"teacher {tdir}: {json.dumps(tp.asdict(c))[:600]}")
    log(f"student {args.mixer}: {params / 1e6:.1f}M params; converted layers {converted}")
    windows = Windows(args.corpus, args.mixture, args.seed, args.doc_bytes)
    long_windows = None
    long_single = None
    if args.long_fraction > 0 and args.long_join:  # continuous rows of any length from consecutive chunks
        long_windows = JoinedChunks(args.corpus, tok, source=args.long_join, seed=args.seed + 1)
        if args.long_join_fraction < 1.0:
            long_single = Windows(args.corpus, args.long_mixture, args.seed + 3, args.seq * 8, min_bytes=args.seq * 3)
    elif args.long_fraction > 0:  # documents of >= ~seq tokens only (about 3 bytes per token at least)
        long_windows = Windows(args.corpus, args.long_mixture, args.seed + 1, args.seq * 8, min_bytes=args.seq * 3)
    chat_windows, render = None, None
    if args.chat_fraction > 0:
        from transformers import AutoTokenizer
        from .common import conversation_messages
        chat_tok = AutoTokenizer.from_pretrained(teacher_dir(args.chat_template_from or args.teacher))

        def render(text):
            msgs = conversation_messages(text)
            return None if msgs is None else chat_tok.apply_chat_template(msgs, tokenize=False,
                                                                          enable_thinking=False)
        chat_windows = Windows(args.corpus, args.chat_mixture, args.seed + 2, args.seq * 8)
    if args.batch_file is not None:
        batches = FileBatches(args.batch_file.expanduser(), args.batch, args.seq)
    else:
        batches = TokenBatches(windows, tok, args.batch, args.seq, long_windows, args.long_fraction,
                               chat_windows, args.chat_fraction, render, long_single, args.long_join_fraction)
    val_texts = holdout_texts(args.corpus, args.val_docs, 8192)
    val_seq = args.val_seq or args.seq
    val_rows, _ = pack_tokens(tok.encode_batch(val_texts), tok.bos, val_seq)
    val_rows = torch.tensor(val_rows[:args.val_rows], dtype=torch.long)
    val_long = None
    if args.val_long:  # held-out books after the first 16 long ones (evaluate.py longnll scores the first 8)
        val_long = long_val_rows(tok, args.corpus, args.val_long, val_seq, joined=bool(args.long_join))
        log(f"long validation: {val_long.shape[0]} held-out books of {val_seq} tokens")
    logf = open(out / "log.jsonl", "a", encoding="utf-8")

    def record(stage: str, row: dict) -> None:
        row = {"stage": stage, "time": round(time.time() - started, 1), **row}
        history[stage].append(row)
        logf.write(json.dumps(row) + "\n")
        logf.flush()
        log(json.dumps(row)[:400])
        if args.empty_cache:  # validation's long no-grad passes leave large cached blocks
            torch.cuda.empty_cache()

    lora_shared = False  # set when --lora-adjacent adapts MLPs the --shared teacher also runs

    def save(tag: str) -> None:
        if args.shared:  # only the new mixers are the student's own; the rest is the teacher
            state = {f"blocks.{i}.{k}": v for i in converted
                     for k, v in student.blocks[i].state_dict().items() if not k.startswith(("mlp.", "norm."))}
            payload = {"config": student.config_dict(), "mixers_only": state, "teacher": str(tdir),
                       "history": history}
            if lora_shared:  # the adjacent-MLP adapters stay factors here: the teacher shares those MLPs
                from efficiency.lora import pack_tensors
                rank = args.lora_adjacent or args.lora_mixers
                payload["lora_adjacent"] = {"rank": rank, "alpha": 2.0 * rank,
                                            "tensors": pack_tensors(student, "adjacent")}
            torch.save(payload, out / f"{tag}.pt")
            return
        state = student.state_dict()
        if args.lora_adjacent or args.lora_mixers:  # save the adapted weights as plain weights (LoRA merged in a copy)
            import copy
            from efficiency.lora import attached_packs, remove_pack
            merged = copy.deepcopy(student)
            for name in attached_packs(merged):
                remove_pack(merged, name, merge=True)
            state = merged.state_dict()
            del merged
        torch.save({"config": student.config_dict(), "state_dict": state, "history": history},
                   out / f"{tag}.pt")

    rng = None  # the recall-example sampler is initialized for stage 2
    rpath = out / "resume.pt"
    resume = torch.load(rpath, map_location="cpu", weights_only=False) if args.resume and rpath.exists() else None
    saved_recall_rng_state = None if resume is None else resume.get("recall_rng_state")
    if resume is not None:  # data samplers, RNG and history as they were at the last eval
        history = resume["history"]
        history.setdefault("resumed", []).append({"stage": resume["stage"], "step": resume["step"],
                                                  "time": time.strftime("%F %T"),
                                                  "args": {k: str(v) for k, v in vars(args).items()}})
        windows.load(resume["windows"])
        if long_windows is not None and resume.get("long_windows") is not None:
            long_windows.load(resume["long_windows"])
        if chat_windows is not None and resume.get("chat_windows") is not None:
            chat_windows.load(resume["chat_windows"])
        if long_single is not None and resume.get("long_single") is not None:
            long_single.load(resume["long_single"])
        torch.set_rng_state(resume["rng"][0])
        torch.cuda.set_rng_state(resume["rng"][1])
        if isinstance(batches, FileBatches):  # batches consumed so far = stage-1 steps (+ stage-2 steps)
            s1 = 0 if args.init is not None and not args.stage1_after_init else int(args.stage1_tokens // (args.batch * args.seq))
            batches.cursor = resume["step"] if resume["stage"] == "stage1" else s1 + resume["step"]
        log(f"resume from {rpath}: {resume['stage']} after step {resume['step']}")

    def save_resume(stage: str, step: int, opt) -> None:
        """Everything a restart needs to continue this stage at `step` (atomic replace)."""
        if not args.resume:
            return
        state = {"stage": stage, "step": step, "opt": None if opt is None else opt.state_dict(),
                 "trainable": {n: p.detach().to("cpu", copy=True) for n, p in student.named_parameters()
                               if p.requires_grad},
                 "windows": windows.state(), "long_windows": None if long_windows is None else long_windows.state(),
                 "chat_windows": None if chat_windows is None else chat_windows.state(),
                 "long_single": None if long_single is None else long_single.state(),
                 "history": history, "rng": [torch.get_rng_state(), torch.cuda.get_rng_state()],
                 "recall_rng_state": None if rng is None else rng.getstate()}
        tmp = out / "resume.pt.tmp"
        torch.save(state, tmp)
        tmp.replace(rpath)

    def restore(opt) -> None:
        named = dict(student.named_parameters())
        with torch.no_grad():
            for n, t in resume["trainable"].items():
                named[n].copy_(t)
        if resume["opt"] is not None:
            opt.load_state_dict(resume["opt"])
            if getattr(opt, "is_paged", False):
                # bitsandbytes loads the 8-bit states as plain CPU tensors and its first step moves them to the GPU
                # unpaged (to_gpu skips only is_paged tensors), so a resumed run held ~3.5 GB more GPU memory than
                # a fresh one and OOMed (Nanbeige nbg4, ledger 2026-10-26 19:30 UTC). Copy them into paged buffers,
                # as get_state_buffer allocates them on a fresh run.
                import bitsandbytes.functional as bnbf
                for p, st in opt.state.items():
                    if not isinstance(p, torch.Tensor) or p.device.type != "cuda" or p.numel() < 1e5:
                        continue
                    for k in ("state1", "state2"):
                        v = st.get(k)
                        if isinstance(v, torch.Tensor) and not getattr(v, "is_paged", False):
                            buf = bnbf.get_paged(*v.shape, dtype=v.dtype, device=p.device)
                            buf.copy_(v)
                            opt.page_mng.paged_tensors.append(buf)
                            st[k] = buf
        log(f"restored {len(resume['trainable'])} trainable tensors" + (" and the optimizer" if resume["opt"] else ""))

    if resume is None:
        record("stage1", {"step": 0, "tokens": 0, **validate(student, teacher, val_rows, device),
                          **validate_long(student, teacher, val_long, device)})
    tokens_per_step = args.batch * args.seq
    budget_s = args.max_minutes * 60

    # ------------------------------------------------------------------ stage 1
    steps1 = 0 if args.init is not None and not args.stage1_after_init else int(args.stage1_tokens // tokens_per_step)
    start1 = 0 if resume is None else (resume["step"] if resume["stage"] == "stage1" else steps1)
    s1_layers = [int(x) for x in args.stage1_layers.split(",") if x] or converted
    if set(s1_layers) - set(converted):
        raise SystemExit(f"--stage1-layers {s1_layers} must be converted layers {converted}")
    if steps1 and converted and start1 < steps1:
        for p in student.parameters():
            p.requires_grad_(False)
        mixer_set = student.mixer_parameters(s1_layers)
        for p in mixer_set:
            p.requires_grad_(True)
        log(f"stage 1: {sum(p.numel() for p in mixer_set) / 1e6:.1f}M trainable mixer parameters, layers {s1_layers}")
        opt = make_optimizer(mixer_set, args.lr1, args.optim)
        if resume is not None:
            restore(opt)
        t0 = time.time()
        for step in range(start1, steps1):
            lr = cosine(step, steps1, args.lr1, args.lr1 * 0.01, args.warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            ids = batches.next().to(device)
            if args.stage1_seq:
                ids = ids[:, :args.stage1_seq]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                taps = teacher_taps(ids)
            opt.zero_grad(set_to_none=True)
            losses = []  # the layers' losses are independent: backward a group of layers at a time
            group = args.stage1_layer_group or len(s1_layers)
            for g in range(0, len(s1_layers), group):
                layers = s1_layers[g:g + group]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    part = stage1_losses(layers, [taps["inputs"][i] for i in layers], [taps["mix"][i] for i in layers])
                (torch.stack(part).sum() / len(s1_layers)).backward()
                losses.extend(x.detach() for x in part)
            del taps
            loss = torch.stack(losses).mean()
            torch.nn.utils.clip_grad_norm_(mixer_set, 1.0)
            opt.step()
            if step % 25 == 0:
                per = [round(float(x.detach()), 4) for x in losses]
                rate = (step + 1 - start1) * tokens_per_step / (time.time() - t0)
                log(f"s1 {step}/{steps1} lr {lr:.2e} loss {float(loss.detach()):.4f} tok/s {rate:.0f} layers {per}")
            if (step + 1) % args.eval_every == 0 or step + 1 == steps1:
                record("stage1", {"step": step + 1, "tokens": (step + 1) * tokens_per_step,
                                  "align_loss": float(loss.detach()),
                                  "per_layer": [float(x.detach()) for x in losses],
                                  "tok_per_s": (step + 1 - start1) * tokens_per_step / (time.time() - t0),
                                  **validate(student, teacher, val_rows, device),
                                  **validate_long(student, teacher, val_long, device)})
                save_resume("stage1", step + 1, opt)
            if time.time() - started > budget_s:
                log("time budget reached in stage 1")
                break
        save("stage1")
        save_resume("stage2", 0, None)
        del opt

    # ------------------------------------------------------------------ stage 2
    steps2 = int(args.stage2_tokens // tokens_per_step)
    params2 = stage2_parameters(student, args.stage2_train, converted) if steps2 else []
    if steps2 and not params2:
        log(f"stage 2 skipped: --stage2-train {args.stage2_train} selects no parameters "
            f"(converted layers: {converted})")
        steps2 = 0
    if steps2 and (args.lora_adjacent or args.lora_mixers):
        from efficiency.lora import add_pack
        layers, names = [], []
        if args.lora_adjacent:
            layers = sorted({j for i in converted for j in (i - 1, i, i + 1) if 0 <= j < c.layers})
            names = [f"blocks.{j}.mlp.{n}" for j in layers for n, m in student.blocks[j].mlp.named_children()
                     if isinstance(m, torch.nn.Linear) and n in ("gate_up", "down", "shared_gate_up", "shared_down")]
        if args.lora_mixers:  # the teacher's own Gated DeltaNet layers (not the converted ones)
            names += [f"blocks.{j}.{n}" for j, b in enumerate(student.blocks)
                      if j not in converted and isinstance(b, tp.GatedDeltaNetLayer) for n in ("proj", "out")]
        rank = args.lora_adjacent or args.lora_mixers
        add_pack(student, "adjacent", rank=rank, alpha=2.0 * rank, targets=names)
        if args.shared:  # the teacher shares these MLPs: its passes read the base weights (packs_bypassed)
            lora_shared = True
            for name, p in student.named_parameters():
                if ".packs." in name:
                    p.data = p.data.float()  # fp32 factors on the bf16 teacher weights
        params2 = [p for p in student.parameters() if p.requires_grad]
        history["lora_adjacent"] = {"rank": rank, "modules": len(names), "layers": layers,
                                    "mixers": bool(args.lora_mixers)}
    if steps2:
        log(f"stage 2: {sum(p.numel() for p in params2) / 1e6:.1f}M trainable ({args.stage2_train})")
        if args.stage2_train == "all" and args.lr2_base_scale != 1.0:
            mixer_ids = {id(p) for p in student.mixer_parameters()}
            groups = [{"params": [p for p in params2 if id(p) in mixer_ids], "lr_scale": 1.0},
                      {"params": [p for p in params2 if id(p) not in mixer_ids], "lr_scale": args.lr2_base_scale}]
            log(f"stage 2 lr: mixers x1.0 ({sum(p.numel() for p in groups[0]['params']) / 1e6:.1f}M), "
                f"other weights x{args.lr2_base_scale} ({sum(p.numel() for p in groups[1]['params']) / 1e6:.1f}M)")
            opt = make_optimizer(groups, args.lr2, args.optim)
        else:
            opt = make_optimizer(params2, args.lr2, args.optim)
        start2 = 0
        if resume is not None and resume["stage"] == "stage2":
            restore(opt)
            start2 = resume["step"]
        if resume is not None:
            # The on-disk resume contains CPU copies of the trainable tensors and
            # optimizer (several GB for a MoE). Keeping the dict alive throughout
            # stage 2 can push the GB10 below Spark's memory-guard floor.
            del resume
            gc.collect()
            log_memory("after releasing loaded resume state")
        t0 = time.time()
        keep = max(1, int(round(args.kd_fraction * (args.seq - 1))))
        tbptt = bool(args.tbptt_chunk) and args.tbptt_chunk < args.seq
        if tbptt:
            log(f"stage 2 truncated BPTT: rows of {args.seq} in chunks of {args.tbptt_chunk}, state carried (detached)")
        recall_items, recall_tok, rng = [], None, None
        mc_items, mc_tok, mc_rng = [], None, None
        mc_val_items = []
        if args.mc_val:
            mc_val_items = [json.loads(line) for line in open(Path(args.mc_val).expanduser(), encoding="utf-8")][:300]
        mckl_items = []
        if args.mc_data or args.mc_val or args.mckl_data:
            import random as _random
            from transformers import AutoTokenizer as _Tok
            mc_tok = _Tok.from_pretrained(teacher_dir(args.chat_template_from or args.teacher))
            mc_rng = _random.Random(args.seed + 1)
        if args.mc_data:
            mc_items = [json.loads(line) for line in open(Path(args.mc_data).expanduser(), encoding="utf-8")]
            log(f"multiple-choice SFT: {len(mc_items)} items, batch {args.mc_batch}, every {args.mc_every} steps, "
                f"weight {args.mc_weight}")
        if args.mckl_data:
            mckl_items = [json.loads(line) for line in open(Path(args.mckl_data).expanduser(), encoding="utf-8")]
            log(f"multiple-choice KL: {len(mckl_items)} documents, batch {args.mckl_batch}, every {args.mckl_every} "
                f"steps, weight {args.mckl_weight}, answer x{args.mckl_answer_weight}, align {args.mckl_align}")
        if args.recall_data:
            import random
            from transformers import AutoTokenizer
            recall_items = [json.loads(line) for line in open(Path(args.recall_data).expanduser(), encoding="utf-8")]
            recall_tok = AutoTokenizer.from_pretrained(teacher_dir(args.chat_template_from or args.teacher))
            rng = random.Random(args.seed)
            if saved_recall_rng_state is not None:
                rng.setstate(saved_recall_rng_state)
                log("restored recall SFT example sampler")
            log(f"recall SFT: {len(recall_items)} verified traces, every {args.recall_every} steps, "
                f"weight {args.recall_weight}")
        thread_items, thread_tok, thread_rng = [], None, None
        if args.thread_data:
            import random as _trandom
            from transformers import AutoTokenizer as _TTok
            thread_items = [json.loads(line) for line in open(Path(args.thread_data).expanduser(), encoding="utf-8")]
            thread_tok = _TTok.from_pretrained(teacher_dir(args.chat_template_from or args.teacher))
            thread_rng = _trandom.Random(args.seed + 2 + start2)  # reseeded by the resume step
            log(f"thread SFT: {len(thread_items)} conversations, every {args.thread_every} steps, "
                f"weight {args.thread_weight}, max {args.thread_max_tokens} tokens")
        copy_rows, copy_tok, copy_rng = None, None, None
        if args.copy_every:
            import random as _crandom
            import numpy as np
            from transformers import AutoTokenizer as _CTok
            if args.batch_file is None:
                raise SystemExit("--copy-every needs --batch-file (the training filler)")
            copy_rows = np.load(args.batch_file.expanduser(), mmap_mode="r")
            copy_tok = _CTok.from_pretrained(teacher_dir(args.chat_template_from or args.teacher))
            copy_rng = _crandom.Random(args.seed + 3 + start2)  # reseeded by the resume step
            log(f"copy SFT: every {args.copy_every} steps, weight {args.copy_weight}, "
                f"{args.copy_min_tokens}-{args.copy_max_tokens} tokens, far {args.copy_far} x{args.copy_far_frac}")
        for step in range(start2, steps2):
            if time.time() - started > budget_s:
                log("time budget reached in stage 2")
                break
            lr = cosine(step, steps2, args.lr2, args.lr2 * 0.1, args.warmup)
            for g in opt.param_groups:
                g["lr"] = lr * g.get("lr_scale", 1.0)
            ids = batches.next().to(device)
            pos = torch.rand(ids.shape[0], args.seq - 1, device=device).argsort(-1)[:, :keep]
            base = (lambda: packs_bypassed(teacher)) if lora_shared else contextlib.nullcontext  # fresh per use
            opt.zero_grad(set_to_none=True)
            if tbptt:  # the teacher reads the whole row; the student trains chunk by chunk from its carried state
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), base():
                    ft_row = teacher_final(ids)[:, :-1]
                watch = step - start2 < 3  # memory of the first steps (the peak is set there)
                if args.empty_cache:
                    torch.cuda.empty_cache()
                if watch:
                    log_memory(f"s2 step {step} after the teacher pass")
                loss, state = torch.zeros((), device=device), None
                if args.full_bptt:
                    loss = reverse_bptt_row(student, teacher, ids, ft_row, pos, args)
                    state = None
                for c0 in (() if args.full_bptt else range(0, args.seq, args.tbptt_chunk)):
                    c1 = min(args.seq, c0 + args.tbptt_chunk)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        fs_chunk, state = student.forward_state(ids[:, c0:c1], state)
                    state = student.detach_state(state)
                    rows, cols = ((pos >= c0) & (pos < c1)).nonzero(as_tuple=True)
                    if not len(rows):
                        continue
                    where = pos[rows, cols]
                    part = tp.fused_head_kl(fs_chunk[rows, where - c0], ft_row[rows, where],
                                            student.head_weight().to(torch.bfloat16), teacher.head_weight(),
                                            chunk=args.kd_chunk, compiled=args.compile, temperature=args.kd_temperature,
                                            reverse=args.kd_reverse) * (len(rows) / pos.numel())
                    part.backward()
                    loss = loss + part.detach()
                    del fs_chunk, part
                    if watch:
                        log_memory(f"s2 step {step} after chunk {c0}-{c1}")
                del ft_row, state
            else:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), base():
                    ft = teacher_final(ids)[:, :-1]
                    ft = ft.gather(1, pos[..., None].expand(-1, -1, c.width)).reshape(-1, c.width)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    fs = student_final(ids)[:, :-1]
                fs = fs.gather(1, pos[..., None].expand(-1, -1, c.width)).reshape(-1, c.width)
                loss = tp.fused_head_kl(fs, ft, student.head_weight().to(torch.bfloat16), teacher.head_weight(),
                                        chunk=args.kd_chunk, compiled=args.compile, temperature=args.kd_temperature,
                                        reverse=args.kd_reverse)
                loss.backward()
            sft = None
            if recall_items and step % args.recall_every == 0:  # the recall tool's SFT traces
                sft = recall_sft_step(student_final, student, recall_items, recall_tok, rng, args)
                if sft.requires_grad:  # (a chunked long trace backpropagated inside)
                    (args.recall_weight * sft).backward()
            thr = None
            if thread_items and step % args.thread_every == 0:  # keeping the thread of a long session
                thr = thread_sft_step(student_final, student, thread_items, thread_tok, thread_rng, args)
                if thr is not None and thr.requires_grad:
                    (args.thread_weight * thr).backward()
            cp = None
            if copy_rows is not None and step % args.copy_every == 0:  # exact copy from beyond the window
                if args.empty_cache:
                    torch.cuda.empty_cache()
                cp = copy_sft_step(student, copy_rows, copy_tok, copy_rng, args, watch=tbptt and step - start2 < 3)
                (args.copy_weight * cp).backward()
                if tbptt and step - start2 < 3:
                    log_memory(f"s2 step {step} after the copy step")
            mc = None
            if mc_items and step % args.mc_every == 0:
                mc = mc_sft_step(student_final, student, mc_items, mc_tok, mc_rng, args)
                (args.mc_weight * mc).backward()
            mckl = None
            if mckl_items and step % args.mckl_every == 0:  # backward inside, document by document
                mckl = mckl_step(student, teacher, mckl_items, mc_tok, mc_rng, args, converted, base)
            torch.nn.utils.clip_grad_norm_(params2, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)  # grads are not held through the next teacher pass
            if tbptt and step - start2 < 3:
                log_memory(f"s2 step {step} after the optimizer step")
            if step % 25 == 0:
                rate = (step + 1 - start2) * tokens_per_step / (time.time() - t0)
                extra = "" if sft is None else f" recall_ce {float(sft.detach()):.4f}"
                if mc is not None:
                    extra += f" mc_ce {float(mc.detach()):.4f}"
                if thr is not None:
                    extra += f" thread_ce {float(thr.detach()):.4f}"
                if cp is not None:
                    extra += f" copy_ce {float(cp.detach()):.4f}"
                if mckl is not None:
                    extra += " mckl {:.4f} ans {:.4f} align {:.4f}".format(*(float(v) for v in mckl))
                log(f"s2 {step}/{steps2} lr {lr:.2e} kl {float(loss.detach()):.4f} tok/s {rate:.0f}{extra}")
            if (step + 1) % args.eval_every == 0 or step + 1 == steps2:
                record("stage2", {"step": step + 1, "tokens": (step + 1) * tokens_per_step,
                                  "kd_loss": float(loss.detach()),
                                  "tok_per_s": (step + 1 - start2) * tokens_per_step / (time.time() - t0),
                                  **validate(student, teacher, val_rows, device),
                                  **validate_long(student, teacher, val_long, device),
                                  **(mc_validate(student, mc_val_items, mc_tok, args) if mc_val_items else {})})
                save("core")
                save_resume("stage2", step + 1, opt)
    save("core")
    history["padding_dropped_tokens"] = batches.dropped
    history["chat_skipped_documents"] = batches.chat_skipped
    history["peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    (out / "summary.json").write_text(json.dumps(
        {"args": history["args"], "copied": history["copied"], "converted": converted,
         "stage1_last": history["stage1"][-1:], "stage2_last": history["stage2"][-1:],
         "peak_gb": history["peak_gb"], "minutes": (time.time() - started) / 60}, indent=2))
    log(f"done in {(time.time() - started) / 60:.1f} min, peak {history['peak_gb']:.1f} GB")


if __name__ == "__main__":
    main()
