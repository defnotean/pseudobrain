"""Programmatic, verified training data and evaluation items for the `recall` tool.

Every item is a long context (held-out or training corpus text as filler) with facts placed at
known depths, a question, and the exact answer. Its trace uses the real tool:

    user: <context ...> Question: ...
    assistant: <tool_call> recall(query=..., near=...) </tool_call>
    user: <tool_response> {SessionHistory.recall(...) on the context} </tool_response>
    assistant: <the exact answer>

The tool result is computed with learning/recall_tool.py on the same history the runtime would
hold, and an item is kept only if the answer string appears verbatim in that result
(verification by exact match). Tasks:
  needle    "The pass key is <digits>." / "The secret word is <word>."
  multihop  "<Name>'s locker number is <n>." ... "The code for locker <n> is <code>." (two
            recalls: name -> locker, locker -> code)
  earlier   multi-turn: an early user turn says something about a topic; much later "What did
            I tell you earlier about <topic>?"
  copy      a long random code/number string early; "Copy the string exactly."

turn_examples() gives each assistant turn as (generation prompt, completion), exactly as the
runtime prompts the model; that is what the conversion's SFT trains on. render() (the whole
conversation plus assistant spans) is kept for inspection.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import string
from pathlib import Path

from . import common  # noqa: F401  (puts the repo and engine/ on sys.path first)
from learning.recall_tool import SCHEMA, SessionHistory

NAMES = ("Alice", "Bruno", "Chandra", "Dmitri", "Elena", "Farid", "Greta", "Hiro", "Ines", "Jonas",
         "Keiko", "Luca", "Mira", "Nadia", "Omar", "Priya", "Quinn", "Rosa", "Sven", "Tariq")
WORDS = ("amber", "bramble", "cobalt", "driftwood", "ember", "fjord", "garnet", "harbor", "indigo",
         "juniper", "kestrel", "lantern", "meadow", "nimbus", "obsidian", "pepper", "quartz", "ripple")
TOPICS = ("my sister's wedding date", "the garage door code", "my favourite tea", "the bug in the parser",
          "the hotel in Lisbon", "my dentist appointment", "the name of my cat", "the budget for the trip")


def _insert(filler: str, facts: list[tuple[float, str]]) -> str:
    """Insert fact sentences at fractional depths, at sentence boundaries of the filler."""
    out = filler
    for frac, fact in sorted(facts, key=lambda f: -f[0]):
        at = int(len(out) * frac)
        dot = out.find(". ", at)
        at = len(out) if dot < 0 else dot + 2
        out = out[:at] + fact + " " + out[at:]
    return out


def make_item(task: str, filler: str, rng: random.Random) -> dict:
    depth = rng.choice((0.05, 0.25, 0.5, 0.75, 0.95))
    if task == "needle":
        if rng.random() < 0.5:
            key = "".join(rng.choice(string.digits) for _ in range(rng.choice((5, 6, 7))))
            fact, question, answer, calls = (f"The pass key is {key}.", "What is the pass key?", key,
                                             [{"query": "pass key"}])
        else:
            word = rng.choice(WORDS) + "-" + "".join(rng.choice(string.digits) for _ in range(3))
            fact, question, answer, calls = (f"The secret word is {word}.", "What is the secret word?", word,
                                             [{"query": "secret word"}])
        context = _insert(filler, [(depth, fact)])
        turns = [context + "\n\nQuestion: " + question]
    elif task == "multihop":
        name = rng.choice(NAMES)
        locker = str(rng.randrange(100, 1000))
        code = "".join(rng.choice(string.digits) for _ in range(4))
        context = _insert(filler, [(depth * 0.5, f"{name}'s locker number is {locker}."),
                                   (0.5 + depth * 0.45, f"The code for locker {locker} is {code}.")])
        question, answer = f"What is the code for {name}'s locker?", code
        calls = [{"query": f"{name}'s locker number"}, {"query": f"code for locker {locker}"}]
        turns = [context + "\n\nQuestion: " + question]
    elif task == "earlier":
        topic = rng.choice(TOPICS)
        detail = rng.choice(("it is on the 14th of March", "it is 4417", "it is oolong with honey",
                             "it drops the last token", "it is the one near the river", "it is next Tuesday at 9",
                             "her name is Pixel", "it is 2,300 euros"))
        early = f"Just so you know, about {topic}: {detail}."
        question, answer = f"What did I tell you earlier about {topic}?", detail
        calls = [{"query": topic}]
        turns = [early, filler, f"Quick question: {question}"]
    elif task == "copy":
        secret = "".join(rng.choice(string.ascii_letters + string.digits + "_-") for _ in range(rng.choice((24, 40, 64))))
        context = _insert(filler, [(depth, f"Here is the token you will need later: {secret}")])
        question, answer = "Copy the token I gave you exactly.", secret
        calls = [{"query": "token you will need later"}]
        turns = [context + "\n\nQuestion: " + question]
    else:
        raise ValueError(task)
    history = SessionHistory()
    for t in turns:
        history.append("user", t)
    results = [history.recall(c["query"], c.get("near")) for c in calls]
    verified = any(answer in r.get("text", "") for res in results for r in res.get("results", []))
    if task == "multihop":  # the second hop must find the code
        verified = any(answer in r["text"] for r in results[-1].get("results", []))
    return {"task": task, "turns": turns, "calls": calls, "results": results, "answer": answer,
            "question": question, "verified": verified}


def answer_found(item: dict, reply: str) -> bool:
    """Exact-match scoring: the answer string is in the reply. For "earlier" items the stored answer
    is the whole remark ("it is next Tuesday at 9"); a reply restating it in its own words ("your
    appointment is next Tuesday at 9") is correct too, so the detail after the leading "it is" /
    "it" / "her name is" also counts (case-insensitive)."""
    answer = item["answer"]
    if answer in reply:
        return True
    if item.get("task") == "earlier":
        core = re.sub(r"^(it is|her name is|it)\s+", "", answer)
        return core.lower() in reply.lower()
    return False


def reply_text(item: dict, natural: bool = False) -> str:
    """The final assistant reply of an item. Bare (default): the answer string itself. Natural: a
    short sentence that contains the answer verbatim (copy items stay the bare string: the task is
    to copy it exactly), so the SFT does not teach terse replies."""
    answer = item["answer"]
    if not natural:
        return answer
    task = item.get("task")
    if task == "needle":
        return f"The pass key is {answer}." if "pass key" in item["question"] else f"The secret word is {answer}."
    if task == "multihop":
        name = re.match(r"What is the code for (.+)'s locker\?", item["question"])
        return f"The code for {name.group(1)}'s locker is {answer}." if name else f"The code is {answer}."
    if task == "earlier":
        return f"You told me: {answer}."
    return answer


FACT_PATTERNS = re.compile(r"The pass key is \S+?\. |The secret word is \S+?\. |\S+'s locker number is \d+\. "
                           r"|The code for locker \d+ is \d+\. |Here is the token you will need later: \S+ ")


def strip_facts(text: str) -> str:
    """An item's context without its inserted fact sentences and question: plain filler again."""
    text = text.split("\n\nQuestion: ")[0]
    return FACT_PATTERNS.sub("", text)


def short_item(item: dict, rng: random.Random, chars: int) -> dict:
    """A new item of the same task on the first `chars` characters of `item`'s own filler: short
    enough that the whole context sits inside the converted model's attention window, so the model
    can answer it with no tool (the no-tool SFT examples: when no tool is offered, answer from
    context instead of emitting a call to a tool that does not exist)."""
    filler = strip_facts(item["turns"][1] if item.get("task") == "earlier" else item["turns"][0])[:chars]
    return make_item(item["task"], filler, rng)


def messages(item: dict, with_tool: bool = True, natural: bool = False, *, omit_earlier: bool = False) -> list[dict]:
    """Chat messages of an item: user turns (earlier turns get short acknowledgements), then
    the recall calls with their results, then the answer (reply_text)."""
    if omit_earlier and (not with_tool or item.get("task") != "earlier"):
        raise ValueError("history omission requires an earlier-conversation item with the recall tool")
    # The tool results were verified against the full history. Only the visible
    # model prompt omits the old fact, simulating detail beyond its attention window.
    turns = item["turns"][1:] if omit_earlier else item["turns"]
    msgs = []
    for i, turn in enumerate(turns):
        msgs.append({"role": "user", "content": turn})
        if i < len(turns) - 1:
            msgs.append({"role": "assistant", "content": "Noted."})
    if with_tool:
        for call, result in zip(item["calls"], item["results"]):
            msgs.append({"role": "assistant", "content": "",
                         "tool_calls": [{"type": "function", "function": {"name": "recall", "arguments": call}}]})
            msgs.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False, separators=(",", ":"))})
    msgs.append({"role": "assistant", "content": reply_text(item, natural)})
    return msgs


def render(tokenizer, item: dict, with_tool: bool = True) -> tuple[str, list[tuple[int, int]]]:
    """(chat-template text, assistant spans) for SFT: the loss covers the assistant turns that
    come after the last user context (the tool calls and the answer)."""
    msgs = messages(item, with_tool)
    text = tokenizer.apply_chat_template(msgs, tools=[SCHEMA] if with_tool else None, tokenize=False)
    spans = []
    prefix_msgs = []
    for m in msgs:
        before = tokenizer.apply_chat_template(prefix_msgs, tools=[SCHEMA] if with_tool else None,
                                               tokenize=False) if prefix_msgs else ""
        prefix_msgs.append(m)
        after = tokenizer.apply_chat_template(prefix_msgs, tools=[SCHEMA] if with_tool else None, tokenize=False)
        if m["role"] == "assistant" and m.get("content") != "Noted.":
            spans.append((len(before), len(after)))
    return text, spans


def turn_examples(tokenizer, item: dict, with_tool: bool = True, natural: bool = False, *,
                  omit_earlier: bool = False) -> list[tuple[str, str]]:
    """(prompt, completion) per trained assistant turn, rendered exactly as the runtime prompts
    the model (recall_eval.py): the chat-template generation prompt with thinking off, then the
    turn's text through its <|im_end|>. render() of the whole conversation is NOT what the model
    sees at each step: Qwen's template gives only the last assistant turn an (empty) think block,
    so a tool-call turn renders differently once later turns follow it."""
    msgs = messages(item, with_tool, natural, omit_earlier=omit_earlier)
    tools = [SCHEMA] if with_tool else None
    out = []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant" or m.get("content") == "Noted.":
            continue
        prompt = tokenizer.apply_chat_template(msgs[:i], tools=tools, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
        full = tokenizer.apply_chat_template(msgs[:i + 1], tools=tools, tokenize=False, enable_thinking=False)
        if not full.startswith(prompt):
            raise ValueError("the chat template renders this assistant turn differently from its generation prompt")
        out.append((prompt, full[len(prompt):].rstrip("\n")))
    return out


def sample_tool_example(tokenizer, item: dict, rng: random.Random, *, natural: bool = False,
                        history_dropout: float = 0.0) -> tuple[str, str]:
    """Opt-in missing-history curriculum; zero preserves the original sampler's RNG draws."""
    if not 0.0 <= history_dropout <= 1.0:
        raise ValueError("recall history dropout must be between 0 and 1")
    omit = bool(item.get("task") == "earlier" and history_dropout
                and rng.random() < history_dropout)
    return rng.choice(turn_examples(tokenizer, item, with_tool=True, natural=natural, omit_earlier=omit))


def bounded_no_tool_example(tokenizer, item: dict, rng: random.Random, chars: int,
                            token_limit: int, natural: bool = False) -> tuple[str, str, list[int], list[int]]:
    """Build a direct-answer example whose entire prompt and reply fit the student's exact window.

    A character cap alone cannot guarantee this: code and random strings can use many tokens per
    character. Retry with less filler when the rendered chat example exceeds the token limit.
    """
    if token_limit < 1:
        raise ValueError("no-tool token limit must be positive")
    cap = chars
    while True:
        short = short_item(item, rng, cap)
        prompt, completion = turn_examples(tokenizer, short, with_tool=False, natural=natural)[-1]
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(c_ids) <= token_limit:
            return prompt, completion, p_ids, c_ids
        if cap <= 128:
            raise ValueError(f"no-tool example exceeds the {token_limit}-token window even with minimal filler")
        cap = max(128, cap // 2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--split", default="train", choices=("train", "holdout"))
    ap.add_argument("--min-chars", type=int, default=8000)
    ap.add_argument("--max-chars", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpus", default="~/corpus_v4")
    args = ap.parse_args(argv)
    from .common import load_byte_shards
    bs = load_byte_shards()
    corpus = bs.ShardedByteCorpus(Path(args.corpus).expanduser(), args.split, seed=args.seed)
    rng = random.Random(args.seed)
    tasks = ("needle", "multihop", "earlier", "copy")
    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        while kept < args.n:
            size = int(rng.uniform(args.min_chars, args.max_chars))
            parts, total = [], 0
            while total < size:
                idx = rng.randrange(len(corpus))
                text = corpus.text(idx)
                parts.append(text)
                total += len(text)
            filler = "\n\n".join(parts)[:size]
            item = make_item(tasks[kept % len(tasks)], filler, rng)
            if not item["verified"]:
                dropped += 1
                continue
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1
    print(json.dumps({"kept": kept, "dropped_unverified": dropped, "out": str(out)}))


if __name__ == "__main__":
    main()
