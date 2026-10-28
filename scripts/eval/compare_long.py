"""Paired long-context comparison of a student against its teacher, per position bucket.

    python -m scripts.transplant.compare_long --teacher ~/x2/eval/q08-teacher \\
        --student ~/x2/eval/q08-swagdn256 [--student ...] [--task longnll32]

Reads evaluate.py results (task longnll or longnll32, which store the mean NLL of every
1024-token block of every row). Both models scored the same rows, so the comparison is paired:
for each bucket, the mean of (student - teacher) over its blocks, a 95% interval from a
bootstrap over rows (blocks of one row move together, so rows are the resampling unit), and
the teacher's own NLL there. Prints a table and returns it as JSON. Mismatched rows,
block lengths or token-row fingerprints are rejected. Older results without a
fingerprint are explicitly marked as having unverified paired data identity.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

BUCKETS = {"0-2k": (0, 2), "2k-8k": (2, 8), "8k-16k": (8, 16), "16k-32k": (16, 32)}  # in 1024-token blocks


def validate_block_rows(teacher_blocks, student_blocks):
    """An incomplete or differently shaped run cannot support a paired row bootstrap."""
    if len(teacher_blocks) != len(student_blocks):
        raise ValueError("long-NLL row counts differ")
    if any(len(t) != len(s) for t, s in zip(teacher_blocks, student_blocks)):
        raise ValueError("long-NLL block counts differ within a row")


def pairing_status(teacher, student):
    """Report verified token-row identity, or explain why a pairing is unverified/invalid."""
    for key in ("docs", "tokens"):
        if key in teacher and key in student and teacher[key] != student[key]:
            raise ValueError(f"long-NLL {key} differ")
    if "block_nll_1024" in teacher and "block_nll_1024" in student:
        validate_block_rows(teacher["block_nll_1024"], student["block_nll_1024"])
        for result in (teacher, student):
            blocks = result["block_nll_1024"]
            if "docs" in result and len(blocks) != result["docs"]:
                raise ValueError("long-NLL rows do not match declared document count")
            if "tokens" in result:
                expected = (result["tokens"] - 1 + 1023) // 1024
                if any(len(row) != expected for row in blocks):
                    raise ValueError("long-NLL blocks do not cover declared token length")
    th, sh = teacher.get("data_sha256"), student.get("data_sha256")
    if th and sh:
        if th != sh:
            raise ValueError("long-NLL token-row hashes differ")
        if teacher.get("data_identity") != student.get("data_identity"):
            raise ValueError("long-NLL fingerprint formats differ")
        if teacher.get("data_identity") == "token-rows-json-v1":
            return {"pairing_verified": True}
        return {"pairing_verified": False,
                "pairing_warning": "token-row fingerprint format unrecognized; paired data identity unverified"}
    return {"pairing_verified": False,
            "pairing_warning": "token-row fingerprint missing; paired data identity unverified"}


def paired(teacher_blocks, student_blocks, lo, hi, draws=2000, seed=0):
    validate_block_rows(teacher_blocks, student_blocks)
    rows = []
    for t, s in zip(teacher_blocks, student_blocks):
        diffs = [b - a for a, b in zip(t[lo:hi], s[lo:hi])]
        if diffs:
            rows.append(diffs)
    if not rows:
        return None
    flat = [d for r in rows for d in r]
    mean = sum(flat) / len(flat)
    rng = random.Random(seed)
    boot = []
    for _ in range(draws):
        pick = [rows[rng.randrange(len(rows))] for _ in rows]
        values = [d for r in pick for d in r]
        boot.append(sum(values) / len(values))
    boot.sort()
    return mean, boot[int(0.025 * draws)], boot[int(0.975 * draws) - 1], len(rows)


def paired_split(teacher_rows, student_rows, bucket, part, draws=2000, seed=0):
    """Per-token gap (student - teacher) on the far-copy tokens (part 0) or the rest (part 1) of a
    bucket: sum over rows of the NLL differences / token count, 95% interval from a row bootstrap.
    Returns (gap, low, high, share of the bucket's tokens in this part) or None."""
    if len(teacher_rows) != len(student_rows):
        raise ValueError("far-copy row counts differ")
    for t, s in zip(teacher_rows, student_rows):
        if (bucket in t) != (bucket in s):
            raise ValueError(f"far-copy bucket {bucket} missing in one run")
        if bucket in t and (t[bucket][1], t[bucket][3]) != (s[bucket][1], s[bucket][3]):
            raise ValueError(f"far-copy token counts differ in {bucket}")
    i = 2 * part
    rows = [(s[bucket][i] - t[bucket][i], t[bucket][i + 1], t[bucket][1] + t[bucket][3])
            for t, s in zip(teacher_rows, student_rows) if bucket in t and t[bucket][i + 1]]
    if not rows:
        return None

    def ratio(pick):
        return sum(r[0] for r in pick) / sum(r[1] for r in pick)

    rng = random.Random(seed)
    boot = sorted(ratio([rows[rng.randrange(len(rows))] for _ in rows]) for _ in range(draws))
    share = sum(r[1] for r in rows) / sum(r[2] for r in rows)
    return ratio(rows), boot[int(0.025 * draws)], boot[int(0.975 * draws) - 1], share


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--student", type=Path, action="append", required=True)
    ap.add_argument("--task", default="longnll32")
    args = ap.parse_args(argv)

    def load(path):
        return json.loads((path.expanduser() / "results.json").read_text(encoding="utf-8"))["tasks"][args.task]

    teacher = load(args.teacher)
    report = {}
    print(f"{'model':28s} " + " ".join(f"{b:>24s}" for b in BUCKETS))
    print(f"{'teacher NLL':28s} " + " ".join(f"{teacher.get(b, float('nan')):>24.3f}" for b in BUCKETS))
    for path in args.student:
        student = load(path)
        try:
            status = pairing_status(teacher, student)
        except ValueError as exc:
            ap.error(f"{path.name}: {exc}; refusing a paired comparison")
        if not status["pairing_verified"]:
            print(f"{path.name}: {status['pairing_warning']}")
        row = {}
        for bucket, (lo, hi) in BUCKETS.items():
            result = paired(teacher["block_nll_1024"], student["block_nll_1024"], lo, hi)
            if result:
                mean, low, high, n = result
                row[bucket] = {"gap": mean, "ci95": [low, high], "rows": n,
                               "teacher": teacher.get(bucket), **status}
        report[path.name] = row
        cells = [f"{row[b]['gap']:+.3f} [{row[b]['ci95'][0]:+.3f},{row[b]['ci95'][1]:+.3f}]" if b in row else "-"
                 for b in BUCKETS]
        print(f"{path.name:28s} " + " ".join(f"{c:>24s}" for c in cells))
        if "far_copy_split" in teacher and "far_copy_split" in student:  # verbatim repeats from beyond 1024
            t_rows, s_rows = teacher["far_copy_split"]["rows"], student["far_copy_split"]["rows"]
            for part, label in ((0, "  far-copy tokens"), (1, "  other tokens")):
                cells = []
                for b in BUCKETS:
                    result = paired_split(t_rows, s_rows, b, part)
                    if result is None or b not in row:
                        cells.append("-")
                        continue
                    gap, low, high, share = result
                    row[b]["far_copy" if part == 0 else "other"] = {"gap": gap, "ci95": [low, high], "share": share}
                    cells.append(f"{gap:+.3f} [{low:+.3f},{high:+.3f}] {100 * share:.0f}%")
                print(f"{label:28s} " + " ".join(f"{c:>24s}" for c in cells))
    print(json.dumps(report))
    return report


if __name__ == "__main__":
    main()
