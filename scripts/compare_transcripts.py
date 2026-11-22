#!/usr/bin/env python3
"""Compare a rust JSON dump against transformers-fp32 gold (text + word timings)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def stats(xs):
    if not xs:
        return {"n": 0}
    xs = sorted(xs)
    return {
        "n": len(xs),
        "mean_ms": sum(xs) / len(xs),
        "p50_ms": xs[len(xs) // 2],
        "p95_ms": xs[min(len(xs) - 1, int(len(xs) * 0.95))],
        "max_ms": xs[-1],
    }


def compare(gold, rust, soft_ms=20.0):
    gt, rt = gold.get("text") or "", rust.get("text") or ""
    out = {
        "text_exact": gt == rt,
        "gold_chars": len(gt),
        "rust_chars": len(rt),
        "gold_text_words": len(gt.split()),
        "rust_text_words": len(rt.split()),
    }
    if gt != rt:
        gw, rw = gt.split(), rt.split()
        i = 0
        while i < len(gw) and i < len(rw) and gw[i] == rw[i]:
            i += 1
        out["common_prefix_words"] = i
        out["first_mismatch_gold"] = gw[max(0, i - 3) : i + 6]
        out["first_mismatch_rust"] = rw[max(0, i - 3) : i + 6]
    gw, rw = gold.get("words") or [], rust.get("words") or []
    out["gold_timed_words"] = len(gw)
    out["rust_timed_words"] = len(rw)
    n = min(len(gw), len(rw))
    start_abs, end_abs = [], []
    over = 0
    text_mis = 0
    examples = []
    for i in range(n):
        if gw[i].get("word") != rw[i].get("word"):
            text_mis += 1
            if len(examples) < 8:
                examples.append({"i": i, "gold": gw[i], "rust": rw[i], "kind": "text"})
            continue
        gs, ge, rs, re = gw[i].get("start"), gw[i].get("end"), rw[i].get("start"), rw[i].get("end")
        if gs is None or rs is None or ge is None or re is None:
            continue
        ds, de = abs(gs - rs) * 1000, abs(ge - re) * 1000
        start_abs.append(ds)
        end_abs.append(de)
        if ds > soft_ms or de > soft_ms:
            over += 1
            if len(examples) < 8:
                examples.append({"i": i, "word": gw[i]["word"], "gold": (gs, ge), "rust": (rs, re),
                                 "dstart_ms": ds, "dend_ms": de, "kind": "timing"})
    out["start"] = stats(start_abs)
    out["end"] = stats(end_abs)
    out["words_over_soft_ms"] = over
    out["aligned_word_text_mismatches"] = text_mis
    out["examples"] = examples
    out["pass_text"] = bool(out["text_exact"])
    out["pass_timestamps"] = over == 0 and text_mis == 0 and len(gw) == len(rw)
    out["pass"] = out["pass_text"] and out["pass_timestamps"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--rust", required=True)
    ap.add_argument("--soft-ms", type=float, default=20.0)
    args = ap.parse_args()
    gold = json.loads(Path(args.gold).read_text())
    rust = json.loads(Path(args.rust).read_text())
    # rust short-form gold files may nest words at top level already
    if "words" not in gold and isinstance(gold, list):
        gold = {"text": "", "words": gold}
    r = compare(gold, rust, args.soft_ms)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
