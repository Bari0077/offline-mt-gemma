"""Evaluate an engine on FLORES+ in all four directions: chrF++, BLEU and speed.

Examples:
  uv run python scripts/run_eval.py --engine marian --name marian --limit 100
  uv run python scripts/run_eval.py --engine gemma --name gemma_zeroshot --limit 100
  uv run python scripts/run_eval.py --engine gemma --adapter runs/pilot_50k/final_adapter --name gemma_pilot
  uv run python scripts/run_eval.py --engine gemma --model runs/medium_250k/merged --name final --split devtest
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from datasets import load_dataset
from sacrebleu.metrics import BLEU, CHRF

from common import DIRECTIONS, FLORES_CODES, MODEL_ID
from engines import Engine, GemmaEngine, MarianEngine


def load_flores(lang: str, split: str) -> list[str]:
    """FLORES+ sentences sorted by id so all languages are aligned."""
    ds = load_dataset("openlanguagedata/flores_plus", FLORES_CODES[lang], split=split).sort("id")
    return list(ds["text"])


def main(args: argparse.Namespace) -> None:
    engine: Engine = (MarianEngine() if args.engine == "marian"
                      else GemmaEngine(args.model, args.adapter))
    chrf, bleu = CHRF(word_order=2), BLEU()
    out_dir = Path("results") / args.name / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for src, tgt in DIRECTIONS:
        srcs, refs = load_flores(src, args.split), load_flores(tgt, args.split)
        assert len(srcs) == len(refs), f"FLORES+ {src}/{tgt} not aligned"
        if args.limit:
            srcs, refs = srcs[:args.limit], refs[:args.limit]

        t0 = time.perf_counter()
        hyps = engine.translate(srcs, src, tgt, batch_size=args.batch_size)
        elapsed = time.perf_counter() - t0

        row = {
            "direction": f"{src}-{tgt}",
            "n": len(srcs),
            "chrF++": round(chrf.corpus_score(hyps, [refs]).score, 2),
            "BLEU": round(bleu.corpus_score(hyps, [refs]).score, 2),
            "ms_per_sentence_batched": round(1000 * elapsed / len(srcs), 1),
        }
        rows.append(row)
        (out_dir / f"{src}-{tgt}.hyp.txt").write_text("\n".join(hyps) + "\n", encoding="utf-8")

        print(f"\n=== {src}->{tgt}: {row}")
        for s, h, r in list(zip(srcs, hyps, refs))[:3]:
            print(f"  SRC: {s}\n  HYP: {h}\n  REF: {r}\n")

    summary = {
        "name": args.name,
        "split": args.split,
        "results": rows,
        "chrF_signature": str(chrf.get_signature()),
        "BLEU_signature": str(bleu.get_signature()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    cols = list(rows[0].keys())
    print("\n| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")
    print(f"\nchrF++ signature: {summary['chrF_signature']}")
    print(f"BLEU signature:   {summary['BLEU_signature']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["gemma", "marian"], default="gemma")
    p.add_argument("--model", default=MODEL_ID, help="HF id or path to a merged model")
    p.add_argument("--adapter", default=None, help="LoRA adapter or checkpoint dir")
    p.add_argument("--name", required=True, help="label for the results folder")
    p.add_argument("--split", choices=["dev", "devtest"], default="dev")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    main(p.parse_args())