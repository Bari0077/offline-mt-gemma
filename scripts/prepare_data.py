"""Download OPUS-100 (en-es, en-fr), clean it, and build the four-direction SFT dataset.

Usage:
    uv run python scripts/prepare_data.py --n-per-pair 50000  --out data/sft_50k
    uv run python scripts/prepare_data.py --n-per-pair 250000 --out data/sft_250k
    uv run python scripts/prepare_data.py --out data/sft_full          # all cleaned pairs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset

from common import FLORES_CODES, OPUS_PAIRS, build_prompt, normalize


def load_flores_english() -> set[str]:
    """All English FLORES+ dev+devtest sentences (removed from training data)."""
    eng = load_dataset("openlanguagedata/flores_plus", FLORES_CODES["en"])
    return {normalize(t) for split in ("dev", "devtest") for t in eng[split]["text"]}


def load_pair(pair: str, split: str) -> pd.DataFrame:
    """Load one OPUS-100 split as a two-column DataFrame, e.g. columns ['en', 'es']."""
    a, b = pair.split("-")
    rows = load_dataset("Helsinki-NLP/opus-100", pair, split=split)["translation"]
    return pd.DataFrame({a: [normalize(r[a]) for r in rows],
                         b: [normalize(r[b]) for r in rows]})


def describe(df: pd.DataFrame) -> dict[str, float]:
    """Size and average sentence length per language (for the dataset card table)."""
    stats: dict[str, float] = {"pairs": int(len(df))}
    for col in df.columns:
        stats[f"avg_words_{col}"] = round(float(df[col].str.split().str.len().mean()), 2)
        stats[f"avg_chars_{col}"] = round(float(df[col].str.len().mean()), 2)
    return stats


def clean(df: pd.DataFrame, exclude_en: set[str], max_chars: int,
          max_ratio: float) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the filters one by one and record how many pairs survive each."""
    a, b = df.columns
    counts = {"raw": len(df)}
    df = df[(df[a].str.len() > 0) & (df[b].str.len() > 0)]
    counts["after_empty"] = len(df)
    df = df[df[a] != df[b]]                                    # untranslated copies
    counts["after_identical"] = len(df)
    la, lb = df[a].str.len(), df[b].str.len()
    df = df[(la <= max_chars) & (lb <= max_chars)]
    counts["after_max_length"] = len(df)
    la, lb = df[a].str.len(), df[b].str.len()
    # +10 chars slack so very short pairs ("OK" / "De acuerdo") are not removed
    df = df[np.maximum(la, lb) <= max_ratio * np.minimum(la, lb) + 10]
    counts["after_length_ratio"] = len(df)
    df = df[~df["en"].isin(exclude_en)]
    counts["after_flores_decontamination"] = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    counts["after_dedup"] = len(df)
    return df, counts


def _format(example: dict) -> dict:
    """TRL conversational prompt-completion format -> loss on the translation only."""
    prompt = build_prompt(example["source"], example["src_lang"], example["tgt_lang"])
    return {"prompt": [{"role": "user", "content": prompt}],
            "completion": [{"role": "assistant", "content": example["target"]}]}


def to_sft(df: pd.DataFrame, pair: str) -> Dataset:
    """Each parallel pair becomes two examples: a->b and b->a."""
    a, b = pair.split("-")
    both = pd.concat([
        pd.DataFrame({"src_lang": a, "tgt_lang": b, "source": df[a], "target": df[b]}),
        pd.DataFrame({"src_lang": b, "tgt_lang": a, "source": df[b], "target": df[a]}),
    ], ignore_index=True)
    ds = Dataset.from_pandas(both, preserve_index=False)
    return ds.map(_format, remove_columns=ds.column_names)


def main(args: argparse.Namespace) -> None:
    exclude = load_flores_english()
    stats: dict = {}
    train_parts: list[Dataset] = []
    val_parts: list[Dataset] = []
    for pair in OPUS_PAIRS:
        raw = {s: load_pair(pair, s) for s in ("train", "validation", "test")}
        stats[pair] = {"raw_splits": {s: describe(d) for s, d in raw.items()}}

        train_df, counts = clean(raw["train"], exclude, args.max_chars, args.max_ratio)
        stats[pair]["cleaning_counts_train"] = counts
        if args.n_per_pair:
            train_df = train_df.sample(n=min(args.n_per_pair, len(train_df)),
                                       random_state=args.seed)
        stats[pair]["used_for_training"] = describe(train_df)

        val_df, _ = clean(raw["validation"], exclude, args.max_chars, args.max_ratio)
        val_df = val_df.sample(n=min(args.n_val_per_pair, len(val_df)), random_state=args.seed)

        train_parts.append(to_sft(train_df, pair))
        val_parts.append(to_sft(val_df, pair))
        print(f"{pair}: {json.dumps(counts)}")

    train = concatenate_datasets(train_parts).shuffle(seed=args.seed)  # mix directions
    val = concatenate_datasets(val_parts).shuffle(seed=args.seed)
    out = Path(args.out)
    train.save_to_disk(str(out / "train"))
    val.save_to_disk(str(out / "validation"))
    stats["sft_examples"] = {"train": len(train), "validation": len(val)}
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print("SFT examples:", stats["sft_examples"])
    print("Example:", json.dumps(train[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/sft_full")
    p.add_argument("--n-per-pair", type=int, default=None, help="None = all cleaned pairs")
    p.add_argument("--n-val-per-pair", type=int, default=250)
    p.add_argument("--max-chars", type=int, default=400)
    p.add_argument("--max-ratio", type=float, default=2.5)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())