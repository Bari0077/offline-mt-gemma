"""Merge a LoRA adapter into Gemma and save a standalone model (for deployment in M3).

Usage: uv run python scripts/merge.py --adapter runs/pilot_50k/final_adapter --out runs/pilot_50k/merged
"""
from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import MODEL_ID


def main(adapter: str, out: str) -> None:
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    merged.to(torch.bfloat16).save_pretrained(out)          # ~540 MB
    AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained(out)
    print(f"Merged model saved to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    main(a.adapter, a.out)