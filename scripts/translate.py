"""Translate one sentence from the terminal.

uv run python scripts/translate.py --src en --tgt es "The clinic opens at nine tomorrow."
uv run python scripts/translate.py --adapter runs/pilot_50k/final_adapter --src es --tgt en "¿Dónde está la farmacia?"
"""
from __future__ import annotations

import argparse

from common import MODEL_ID
from engines import GemmaEngine, MarianEngine

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("text")
    p.add_argument("--src", required=True, choices=["en", "es", "fr"])
    p.add_argument("--tgt", required=True, choices=["en", "es", "fr"])
    p.add_argument("--engine", choices=["gemma", "marian"], default="gemma")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--adapter", default=None)
    a = p.parse_args()
    engine = MarianEngine() if a.engine == "marian" else GemmaEngine(a.model, a.adapter)
    print(engine.translate([a.text], a.src, a.tgt)[0])