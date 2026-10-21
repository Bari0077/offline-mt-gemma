"""Shared constants and helpers for the offline translator."""
from __future__ import annotations

import unicodedata

import torch

MODEL_ID = "google/gemma-3-270m-it"
LANG_NAMES: dict[str, str] = {"en": "English", "es": "Spanish", "fr": "French"}
FLORES_CODES: dict[str, str] = {"en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn"}
DIRECTIONS: list[tuple[str, str]] = [("en", "es"), ("es", "en"), ("en", "fr"), ("fr", "en")]
OPUS_PAIRS: list[str] = ["en-es", "en-fr"]
PROMPT_TEMPLATE = (
    "Translate the following text from {src} to {tgt}. "
    "Reply with the translation only.\n\n{text}"
)


def normalize(text: str) -> str:
    """NFC-normalize and collapse all whitespace to single spaces."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def build_prompt(text: str, src: str, tgt: str) -> str:
    """The single instruction used for training AND inference (must stay identical)."""
    return PROMPT_TEMPLATE.format(src=LANG_NAMES[src], tgt=LANG_NAMES[tgt], text=text)


def get_device() -> str:
    """Prefer CUDA (Colab), then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"