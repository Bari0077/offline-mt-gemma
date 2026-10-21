"""Translation engines: Gemma (base / base+adapter / merged) and OPUS-MT Marian specialists."""
from __future__ import annotations

from typing import Protocol

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from common import MODEL_ID, build_prompt, get_device


class Engine(Protocol):
    def translate(self, texts: list[str], src: str, tgt: str, batch_size: int = 16) -> list[str]: ...


class GemmaEngine:
    """Decoder-only translator using the exact chat prompt from training."""

    def __init__(self, model_path: str = MODEL_ID, adapter: str | None = None,
                 device: str | None = None, max_new_tokens: int = 256) -> None:
        self.device = device or get_device()
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "left"                 # required for batched generation
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
        if adapter:
            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        self.model = model.to(self.device).eval()

    @torch.inference_mode()
    def translate(self, texts: list[str], src: str, tgt: str, batch_size: int = 16) -> list[str]:
        outputs: list[str] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": build_prompt(t, src, tgt)}],
                    tokenize=False, add_generation_prompt=True)
                for t in batch
            ]
            enc = self.tokenizer(prompts, return_tensors="pt", padding=True,
                                 add_special_tokens=False).to(self.device)  # template adds <bos>
            generated = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                            do_sample=False)                 # greedy
            new_tokens = generated[:, enc["input_ids"].shape[1]:]
            outputs.extend(self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        return [" ".join(o.split()) for o in outputs]


class MarianEngine:
    """One OPUS-MT model per direction, loaded on demand (one in memory at a time)."""

    def __init__(self, device: str | None = None, num_beams: int = 4) -> None:
        self.device = device or get_device()
        self.num_beams = num_beams
        self._loaded: dict[str, tuple] = {}

    def _get(self, src: str, tgt: str) -> tuple:
        name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        if name not in self._loaded:
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSeq2SeqLM.from_pretrained(name).to(self.device).eval()
            self._loaded = {name: (tok, model)}
        return self._loaded[name]

    @torch.inference_mode()
    def translate(self, texts: list[str], src: str, tgt: str, batch_size: int = 16) -> list[str]:
        tok, model = self._get(src, tgt)
        outputs: list[str] = []
        for start in range(0, len(texts), batch_size):
            enc = tok(texts[start:start + batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=512).to(self.device)
            generated = model.generate(**enc, num_beams=self.num_beams, max_new_tokens=256)
            outputs.extend(tok.batch_decode(generated, skip_special_tokens=True))
        return [o.strip() for o in outputs]