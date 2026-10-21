"""LoRA fine-tuning of Gemma 3 270M IT for en<->es and en<->fr (one adapter, four directions).

Smoke test: uv run python scripts/train.py --data-dir data/sft_50k --out-dir runs/smoke --max-steps 30 --eval-steps 15
Pilot:      uv run python scripts/train.py --data-dir data/sft_50k --out-dir runs/pilot_50k --eval-steps 500
Medium:     uv run python scripts/train.py --data-dir data/sft_250k --out-dir runs/medium_250k --eval-steps 2000
Resume:     same command + --resume
"""
from __future__ import annotations

import os

# Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")       # CPU fallback for missing MPS ops
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "gemma270m-translation")

import argparse
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

from common import MODEL_ID, get_device


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device()
    use_bf16 = args.bf16 or (device == "cuda" and torch.cuda.is_bf16_supported())
    print(f"device={device}  bf16={use_bf16}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16 if use_bf16 else torch.float32,
        attn_implementation="eager",          # recommended for training Gemma 3
    )

    train_ds = load_from_disk(f"{args.data_dir}/train")
    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    val_ds = load_from_disk(f"{args.data_dir}/validation")
    print(f"train={len(train_ds):,}  validation={len(val_ds):,}")

    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.05,
        target_modules="all-linear",          # q,k,v,o,gate,up,down in every layer
        task_type="CAUSAL_LM",
    )

    config = SFTConfig(
        output_dir=args.out_dir,
        run_name=Path(args.out_dir).name,
        seed=args.seed,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,             # -1 = use epochs
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        max_length=256,                       # tokens; prompt + translation
        bf16=use_bf16,
        gradient_checkpointing=False,         # small model: speed over memory
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        logging_steps=25,
        report_to="mlflow",
        dataloader_pin_memory=(device == "cuda"),
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.model.print_trainable_parameters()
    trainer.train(resume_from_checkpoint=True if args.resume else None)

    final_dir = Path(args.out_dir) / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved adapter to {final_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/sft_50k")
    p.add_argument("--out-dir", default="runs/pilot_50k")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=2)      # effective batch = 32
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--eval-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true", help="try on macOS 14+; faster if it works")
    p.add_argument("--resume", action="store_true")
    main(p.parse_args())