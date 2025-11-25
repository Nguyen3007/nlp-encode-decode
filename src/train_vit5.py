from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from .config import get_experiment, ExperimentConfig
from .data import load_tokenized
from .metrics import build_metrics_fn


# =========================================================
# 1. TrainingArguments cho ViT5
# =========================================================

def build_training_arguments(cfg: ExperimentConfig) -> Seq2SeqTrainingArguments:
    """
    Tạo Seq2SeqTrainingArguments từ ExperimentConfig.
    Ở đây đã bật luôn TensorBoard + load best model.
    """

    output_dir = cfg.checkpoint_dir
    tb_log_dir = output_dir / "logs"

    args = Seq2SeqTrainingArguments(
        # ---- Đường dẫn checkpoint ----
        output_dir=str(output_dir),
        overwrite_output_dir=False,

        # ---- Epoch & batch ----
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=getattr(cfg, "per_device_eval_batch_size",
                                           cfg.per_device_train_batch_size),
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,

        # ---- Scheduler / warmup ----
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,

        # ---- Label smoothing ----
        label_smoothing_factor=cfg.label_smoothing_factor,

        # ---- Evaluation / saving ----
        eval_strategy="steps",            # <== API mới (không dùng evaluation_strategy nữa)
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,

        # ---- Logging + TensorBoard ----
        logging_strategy="steps",
        logging_steps=100,
        logging_dir=str(tb_log_dir),
        report_to=["tensorboard"],        # bật TensorBoard

        # ---- Generate khi eval ----
        predict_with_generate=True,

        # ---- Khác ----
        load_best_model_at_end=True,
        metric_for_best_model="bleu",     # lấy checkpoint BLEU cao nhất
        greater_is_better=True,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,

        # Mixed precision: chỉ bật khi có GPU
        fp16=torch.cuda.is_available(),
    )

    return args


# =========================================================
# 2. Hàm train chính cho ViT5
# =========================================================

def train_vit5(exp_name: str, resume: bool = True):
    """
    Train ViT5 cho 1 experiment (vit5_original hoặc vit5_augmented).
    """
    cfg: ExperimentConfig = get_experiment(exp_name)

    print(f"[INFO] ==== Training ViT5 for experiment: {cfg.name} ====")
    print(f"[INFO] Model name     : {cfg.model_name}")
    print(f"[INFO] Checkpoint dir : {cfg.checkpoint_dir}")
    print(f"[INFO] Tokenized dir  : {cfg.tokenized_dir}")

    # 1) Thiết lập thiết bị
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    # 2) Load tokenizer + model
    print("[INFO] Loading tokenizer & model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)
    model.to(device)

    # 3) Load tokenized datasets
    print("[INFO] Loading tokenized datasets...")
    train_ds, dev_ds, test_ds = load_tokenized(exp_name)
    print(f"[INFO] train size: {len(train_ds)}")
    print(f"[INFO] dev   size: {len(dev_ds)}")
    print(f"[INFO] test  size: {len(test_ds)}")

    # 4) Data collator & metrics
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
    )

    compute_metrics = build_metrics_fn(tokenizer)

    # 5) Training arguments (có TensorBoard)
    training_args = build_training_arguments(cfg)

    # 6) Tìm checkpoint gần nhất nếu resume=True
    resume_checkpoint: Optional[str] = None
    if resume and cfg.checkpoint_dir.exists():
        ckpts = [
            d for d in cfg.checkpoint_dir.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
        ]
        if ckpts:
            latest = sorted(
                ckpts,
                key=lambda p: int(p.name.split("-")[1])
            )[-1]
            resume_checkpoint = str(latest)
            print(f"[INFO] Resume from checkpoint: {resume_checkpoint}")
        else:
            print("[INFO] No checkpoint found, start new training.")
    else:
        print("[INFO] Start new training.")

    # 7) Tạo Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,   # dùng API mới, tránh warning
    )

    # 8) Bắt đầu train
    print("[INFO] ====== START TRAINING ======")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # 9) Đánh giá trên test set
    print("[INFO] ====== EVALUATE ON TEST SET ======")
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    print("[INFO] Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    # 10) Lưu best model (sau khi load_best_model_at_end)
    best_dir = cfg.checkpoint_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"[INFO] Saved best model to: {best_dir}")


# =========================================================
# 3. Chạy từ command line
# =========================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="vit5_original | vit5_augmented",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Không resume từ checkpoint cũ, luôn train mới từ đầu.",
    )
    args = parser.parse_args()

    train_vit5(args.exp, resume=not args.no_resume)
