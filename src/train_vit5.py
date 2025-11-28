from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

from .config import get_experiment, ExperimentConfig
from .data import load_tokenized
from .metrics import build_metrics_fn


def build_training_arguments(cfg: ExperimentConfig) -> Seq2SeqTrainingArguments:
    """
    Tạo Seq2SeqTrainingArguments từ ExperimentConfig.
    """
    output_dir = cfg.checkpoint_dir

    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,

        # Hyperparams lấy từ config
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        label_smoothing_factor=cfg.label_smoothing_factor,

        # Logging & saving
        logging_steps=100,
        save_steps=1000,
        eval_strategy="steps",
        eval_steps=1000,
        save_total_limit=3,

        # Generate + metric
        predict_with_generate=True,
        generation_max_length=cfg.max_target_length,
        # Không log lên wandb, tensorboard...
        report_to="none",

        # Mixed precision nếu có GPU
        fp16=cfg.fp16,

        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=cfg.dataloader_pin_memory,

        # Load best model theo metric
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
    )
    return args


def train_vit5(exp_name: str, resume: bool = False):
    """
    Train Vit5 cho 1 experiment (vit5_original hoặc vit5_augmented).
    """
    cfg: ExperimentConfig = get_experiment(exp_name)

    print(f"[INFO] ==== Training for experiment: {cfg.name} ====")
    print(f"[INFO] Model name     : {cfg.model_name}")
    print(f"[INFO] Checkpoint dir : {cfg.checkpoint_dir}")
    print(f"[INFO] Tokenized dir  : {cfg.tokenized_dir}")

    # 1) Load tokenizer & model
    print("[INFO] Loading tokenizer & model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"[INFO] Device: {device}")

    # 2) Load tokenized datasets
    print("[INFO] Loading tokenized datasets...")
    train_ds, dev_ds, test_ds = load_tokenized(exp_name)
    print(f"[INFO] train size: {len(train_ds)}")
    print(f"[INFO] dev   size: {len(dev_ds)}")
    print(f"[INFO] test  size: {len(test_ds)}")

    # 3) Build training arguments
    training_args = build_training_arguments(cfg)

    # 4) Data collator & metrics
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    compute_metrics = build_metrics_fn(tokenizer)

    # 5) Build trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 6) Resume hoặc train mới
    if resume:
        print("[INFO] Resume training from last checkpoint in output_dir...")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("[INFO] Start new training...")
        trainer.train()

    # 7) Đánh giá trên dev (trainer đã làm rồi), mình đánh giá thêm trên test
    print("[INFO] Evaluating on test set...")
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    print("[RESULT] Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    # 8) Lưu model cuối cùng (đã là best model nếu load_best_model_at_end=True)
    final_dir = cfg.checkpoint_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving final model to: {final_dir}")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("[INFO] ==== Done training for experiment:", cfg.name, "====")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Tên experiment: vit5_original | vit5_augmented | hust_original | hust_augmented",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Nếu set, sẽ tiếp tục train từ checkpoint gần nhất trong output_dir",
    )

    args = parser.parse_args()
    train_vit5(args.exp, resume=args.resume)
