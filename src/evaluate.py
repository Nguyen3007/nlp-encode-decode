from __future__ import annotations

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

from .config import get_experiment, ExperimentConfig
from .data import load_tokenized
from .metrics import build_metrics_fn
from .train_vit5 import build_training_arguments


def run_eval(exp_name: str):
    """
    Đánh giá lại 1 experiment trên tập test, dùng đúng:
      - config (ExperimentConfig)
      - tokenization
      - metrics (BLEU, ROUGE)
      - Seq2SeqTrainingArguments (giống train_vit5)
    """
    cfg: ExperimentConfig = get_experiment(exp_name)

    model_dir = cfg.checkpoint_dir / "final"
    print(f"[INFO] Loading model from: {model_dir}")

    # 1) Load tokenizer & model từ thư mục final
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"[INFO] Device: {device}")

    # 2) Load lại tokenized datasets
    print("[INFO] Loading tokenized datasets...")
    train_ds, dev_ds, test_ds = load_tokenized(exp_name)
    print(f"[INFO] train size: {len(train_ds)}")
    print(f"[INFO] dev   size: {len(dev_ds)}")
    print(f"[INFO] test  size: {len(test_ds)}")

    # 3) Build training/eval arguments (dùng lại của train_vit5)
    training_args = build_training_arguments(cfg)

    # 4) Data collator & metrics
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    compute_metrics = build_metrics_fn(tokenizer)

    # 5) Seq2SeqTrainer chỉ dùng cho evaluate
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=test_ds,
        data_collator=data_collator,
        processing_class=tokenizer,  # tránh FutureWarning
        compute_metrics=compute_metrics,
    )

    # 6) Evaluate trên test
    print("[INFO] Running evaluation on TEST set...")
    metrics = trainer.evaluate(test_ds, metric_key_prefix="test")

    print("\n[RESULT] Test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Tên experiment: vit5_original | vit5_augmented | hust_original | hust_augmented",
    )
    args = parser.parse_args()

    run_eval(args.exp)
