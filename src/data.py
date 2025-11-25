from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

from .config import get_experiment, ExperimentConfig


# =========================================================
# 1. ĐỌC CSV VÀ CHUẨN HÓA CỘT en / vi -> source / target
# =========================================================

def load_raw_dataframe(path: Path) -> pd.DataFrame:
    """
    Đọc 1 file CSV.
    File train_original, dev, test: chỉ có 2 cột en, vi.
    File train_augmented: có thêm en_len, vi_len, ratio,... -> sẽ bỏ.
    """
    df = pd.read_csv(path)

    if "en" not in df.columns or "vi" not in df.columns:
        raise ValueError(
            f"Expected columns 'en' and 'vi' in {path}, "
            f"but got columns: {list(df.columns)}"
        )

    df = df[["en", "vi"]].copy()
    df.rename(columns={"en": "source", "vi": "target"}, inplace=True)
    return df


def load_raw_datasets(cfg: ExperimentConfig) -> Dict[str, Dataset]:
    """
    Đọc 3 file CSV (train / dev / test) dựa trên config,
    trả về 3 Dataset HuggingFace, mỗi cái có 2 cột:
        - source
        - target
    """
    print(f"[INFO] Loading raw data for experiment: {cfg.name}")
    print(f"  train: {cfg.train_file}")
    print(f"  dev  : {cfg.dev_file}")
    print(f"  test : {cfg.test_file}")

    dfs = {
        "train": load_raw_dataframe(cfg.train_file),
        "dev": load_raw_dataframe(cfg.dev_file),
        "test": load_raw_dataframe(cfg.test_file),
    }

    datasets = {
        split: Dataset.from_pandas(df, preserve_index=False)
        for split, df in dfs.items()
    }

    for split, ds in datasets.items():
        print(f"[INFO] {split} size: {len(ds)}")

    return datasets


# =========================================================
# 2. TOKENIZER & HÀM TOKENIZE
# =========================================================

def get_tokenizer(cfg: ExperimentConfig):
    """
    Lấy tokenizer tương ứng với model của experiment.
    """
    print(f"[INFO] Loading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    return tokenizer


def tokenize_function(examples, tokenizer, cfg: ExperimentConfig):
    # 1. Chuẩn bị prefix:
    # Vit5 & HUST T5 đều hoạt động tốt với prefix
    inputs = ["en: " + str(s) for s in examples["source"]]
    targets = ["vi: " + str(t) for t in examples["target"]]

    # 2. Tokenize input
    model_inputs = tokenizer(
        inputs,
        max_length=cfg.max_source_length,
        padding="max_length",
        truncation=True,
    )

    # 3. Tokenize target (chuẩn mới)
    labels_raw = tokenizer(
        text_target=targets,
        max_length=cfg.max_target_length,
        padding="max_length",
        truncation=True,
    )["input_ids"]

    # 4. Thay pad_token_id = -100 để mô hình bỏ qua khi tính loss
    labels = [
        [-100 if token == tokenizer.pad_token_id else token for token in seq]
        for seq in labels_raw
    ]

    model_inputs["labels"] = labels
    return model_inputs



# =========================================================
# 3. CHUẨN BỊ & LƯU DATA ĐÃ TOKENIZE
# =========================================================

def prepare_and_save_tokenized(exp_name: str):
    """
    Pipeline:
      1) Lấy config experiment
      2) Đọc raw CSV -> Dataset (source, target)
      3) Tokenize train/dev/test
      4) Lưu xuống data/tokenized/<model>/<exp_name>/{train,dev,test}
    """
    cfg: ExperimentConfig = get_experiment(exp_name)
    print(f"[INFO] ==== Prepare tokenized datasets for: {cfg.name} ====")

    # Tạo thư mục gốc cho tokenized nếu chưa tồn tại
    cfg.tokenized_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Tokenized root dir: {cfg.tokenized_dir}")

    # 1) Load raw datasets
    raw_datasets = load_raw_datasets(cfg)

    # 2) Load tokenizer
    tokenizer = get_tokenizer(cfg)

    # 3) Tokenize cho từng split
    for split in ["train", "dev", "test"]:
        out_dir = cfg.tokenized_dir / split
        if out_dir.exists():
            print(f"[INFO] Tokenized {split} already exists at {out_dir}, skip.")
            continue

        ds = raw_datasets[split]
        print(f"[INFO] Tokenizing {split} set, {len(ds)} examples...")

        tokenized_ds = ds.map(
            lambda batch: tokenize_function(batch, tokenizer, cfg),
            batched=True,
            remove_columns=ds.column_names,  # bỏ cột source/target text, chỉ giữ ids
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        tokenized_ds.save_to_disk(out_dir)
        print(f"[INFO] Saved tokenized {split} to {out_dir}")

    print(f"[INFO] ==== Done prepare tokenized for: {cfg.name} ====")


def load_tokenized(exp_name: str) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load lại 3 dataset đã tokenized cho 1 experiment.
    Trả về: train_ds, dev_ds, test_ds
    """
    cfg = get_experiment(exp_name)
    base = cfg.tokenized_dir

    train_ds = load_from_disk(base / "train")
    dev_ds = load_from_disk(base / "dev")
    test_ds = load_from_disk(base / "test")

    return train_ds, dev_ds, test_ds


# =========================================================
# 4. CHẠY TRỰC TIẾP FILE NÀY ĐỂ TẠO TOKENIZED DATA
# =========================================================

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

    prepare_and_save_tokenized(args.exp)
