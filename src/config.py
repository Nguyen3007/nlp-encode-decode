from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


# =============================
# 1. ĐƯỜNG DẪN CƠ BẢN CỦA PROJECT
# =============================

# Thư mục gốc: nlp-encode-decode/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# data/, checkpoints/
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TOKENIZED_DIR = DATA_DIR / "tokenized"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

TRAIN_ORIG_FILE = RAW_DIR / "train_original.csv"
TRAIN_AUG_FILE = RAW_DIR / "train_augmented.csv"
DEV_FILE = RAW_DIR / "dev.csv"
TEST_FILE = RAW_DIR / "test.csv"


# =============================
# 2. CẤU HÌNH CHO 1 EXPERIMENT
# =============================

@dataclass
class ExperimentConfig:
    # Tên ngắn của experiment, ví dụ: "vit5_original"
    name: str

    # Tên checkpoint model trên HuggingFace
    model_name: str

    # File raw
    train_file: Path
    dev_file: Path
    test_file: Path

    # Nơi lưu checkpoint
    checkpoint_dir: Path

    # Nơi lưu dataset đã token hóa (cho experiment này)
    # Ví dụ: data/tokenized/vit5/vit5_original/
    tokenized_dir: Path

    # Hyperparameters cơ bản (có thể chỉnh sau)
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    label_smoothing_factor: float = 0.1

    dataloader_num_workers: int = 2
    dataloader_pin_memory: bool = True
    fp16: bool = True

    max_source_length: int = 128
    max_target_length: int = 128


# =============================
# 3. TÊN MODEL TRÊN HUGGINGFACE
# =============================

# Vit5 base của VietAI
VIT5_MODEL_NAME = "VietAI/vit5-base"

# Model nhỏ của HUST
HUST_SMALL_MODEL_NAME = "NlpHUST/t5-en-vi-small"


# =============================
# 4. ĐỊNH NGHĨA 4 EXPERIMENT
# =============================

EXPERIMENTS: dict[str, ExperimentConfig] = {
    # 1) Vit5-base + train gốc
    "vit5_original": ExperimentConfig(
        name="vit5_original",
        model_name=VIT5_MODEL_NAME,
        train_file=TRAIN_ORIG_FILE,
        dev_file=DEV_FILE,
        test_file=TEST_FILE,
        checkpoint_dir=CHECKPOINTS_DIR / "vit5_original",
        tokenized_dir=TOKENIZED_DIR / "vit5" / "vit5_original",
    ),

    # 2) Vit5-base + train augmented
    "vit5_augmented": ExperimentConfig(
        name="vit5_augmented",
        model_name=VIT5_MODEL_NAME,
        train_file=TRAIN_AUG_FILE,
        dev_file=DEV_FILE,
        test_file=TEST_FILE,
        checkpoint_dir=CHECKPOINTS_DIR / "vit5_augmented",
        tokenized_dir=TOKENIZED_DIR / "vit5" / "vit5_augmented",
    ),

    # 3) HUST small + train gốc
    "hust_original": ExperimentConfig(
        name="hust_original",
        model_name=HUST_SMALL_MODEL_NAME,
        train_file=TRAIN_ORIG_FILE,
        dev_file=DEV_FILE,
        test_file=TEST_FILE,
        checkpoint_dir=CHECKPOINTS_DIR / "hust_original",
        tokenized_dir=TOKENIZED_DIR / "hust" / "hust_original",

    # --- override hyperparams so với default ---
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        label_smoothing_factor=0.1,
        dataloader_num_workers=8,  # Tận dụng CPU nhiều core
        dataloader_pin_memory=True,
        fp16=False,
        max_source_length=128,
        max_target_length=128,
    ),

    # 4) HUST small + train augmented
    "hust_augmented": ExperimentConfig(
        name="hust_augmented",
        model_name=HUST_SMALL_MODEL_NAME,
        train_file=TRAIN_AUG_FILE,
        dev_file=DEV_FILE,
        test_file=TEST_FILE,
        checkpoint_dir=CHECKPOINTS_DIR / "hust_augmented",
        tokenized_dir=TOKENIZED_DIR / "hust" / "hust_augmented",

        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        label_smoothing_factor=0.1,
        dataloader_num_workers=8,  # Tận dụng CPU nhiều core
        dataloader_pin_memory=True,
        fp16=False,
        max_source_length=128,
        max_target_length=128,
    ),
}


def get_experiment(name: str) -> ExperimentConfig:
    """
    Lấy config theo tên experiment.

    Ví dụ:
        from src.config import get_experiment
        cfg = get_experiment("vit5_original")
    """
    if name not in EXPERIMENTS:
        valid = ", ".join(EXPERIMENTS.keys())
        raise ValueError(f"Unknown experiment '{name}'. Valid: {valid}")
    return EXPERIMENTS[name]
