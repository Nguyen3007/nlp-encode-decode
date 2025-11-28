from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer
from src.config import get_experiment
from src.data import load_tokenized
from src.metrics import build_metrics_fn

def run_eval(exp_name: str):
    cfg = get_experiment(exp_name)

    print(f"[INFO] Loading model from: {cfg.checkpoint_dir / 'final'}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint_dir / "final")
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.checkpoint_dir / "final")

    train_ds, dev_ds, test_ds = load_tokenized(exp_name)

    compute_metrics = build_metrics_fn(tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("[INFO] Running eval on TEST...")
    metrics = trainer.evaluate(test_ds, metric_key_prefix="test")

    print(metrics)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True)
    args = parser.parse_args()

    run_eval(args.exp)
