from evaluate import load
import numpy as np


def build_metrics_fn(tokenizer):
    """
    Hàm bao đóng (Closure) để truyền tokenizer vào hàm compute_metrics.
    """
    # Load metrics một lần ở ngoài để đỡ tốn RAM load lại nhiều lần
    bleu = load("bleu")
    rouge = load("rouge")

    pad_id = tokenizer.pad_token_id
    # QUAN TRỌNG: Dùng len(tokenizer) để bao gồm cả special tokens
    max_vocab_index = len(tokenizer) - 1

    def safe_decode(batch_ids):
        """Giải mã token IDs thành chữ, xử lý lỗi out-of-range."""
        safe_batch = []
        for seq in batch_ids:
            # Đảm bảo ID là int
            ids = [int(x) for x in seq]

            # Thay -100 (ignore index) bằng pad_id
            ids = [pad_id if x == -100 else x for x in ids]

            # KỸ THUẬT CLIP: Ép index về vùng an toàn [0, max_vocab_index]
            # Bất chấp model sinh ra số gì, cũng không bao giờ crash
            ids = [pad_id if (x < 0 or x > max_vocab_index) else x for x in ids]

            safe_batch.append(ids)

        return tokenizer.batch_decode(safe_batch, skip_special_tokens=True)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred

        # 1. Decode an toàn
        pred_str = safe_decode(preds)
        label_str = safe_decode(labels)

        # 2. Hậu xử lý (Post-processing): Bỏ prefix "vi: " nếu có
        pred_str = [str(t).replace("vi:", "").strip() for t in pred_str]
        label_str = [str(t).replace("vi:", "").strip() for t in label_str]

        # 3. Tính BLEU
        # BLEU yêu cầu reference là list of list: [['câu 1'], ['câu 2']]
        bleu_score = bleu.compute(
            predictions=pred_str,
            references=[[ref] for ref in label_str],
            max_order=4
        )

        # 4. Tính ROUGE
        rouge_score = rouge.compute(
            predictions=pred_str,
            references=label_str
        )

        return {
            "bleu": bleu_score["bleu"],
            "rouge1": rouge_score["rouge1"],
            "rouge2": rouge_score["rouge2"],
            "rougeL": rouge_score["rougeL"]
        }

    return compute_metrics