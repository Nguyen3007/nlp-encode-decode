# Vietnamese Machine Translation — Encoder–Decoder Experiments

Fine-tuning ViT5 & HUST T5 (Seq2Seq) trên tập IWSLT (English → Vietnamese)

Dự án này triển khai 4 thí nghiệm fine-tuning mô hình dịch máy encoder–decoder trên tập dữ liệu En–Vi (IWSLT). Pipeline được thiết kế để dễ phát triển cục bộ, huấn luyện trên GPU thuê (VastAI), lưu model lên Hugging Face và đánh giá bằng BLEU / ROUGE / PPL. Có demo Colab trực quan để chạy nhanh.

---

## Nội dung chính
- Mục tiêu
- Các thí nghiệm
- Hướng dẫn pipeline (tokenize → train → evaluate)
- Kết quả tóm tắt
- Liên kết hữu ích

---

## Mục tiêu dự án
Xây dựng và đánh giá hiệu năng các mô hình Seq2Seq Transformer cho Machine Translation (English → Vietnamese), đồng thời so sánh ảnh hưởng của dữ liệu tăng cường (synthetic / augmented data) trên các mô hình lớn và nhỏ.

So sánh giữa:
- ViT5-base (VietAI/vit5-base) — baseline mạnh
- HUST T5 Small (NlpHUST/t5-en-vi-small) — mô hình nhỏ, nhanh
- Mỗi mô hình thử với dữ liệu Original và Augmented

---

## Các thí nghiệm
Tất cả config nằm trong `src/config.py`.

1. ViT5-base + Original Data  
   - Model: `VietAI/vit5-base`  
   - Train size: ~133k

2. ViT5-base + Augmented Data  
   - Giữ model, thêm synthetic data để kiểm tra cải thiện

3. HUST T5 Small + Original Data  
   - Model: `NlpHUST/t5-en-vi-small`

4. HUST T5 Small + Augmented Data

---

## Cách chạy (Pipeline)

1) Tokenize toàn bộ dataset
```bash
# ViT5 original
python -m src.data --exp vit5_original

# ViT5 augmented
python -m src.data --exp vit5_augmented

# HUST original
python -m src.data --exp hust_original

# HUST augmented
python -m src.data --exp hust_augmented
```

2) Training (ví dụ chạy trên VastAI)
```bash
# Training ViT5
python -m src.train_vit5 --exp vit5_original

# Resume training
python -m src.train_vit5 --exp vit5_original --resume
```

3) Evaluate (BLEU / ROUGE / Loss / PPL)
```bash
python -m src.evaluate --exp vit5_original
```

Gợi ý: kiểm tra config ở `src/config.py` để điều chỉnh hyperparams, batch-size, lr, checkpoint path, v.v.

---

## Kết quả tóm tắt (Test set)

| Experiment        | BLEU   | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------------------|--------|---------|---------|---------|
| HUST Original     | 0.3276 | 0.7440  | 0.5272  | 0.6636  |
| HUST Augmented    | 0.3287 | 0.7470  | 0.5354  | 0.6666  |
| ViT5 Original     | ~0.41–0.45 (tùy môi trường) | ~0.66+ | ~0.57+ | — |
| ViT5 Augmented    | Tăng nhẹ so với original (tùy môi trường) | — | — | — |

Ghi chú: Các kết quả ViT5 có thể thay đổi tùy môi trường huấn luyện (GPU, seed, batch size).

---

## Demo & Models

- Colab Evaluation: https://colab.research.google.com/your_notebook
- Hugging Face models:
  - https://huggingface.co/NguyenwillG/hust_original
  - https://huggingface.co/NguyenwillG/hust_augmented

---

## Cấu trúc repo (tóm tắt)
- src/
  - config.py — tất cả các cấu hình experiments
  - data.py — preprocessing / tokenize
  - train_vit5.py — training script cho ViT5
  - evaluate.py — đánh giá BLEU/ROUGE/PPL
- scripts/ hoặc notebooks/ — (nếu có) helper scripts, Colab demo

---

## Gợi ý nâng cao
- Dùng Git LFS cho checkpoint lớn nếu lưu trên GitHub.
- Thử nhiều seed để ổn định kết quả BLEU.
- Theo dõi training bằng Weights & Biases / TensorBoard.

---

## Liên hệ
- Tác giả: NguyenwillG (Nguyen3007)  
- Repo: https://github.com/Nguyen3007/nlp-encode-decode
