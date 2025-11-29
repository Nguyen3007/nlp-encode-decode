🌐 Vietnamese Machine Translation – Encoder–Decoder Experiments
Fine-tuning ViT5 & HUST T5 (Seq2Seq) on IWSLT En–Vi Dataset

Dự án này triển khai 4 thí nghiệm fine-tuning mô hình dịch máy encoder–decoder trên tập dữ liệu En–Vi (IWSLT).
Pipeline được thiết kế theo phong cách :

Local dev (PyCharm)

Training trên GPU thuê VastAI

Lưu model lên HuggingFace

Evaluate bằng BLEU/ROUGE/PPL

Colab demo trực quan

🚀 1. Mục tiêu dự án

Xây dựng và đánh giá hiệu quả của các mô hình Seq2Seq Transformer dùng cho bài toán Machine Translation (English → Vietnamese).

So sánh hiệu năng giữa:

Mô hình	Dữ liệu train	Mục tiêu
ViT5-base	Original	Baseline mạnh với mô hình lớn
ViT5-base	Augmented	Kiểm tra hiệu quả của synthetic data
HUST T5 Small	Original	Mô hình nhỏ, nhanh, so sánh với ViT5
HUST T5 Small	Augmented	Mô hình nhỏ + synthetic data

⚙️ thí nghiệm (Experiments)

Tất cả các config lưu trong src/config.py.

(1) ViT5-base + Original Data

Model: VietAI/vit5-base

Train size: 133k

(2) ViT5-base + Augmented Data

Model giữ nguyên

So sánh hiệu quả tăng BLEU không

(3) HUST T5 Small + Original Data

Model: NlpHUST/t5-en-vi-small


(4) HUST T5 Small + Augmented Data


🏗 Pipeline huấn luyện


Bước 1 – Tokenize toàn bộ dataset


python -m src.data --exp vit5_original


python -m src.data --exp vit5_augmented


python -m src.data --exp hust_original


python -m src.data --exp hust_augmented


Bước 2 – Training (trên VastAI)

Example:

python -m src.train_vit5 --exp vit5_original

Resume training:

python -m src.train_vit5 --exp vit5_original --resume

Bước 3 – Evaluate (BLEU, ROUGE, Loss)
python -m src.evaluate --exp vit5_original

📊  Kết quả chi tiết

📌 BLEU & ROUGE trên tập Test

Experiment	BLEU	ROUGE-1	ROUGE-2	ROUGE-L

HUST Original	0.3276	0.7440	0.5272	0.6636

HUST Augmented	0.3287	0.7470	0.5354	0.6666

ViT5 Original	(tùy môi trường)	~0.41–0.45	~0.66+	~0.57+

ViT5 Augmented	(tùy môi trường)	Tăng nhẹ so với original	


🔗  Demo Colab


[🔗 Colab Evaluation](https://colab.research.google.com/your_notebook)

☁️ Model trên HuggingFace

https://huggingface.co/NguyenwillG/hust_original

https://huggingface.co/NguyenwillG/hust_augmented
