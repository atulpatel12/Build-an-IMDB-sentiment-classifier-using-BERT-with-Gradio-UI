# 🎬 IMDB Sentiment Classifier using BERT & Gradio

An end-to-end Natural Language Processing (NLP) pipeline that fine-tunes a pre-trained **BERT** model (`bert-base-uncased`) to classify movie review sentiments as either **Positive** or **Negative**. The project includes full dataset processing, model fine-tuning with Hugging Face's `Trainer` API, performance evaluation, and an interactive **Gradio** web application for real-time review inference.

---

## 📌 Key Highlights & Features

- **Pre-trained Architecture:** Leverages Transformer-based `bert-base-uncased` for deep contextual text embeddings.
- **Dataset:** Fine-tuned on the benchmark [IMDB Movie Reviews](https://huggingface.co/datasets/stanfordnlp/imdb) dataset.
- **Efficient Tokenization & Dynamic Padding:** Uses `AutoTokenizer` and `DataCollatorWithPadding` to dynamically pad sequences per batch up to a maximum length of 256 tokens, optimizing compute efficiency.
- **Mixed Precision Training:** Utilizes FP16 training with PyTorch and CUDA for fast GPU acceleration.
- **Interactive Web Interface:** Integrates a built-in **Gradio UI** allowing users to enter custom movie reviews and receive instant sentiment predictions along with model confidence scores.

---

## 📊 Performance & Evaluation Results

After fine-tuning for **2 epochs** with a learning rate of `2e-5`:

| Metric | Score |
| :--- | :--- |
| **Validation Accuracy** | **89.50%** |
| **Test Set Accuracy** | **98.35%** |
| **Test Set F1 Score** | **0.9835** |

---

## 🛠️ Tech Stack & Libraries

- **Language:** Python 3.10+
- **Deep Learning Framework:** PyTorch & CUDA
- **NLP Ecosystem:** Hugging Face `transformers`, `datasets`, and `accelerate`
- **Evaluation:** Scikit-Learn (`accuracy_score`, `f1_score`)
- **Web UI:** Gradio

---

## 🚀 How to Run in Google Colab

1. Click the **"Open in Colab"** badge at the top of the notebook (`.ipynb`).
2. Select a GPU runtime: **Runtime > Change runtime type > T4 GPU**.
3. Run all cells sequentially to install dependencies, load data, fine-tune the model, and launch the Gradio public interface link.
