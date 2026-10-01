# End-to-End Speech-to-Text (STT) Pipeline using CTC

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

A pure PyTorch implementation of a Speech-to-Text (STT) acoustic model. This project mathematically verifies a custom Connectionist Temporal Classification (CTC) architecture by successfully overfitting a highly restricted dataset of exactly 20 LibriSpeech audio samples to a **0.00% Character Error Rate (CER)**.

##  Project Objective
Built without relying on high-level STT wrappers (e.g., HuggingFace), this pipeline demonstrates raw tensor manipulation, custom dynamic batching for variable-length sequences, and the fundamental mechanics of CTC loss and greedy decoding.

##  Architecture
- **Frontend:** 80-bin Log-Mel Spectrogram extraction (FFT: 400, Hop: 160).
- **Subsampling:** 2-layer 2D CNN (Stride 2) to extract local acoustic features and halve the temporal dimension.
- **Acoustic Model:** 2-layer Bidirectional LSTM (256 hidden units) for temporal sequence modeling.
- **Loss & Decoding:** `nn.CTCLoss` with a 29-character vocabulary (`a-z`, space, apostrophe, `<BLANK>`), resolved via a custom Greedy CTC Decoder.

## 📂 Project Structure
```text
stt_ctc/
├── data/                   # Auto-downloaded LibriSpeech dev-clean samples
├── outputs/
│   ├── training_log.txt    # Epoch-by-epoch loss and CER metrics
│   └── predictions.txt     # Final Target vs. Prediction outputs
├── src/
│   ├── dataset.py          # Downloading, filtering, and CTC dynamic collator
│   ├── decoder.py          # Greedy CTC decoder and CER calculation
│   ├── features.py         # Log-Mel Spectrogram extraction
│   ├── model.py            # CNN + BiLSTM neural network
│   ├── tokenizer.py        # 29-char text normalization and mapping
│   └── train.py            # Training loop and orchestrator
├── design.md               # Architecture design constraints
├── README.md               # Project documentation
├── requirements.txt        # Dependencies
└── stt_ctc.ipynb           # Cloud-ready Colab execution artifact


🚀 How to Run

1. Clone the repository:

git clone [https://github.com/Prakhar00001/stt_ctc.git](https://github.com/Prakhar00001/stt_ctc.git)
cd stt_ctc

2. Install dependencies:

pip install -r requirements.txt

3. Trigger the training pipeline:

python -m src.train

📊 Results
The model achieves perfect memorization of the 20-sample dataset across 800 epochs, demonstrating stable gradient descent and flawless sequence alignment.

Initial State: ~3.34 Loss | 100.00% CER

Final State: <0.0010 Loss | 0.00% CER


