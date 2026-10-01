# System Architecture & Design Document

This document outlines the architectural decisions, structural constraints, and hyperparameter choices for the pure PyTorch STT CTC pipeline.

## 1. System Constraints
- **Framework:** Pure PyTorch (No external STT wrappers or high-level abstract libraries).
- **Dataset Limit:** Strictly limited to exactly 20 LibriSpeech audio samples to mathematically prove algorithmic overfitting.
- **Vocabulary:** 29 classes (`a-z`, `<space>`, `'`, and index `0` reserved for `<BLANK>`).

## 2. Acoustic Frontend (`features.py`)
To convert raw audio waveforms into learnable features, the pipeline utilizes Log-Mel Spectrograms:
- **Sample Rate:** 16,000 Hz
- **N_Mels (Features):** 80 bins
- **N_FFT (Window Size):** 400 (25ms window)
- **Hop Length:** 160 (10ms stride)
- **Normalization:** Z-score normalization (mean subtraction and standard deviation division) is applied per-spectrogram to stabilize early gradient flow.

## 3. Neural Network Topology (`model.py`)
The acoustic model is a sequence-to-sequence neural network designed to map varying-length audio frames to character probabilities.

### A. 2D CNN Subsampling
- **Structure:** 2-layer `Conv2d` with `BatchNorm2d` and `ReLU` activations.
- **Stride:** `(2, 2)` on the first layer, which reduces the temporal dimension by a factor of 2.
- **Purpose:** Extracts local acoustic textures while halving the sequence length. This acts as a downsampler, significantly reducing the computational bottleneck for the subsequent RNN layers.

### B. Temporal Modeling (BiLSTM)
- **Structure:** 2-layer Bidirectional LSTM.
- **Hidden Size:** 256 units.
- **Purpose:** Captures long-range phonetic context (both past and future audio frames) which is essential for resolving speech ambiguities and forming coherent words.

### C. Linear Classifier
- **Structure:** `nn.Linear(256 * 2, 29)`.
- **Purpose:** Projects the 512-dimensional BiLSTM output at each timestep down to the 29 target vocabulary classes. It is followed by `LogSoftmax` across the class dimension to generate valid probability distributions.

## 4. Loss & Decoding Pipeline (`decoder.py`)
- **Objective Function:** `nn.CTCLoss(blank=0, zero_infinity=True)`. The `zero_infinity=True` flag is actively utilized to prevent gradient explosions if the network encounters sequences where target lengths momentarily exceed input lengths during padding.
- **Decoding Strategy:** Greedy CTC Decoding. The algorithm takes the `argmax` of the probability matrix at each timestep, collapses consecutive identical phonetic predictions into single characters, and explicitly strips the `<BLANK>` tokens to reconstruct the final English string.