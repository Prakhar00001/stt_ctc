import torch
import torchaudio.transforms as T

class LogMelFeatureExtractor(torch.nn.Module):
    def __init__(self, sr=16000, n_fft=400, hop=160, n_mels=80):
        super().__init__()
        self.mel = T.MelSpectrogram(sample_rate=sr, n_fft=n_fft, win_length=n_fft, hop_length=hop, n_mels=n_mels, center=True)
        
    def forward(self, w):
        if w.dim() == 1: w = w.unsqueeze(0)
        log_mel = torch.log(torch.clamp(self.mel(w), min=1e-5))
        return ((log_mel - log_mel.mean(dim=-1, keepdim=True)) / (log_mel.std(dim=-1, keepdim=True) + 1e-5)).unsqueeze(1)