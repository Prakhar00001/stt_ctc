import os, tarfile, urllib.request, torch, torchaudio
from torch.utils.data import Dataset
from .tokenizer import CTCTokenizer
from .features import LogMelFeatureExtractor

class SpeechOverfitDataset(Dataset):
    def __init__(self, data_dir, num_samples=20):
        self.samples, self.tok = [], CTCTokenizer()
        tar_path = os.path.join(data_dir, "dev-clean.tar.gz")
        if not os.path.exists(os.path.join(data_dir, "LibriSpeech")):
            print("Downloading LibriSpeech...")
            urllib.request.urlretrieve("https://www.openslr.org/resources/12/dev-clean.tar.gz", tar_path)
            with tarfile.open(tar_path, "r:gz") as tar: tar.extractall(data_dir)
        
        cands = []
        for r, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".trans.txt"):
                    with open(os.path.join(r, file)) as f:
                        for line in f:
                            parts = line.strip().split(" ", 1)
                            if len(parts) == 2 and os.path.exists(p := os.path.join(r, f"{parts[0]}.flac")):
                                meta = torchaudio.info(p)
                                dur = meta.num_frames / meta.sample_rate
                                if 1.2 <= dur <= 3.2 and (((meta.num_frames // 160 + 1) + 1) // 2) > len(self.tok.normalize(parts[1])) + 4:
                                    cands.append((p, self.tok.normalize(parts[1]), dur))
        
        for p, t, d in sorted(cands, key=lambda x: x[2])[:num_samples]:
            w, sr = torchaudio.load(p)
            if sr != 16000: w = torchaudio.transforms.Resample(sr, 16000)(w)
            self.samples.append((w.mean(dim=0, keepdim=True).squeeze(0) if w.shape[0] > 1 else w.squeeze(0), t, d))

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

class CTCCollateFn:
    def __init__(self, fe, tok): 
        self.fe, self.tok = fe, tok
    def __call__(self, batch):
        w, txt, _ = zip(*batch)
        mels = [self.fe(x).squeeze(0) for x in w]
        pad_mels = torch.zeros(len(mels), 1, 80, max(m.shape[-1] for m in mels), dtype=torch.float32)
        for i, m in enumerate(mels): pad_mels[i, :, :, :m.shape[-1]] = m
        enc = [self.tok.encode(t) for t in txt]
        return {
            "padded_mels": pad_mels, 
            "mel_lengths": torch.tensor([m.shape[-1] for m in mels]), 
            "targets": torch.cat([torch.tensor(t) for t in enc]), 
            "target_lengths": torch.tensor([len(t) for t in enc]), 
            "transcripts": list(txt)
        }