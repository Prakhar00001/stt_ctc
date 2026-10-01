import os, torch, torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import SpeechOverfitDataset, CTCCollateFn
from src.features import LogMelFeatureExtractor
from src.tokenizer import CTCTokenizer
from src.model import SpeechToTextModel
from src.decoder import GreedyCTCDecoder, compute_cer

def run():
    torch.manual_seed(42)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = CTCTokenizer()
    
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    dl = DataLoader(SpeechOverfitDataset("data"), batch_size=20, collate_fn=CTCCollateFn(LogMelFeatureExtractor().to(dev), tok))
    model = SpeechToTextModel().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 180)
    crit = nn.CTCLoss(blank=0, zero_infinity=True)
    dec = GreedyCTCDecoder(tok)
    
    batch = next(iter(dl))
    mels, mel_lens, tgts, tgt_lens, refs = batch["padded_mels"].to(dev), batch["mel_lengths"].to(dev), batch["targets"].to(dev), batch["target_lengths"].to(dev), batch["transcripts"]
    
    logs = []
    print(f"Training started on {dev}...")
    for ep in range(1, 181):
        model.train()
        opt.zero_grad()
        probs = model(mels)
        in_lens = model.get_seq_lens(mel_lens)
        loss = crit(probs.transpose(0, 1), tgts, in_lens, tgt_lens)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        
        if ep % 20 == 0 or ep == 180:
            model.eval()
            with torch.no_grad():
                preds = dec.decode(model(mels), in_lens)
                cer = sum(compute_cer(r, p) for r, p in zip(refs, preds)) / 20 * 100
                log = f"Epoch {ep:03d} | Loss: {loss.item():.4f} | CER: {cer:.2f}%"
                print(log)
                logs.append(log)

    with open("outputs/training_log.txt", "w") as f: 
        f.write("\n".join(logs))
    
    model.eval()
    with torch.no_grad():
        final_preds = dec.decode(model(mels), model.get_seq_lens(mel_lens))
        with open("outputs/predictions.txt", "w") as f: 
            f.write("\n\n".join([f"Target: {r}\nPred: {p}\nCER: {compute_cer(r,p)*100:.2f}%" for r, p in zip(refs, final_preds)]))
            
if __name__ == "__main__": 
    run()