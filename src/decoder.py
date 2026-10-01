import torch

def compute_cer(r, h):
    d = [[i+j if i*j==0 else 0 for j in range(len(h)+1)] for i in range(len(r)+1)]
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            d[i][j] = d[i-1][j-1] if r[i-1]==h[j-1] else 1+min(d[i-1][j], d[i][j-1], d[i-1][j-1])
    return d[len(r)][len(h)] / max(1, len(r))

class GreedyCTCDecoder:
    def __init__(self, tok): self.tok = tok
    
    def decode(self, probs, lens):
        out = []
        for seq, l in zip(torch.argmax(probs, -1).tolist(), lens.tolist()):
            collapsed = [t for i, t in enumerate(seq[:l]) if t != 0 and (i == 0 or t != seq[i-1])]
            out.append(self.tok.decode(collapsed))
        return out