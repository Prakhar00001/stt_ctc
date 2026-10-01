import torch, torch.nn as nn, torch.nn.functional as F

class SpeechToTextModel(nn.Module):
    def __init__(self, v=29, h=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, (2, 2), 1), nn.BatchNorm2d(32), nn.ReLU(True), 
            nn.Conv2d(32, 32, 3, (1, 2), 1), nn.BatchNorm2d(32), nn.ReLU(True)
        )
        self.lstm = nn.LSTM(32*20, h, 2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(h*2, v)

    def get_seq_lens(self, lengths): 
        return torch.div(lengths + 1, 2, rounding_mode='floor')

    def forward(self, x):
        x = self.conv(x.transpose(2, 3))
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * f)
        out, _ = self.lstm(x)
        return F.log_softmax(self.fc(out), dim=-1)