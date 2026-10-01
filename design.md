Create `design.md`:
```markdown
# Architecture
- Audio: 16kHz, Mel: 80 bins, FFT: 400, Hop: 160
- CNN Subsampling: Factor 2 (Time dimension halved)
- BiLSTM: 2 layers, 256 hidden
- Loss: CTCLoss(blank=0, zero_infinity=True)