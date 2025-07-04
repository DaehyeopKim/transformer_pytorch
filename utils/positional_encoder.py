import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len=5000):
        self.dim = dim
        self.max_len = max_len
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Input sequence length {seq_len} exceeds maximum length {self.max_len}")

        t = torch.arange(seq_len, device=x.device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        freqs = freqs.unsqueeze(0).expand(x.size(0), -1, -1)

        return freqs