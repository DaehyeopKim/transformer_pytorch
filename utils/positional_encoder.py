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

class AbsoluteSinusoidalPositionalEmbedding(nn.Module):
    '''
    Absolute Sinusoidal Positional Encoding
    This method is introduced in the paper "Attention Is All You Need" by Vaswani et al.
    '''
    def __init__(self, dim, max_len=5000, b = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.b = b
        self.pe = self._generate_pe()

    def _generate_pe(self):
        '''
        Generate the positional encoding matrix.
        Return:
            pe: Positional encoding matrix of shape (1, max_len, dim)
        '''
        pe = torch.zeros(self.max_len, self.dim).float() # pe.shape = (max_len, dim)
        position = torch.arange(0, self.max_len).unsqueeze(1).float() # position.shape = (max_len, 1)
        even_div_term = self.b ** (torch.arange(0, self.dim, 2).float() / float(self.dim)) # even_div_term.shape = (dim/2,)
        odd_div_term = self.b ** (torch.arange(1, self.dim, 2).float() / float(self.dim))
        pe[:, 0::2] = torch.sin(position / even_div_term)
        pe[:, 1::2] = torch.cos(position / odd_div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        '''
        Generate positional encoding for the input tensor x.
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
        Return:
            pe: Positional encoding tensor of shape (1, seq_len, dim)
        '''
        return x + self.pe[:, :x.size(1), :]