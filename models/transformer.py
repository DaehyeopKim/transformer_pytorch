'''
1. All modules are inherited from nn.Module. In this way, we can use the built-in methods of nn.Module such as 
'.to()', '.cuda()', '.cpu()', '.eval()', and '.train()'. These allows us to move all the parameters of the modules
 and the sub modules to GPU or CPU, and also to switch between training and evaluation modes.
See https://github.com/pytorch/pytorch/blob/v2.7.0/torch/nn/modules/module.py for more details.

2. This code is stongly inspired by the original paper "Attention is All You Need" and the official PyTorch implementation.
See https://github.com/pytorch/pytorch/blob/v2.7.0/torch/nn/modules/transformer.py, https://arxiv.org/abs/1706.03762
'''

import torch
import torch.nn as nn   
import torch.nn.functional as F
import logging

__all__ = [
    "FeedForward",
    "Attention",
    "SingleHeadAttention",
    "MultiHeadAttention",
    "DecoderBlock",
    "EncoderBlock",
    "Encoder",
    "Decoder",
    "Transformer",
]

# Set up logging configuration
def logging_setup():
    """
    Set up logging configuration for the transformer model.
    """
    global logger
    logger = logging.getLogger("models/transformer.py")

    file_handler = logging.FileHandler()
    file_handler.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs 

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Set to INFO for less verbose logs


    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
logging_setup()

def generate_mask(seq_len, device=None):
    """
    Generate a square mask for the sequence length.
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    if device is not None:
        mask = mask.to(device)
    return mask

class FeedForward(nn.Module):
    '''
    3.3 Position-wise Feed-Forward Networks
    '''
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        out1 = self.w1(x)
        out2 = F.relu(out1)
        output = self.w2(out2)
        return output
    
    def backward(self, grad_output):
        grad_out2 = self.w2.backward(grad_output)
        grad_out1 = F.relu.backward(grad_out2)
        grad_x = self.w1.backward(grad_out1)
        return grad_x
    
class Embedding(nn.Module):
    """
    This class implements the embedding layer for the transformer model.
    See https://github.com/pytorch/pytorch/blob/v2.7.0/torch/nn/modules/transformer.py, it doesn't contain
    word embedding, but we implement it here for simplicity. 
    The distance of two tokens in this embedding space represents the similarity between the two tokens.  
    """
    def __init__(self, token_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(token_size, embed_dim)

    def forward(self, x):
        return self.embedding(x)

class Attention(nn.Module):
    def __init__(self, embed_dim, q_dim, k_dim, v_dim, self_attention=False):
        super().__init__()
        self.wq = nn.Linear(embed_dim, q_dim)
        self.wk = nn.Linear(embed_dim, k_dim)
        self.wv = nn.Linear(embed_dim, v_dim)


    def forward(self, x):
        # Dummy forward method
        return x * 2
    
class MultiHeadAttention(nn.Module):
    """
    See https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/activation.py#L977.
    MHA is bulit in 'activation.py' and there's no sub-module. 
    But, for simplicity, we implement it here with sub-module(class Attention).
    This class implements the Multi-Head Attention mechanism.
    """
    def __init__(self, model_name):
        self.model_name = model_name

    def forward(self, x):
        # Dummy forward method
        return x * 2
    
class DecoderBlock:
    def __init__(self, model_name):
        self.model_name = model_name

    def forward(self, x):
        # Dummy forward method
        return x * 2
    
class EncoderBlock:
    def __init__(self, model_name):
        self.model_name = model_name

    def forward(self, x):
        # Dummy forward method
        return x * 2

class Encoder:
    def __init__(self, model_name):
        self.model_name = model_name

    def forward(self, x):
        # Dummy forward method
        return x * 2
    
class Decoder:
    def __init__(self, model_name):
        self.model_name = model_name

    def forward(self, x):
        # Dummy forward method
        return x * 2

class Transformer(nn.Module):
    def __init__(self, model_name):
        self.model_name = model_name

    def forward(self, x):
        # Dummy forward method
        return x * 2
