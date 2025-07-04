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
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))  # Fill the upper triangle with -inf
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
        """
        Embedding dimesion should be divisible by the number of heads in MultiHeadAttention.
        """
        super().__init__()
        self.embedding = nn.Embedding(token_size, embed_dim)

    def forward(self, x):
        return self.embedding(x)

class MultiHeadAttetion(nn.Module):
    """
    This class implements the Multi-Head Attention mechanism.
    See https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/activation.py#L977.
    MHA is bulit in 'activation.py'. But, for simplicity, we implement it here.
    """
    def __init__(self, embed_dim, head_num, k_dim, v_dim, cross_attention=False):
        super().__init__()
        if embed_dim % head_num != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by head_num {head_num}.")
        
        self.head_dim = embed_dim // head_num
        self.head_num = head_num
        self.cross_attention = cross_attention
        self.k_dim = k_dim

        self.wq = nn.Linear(self.head_dim, k_dim)
        self.wk = nn.Linear(self.head_dim, k_dim)
        self.wv = nn.Linear(self.head_dim, v_dim)
        self.wo = nn.Linear(v_dim, embed_dim)  # Output linear layer to concatenate and combine heads

    def forward(self, x, memory = None, mask=None, dropout=None, return_attention=False):
        """
        Forward pass for the attention mechanism. 
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            memory (torch.Tensor, optional): Memory tensor for cross-attention. Defaults to None (batch_size, seq_len, embed_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
            torch.Tensor (optional) : Attention scores of shape (batch_size, head_num, seq_len, seq_len) if return_attention is True.
        
        """
        if x.dim() != 3:
            raise ValueError(f"Input tensor x must be 3-dimensional, got {x.dim()} dimensions.")

        if self.cross_attention:
            if memory is None:
                raise ValueError("Memory tensor must be provided for cross-attention.")
            if memory.dim() != 3:
                raise ValueError(f"Memory tensor must be 3-dimensional, got {memory.dim()} dimensions.")
            if memory.size(2) != embed_dim:
                raise ValueError(f"Memory tensor's last dimension must match embed_dim {embed_dim}, got {memory.size(2)}.")
            if memory.size(1) != seq_len:
                raise ValueError(f"Memory tensor's second dimension must match seq_len {seq_len}, got {memory.size(1)}.")
            if memory.size(0) != x.size(0):
                raise ValueError(f"Memory tensor's first dimension must match batch_size {x.size(0)}, got {memory.size(0)}.")

        # Reshape x to (batch_size, seq_len, head_num, head_dim)
        batch_size, seq_len, embed_dim = x.shape
        x = x.reshape(batch_size, seq_len, self.head_num, self.head_dim)
        x = x.permute(0, 2, 1, 3)  # x.shape = (batch_size, head_num, seq_len, head_dim)

        if self.cross_attention:
            # Reshape memory to (batch_size, seq_len, head_num, head_dim)
            memory = memory.reshape(batch_size, seq_len, self.head_num, self.head_dim)
            memory = memory.permute(0, 2, 1, 3)


        # Compute query, key, and value matrices
        q = self.wq(x) # q.shape = (batch_size, head_num, seq_len, k_dim)
        if self.cross_attention:
            k = self.wk(memory) # k.shape = (batch_size, head_num, seq_len, k_dim)
            v = self.wv(memory) # v.shape = (batch_size, head_num, seq_len, v_dim)
        else:
            k = self.wk(x) # k.shape = (batch_size, head_num, seq_len, k_dim)
            v = self.wv(x) # v.shape = (batch_size, head_num, seq_len, v_dim)

        # Compute attention scores
        similarity = torch.matmul(q, k.transpose(-2, -1)) / (self.k_dim ** 0.5)  # similarity.shape = (batch_size, head_num, seq_len, seq_len)
        if mask is not None:
            if mask.dim() != 2 or mask.size(0) != seq_len or mask.size(1) != seq_len:
                raise ValueError(f"Mask must be 2-dimensional with shape (seq_len, seq_len), got {mask.shape}.")
            mask = mask.unsqueeze(0).unsqueeze(0)
            similarity = similarity*mask  # Apply mask to similarity scores
        
        attn_score = F.softmax(similarity, dim=-1)  # attn_score.shape = (batch_size, head_num, seq_len, seq_len)

        # Compute attention output
        output = torch.matmul(attn_score, v)  # output.shape = (batch_size, head_num, seq_len, v_dim)
        output = output.permute(0, 2, 1, 3)  # output.shape = (batch_size, seq_len, head_num, v_dim)
        output = self.wo(output.reshape(batch_size, seq_len, -1))  # output.shape = (batch_size, seq_len, embed_dim)

        if return_attention:
            return output, attn_score
        else:
            return output
        
    def backward(self, grad_output):
        """
        Backward pass for the attention mechanism.
        Args:
            grad_output (torch.Tensor): Gradient of the output tensor.
        """
        grad_x = grad_output

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
