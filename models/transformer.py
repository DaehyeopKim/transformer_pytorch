'''
1. All modules are inherited from nn.Module. In this way, we can use the built-in methods of nn.Module such as 
'.to()', '.cuda()', '.cpu()', '.eval()', and '.train()'. These allows us to move all the parameters of the modules
 and the sub modules to GPU or CPU, and also to switch between training and evaluation modes.
See https://github.com/pytorch/pytorch/blob/v2.7.0/torch/nn/modules/module.py for more details.

2. This code is stongly inspired by the original paper "Attention is All You Need" and the official PyTorch implementation.
See https://github.com/pytorch/pytorch/blob/v2.7.0/torch/nn/modules/transformer.py, https://arxiv.org/abs/1706.03762

3. We don't have to implement the backward pass for each module, because PyTorch automatically computes the gradients
   using autograd. But, we implement the backward pass for practice and to understand how the gradients are computed.
   See https://pytorch.org/docs/stable/notes/autograd.html for more details.
'''

import torch
import torch.nn as nn   
import torch.nn.functional as F
import logging
import utils.positional_encoder as positional_encoder
import utils.inference_method as inference_method

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

logging_setup()

def generate_mask(seq_len, device=None):
    """
    Generate a square mask for the sequence length.
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf')) 
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
        """
        Backward pass for the feed forward layer.
        Note: This is implemented for educational purposes to understand gradient flow.
        """
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

    def backward(self, grad_output):
        """
        Backward pass for the embedding layer.
        Note: This is implemented for educational purposes to understand gradient flow.
        Args:
            grad_output (torch.Tensor): Gradient of the output tensor.
        Returns:
            torch.Tensor: Gradient of the input tensor.
        """
        return self.embedding.backward(grad_output)

class MultiHeadAttention(nn.Module):
    """
    This class implements the Multi-Head Attention mechanism.
    See https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/activation.py#L977.
    MHA is bulit in 'activation.py'. But, for simplicity, we implement it here 'transformer.py'.
    """
    def __init__(self, embed_dim, head_num, k_dim, v_dim, cross_attention=False):
        super().__init__()
        if embed_dim % head_num != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by head_num {head_num}.")
        
        self.head_dim = embed_dim // head_num
        self.head_num = head_num
        self.cross_attention = cross_attention
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.wq = nn.Linear(embed_dim, head_num * k_dim)
        self.wk = nn.Linear(embed_dim, head_num * k_dim)
        self.wv = nn.Linear(embed_dim, head_num * v_dim)
        self.wo = nn.Linear(head_num * v_dim, embed_dim) 

    def forward(self, x, memory = None, mask=None, return_attention=False):
        """
        Forward pass for the attention mechanism. 
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            memory (torch.Tensor, optional): Memory tensor for cross-attention. Defaults to None (batch_size, seq_len, embed_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
            torch.Tensor (optional) : Attention scores of shape (batch_size, head_num, seq_len, seq_len) if return_attention is True.
        Note:
            The shape of self-attention score is (batch_size, head_num, seq_len, seq_len).
            The shape of self-attention value is (batch_size, head_num, seq_len, v_dim).

            The shape of cross-attention score is (batch_size, head_num, seq_len, memory_seq_len).
            The shape of cross-attention value is (batch_size, head_num, memory_seq_len, v_dim).
        """
        batch_size, seq_len, embed_dim = x.shape
        if x.dim() != 3:
            raise ValueError(f"Input tensor x must be 3-dimensional, got {x.dim()} dimensions.")

        if self.cross_attention:
            if memory is None:
                raise ValueError("Memory tensor must be provided for cross-attention.")
            if memory.dim() != 3:
                raise ValueError(f"Memory tensor must be 3-dimensional, got {memory.dim()} dimensions.")
            if memory.size(2) != embed_dim:
                raise ValueError(f"Memory tensor's last dimension must match embed_dim {embed_dim}, got {memory.size(2)}.")
            if memory.size(0) != x.size(0):
                raise ValueError(f"Memory tensor's first dimension must match batch_size {x.size(0)}, got {memory.size(0)}.")


        # 1. Compute query, key, and value matrices
        self.q = self.wq(x) # q.shape = (batch_size, seq_len, head_num * k_dim)
        if self.cross_attention:
            self.k = self.wk(memory) # k.shape = (batch_size, seq_len, head_num * k_dim)
            self.v = self.wv(memory) # v.shape = (batch_size, seq_len, head_num * v_dim)
        else:
            self.k = self.wk(x) # k.shape = (batch_size, seq_len, head_num * k_dim)
            self.v = self.wv(x) # v.shape = (batch_size, seq_len, head_num * v_dim)

        # 2. Reshape and permute to get the correct dimensions for multi-head attention
        self.q = self.q.reshape(batch_size, seq_len, self.head_num, self.k_dim).transpose(1,2)  # q.shape = (batch_size, head_num, seq_len, k_dim)
        
        if self.cross_attention:
            # For cross-attention, K and V come from memory with potentially different seq_len
            memory_seq_len = memory.size(1)
            self.k = self.k.reshape(batch_size, memory_seq_len, self.head_num, self.k_dim).transpose(1,2)  # k.shape = (batch_size, head_num, memory_seq_len, k_dim)
            self.v = self.v.reshape(batch_size, memory_seq_len, self.head_num, self.v_dim).transpose(1,2)  # v.shape = (batch_size, head_num, memory_seq_len, v_dim)
        else:
            # For self-attention, all use the same seq_len
            self.k = self.k.reshape(batch_size, seq_len, self.head_num, self.k_dim).transpose(1,2)  # k.shape = (batch_size, head_num, seq_len, k_dim)
            self.v = self.v.reshape(batch_size, seq_len, self.head_num, self.v_dim).transpose(1,2)  # v.shape = (batch_size, head_num, seq_len, v_dim)

        # 3. Compute attention scores
        # For cross-attention: similarity.shape = (batch_size, head_num, seq_len, memory_seq_len)
        # For self-attention: similarity.shape = (batch_size, head_num, seq_len, seq_len)
        similarity = torch.matmul(self.q, self.k.transpose(-2, -1)) / (self.k_dim ** 0.5)
        
        if mask is not None:
            # Note: Mask is only applicable for self-attention (causal mask)
            # Cross-attention typically doesn't use causal masking
            if not self.cross_attention:
                if mask.dim() != 2 or mask.size(0) != seq_len or mask.size(1) != seq_len:
                    raise ValueError(f"Mask must be 2-dimensional with shape (seq_len, seq_len), got {mask.shape}.")
                mask = mask.unsqueeze(0).unsqueeze(0)
                similarity = similarity + mask  # Apply mask to similarity scores
        
        self.attn_score = F.softmax(similarity, dim=-1)  # attn_score.shape = (batch_size, head_num, seq_len, seq_len)

        # 4. Compute attention output
        output = torch.matmul(self.attn_score, self.v)  # output.shape = (batch_size, head_num, seq_len, v_dim)
        output = output.permute(0, 2, 1, 3)  # output.shape = (batch_size, seq_len, head_num, v_dim)
        output = self.wo(output.reshape(batch_size, seq_len, -1))  # output.shape = (batch_size, seq_len, embed_dim)

        if return_attention:
            return output, self.attn_score
        else:
            return output
            
    def backward(self, grad_output, memory=None):
        """
        Backward pass for the attention mechanism, supporting both self- and cross-attention.
        Args:
            grad_output (torch.Tensor): Gradient of the output tensor (batch, seq, embed_dim)
            memory (torch.Tensor, optional): Memory tensor for cross-attention (batch, seq, embed_dim)
        Returns:
            grad_x, grad_memory (if cross_attention)
        """
        batch_size, seq_len, embed_dim = grad_output.shape

        # 1. Output projection backward
        grad_output = self.wo.backward(grad_output)  # (batch, seq_len, head_num * v_dim)
        grad_output = grad_output.reshape(batch_size, seq_len, self.head_num, self.v_dim)
        grad_output = grad_output.permute(0, 2, 1, 3)  # (batch, head_num, seq_len, v_dim)

        # 2. Attention output backward
        grad_attn_score = torch.matmul(grad_output, self.v.transpose(-2, -1))  # (batch, head_num, seq_len, seq_len)
        grad_v = torch.matmul(self.attn_score.transpose(-2, -1), grad_output)  # (batch, head_num, seq_len, v_dim)

        # 3. Softmax backward
        grad_similarity = F.softmax.backward(grad_attn_score, self.attn_score)  # (batch, head_num, seq_len, seq_len)

        # 4. Q, K backward
        grad_q = torch.matmul(grad_similarity, self.k) / (self.k_dim ** 0.5)
        grad_k = torch.matmul(grad_similarity.transpose(-2, -1), self.q) / (self.k_dim ** 0.5)

        # 5. Reshape gradients
        grad_q = grad_q.transpose(1, 2).reshape(batch_size, seq_len, -1) # (batch, seq_len, head_num * k_dim)
        grad_k = grad_k.transpose(1, 2).reshape(batch_size, seq_len, -1) # (batch, seq_len, head_num * k_dim)
        grad_v = grad_v.transpose(1, 2).reshape(batch_size, seq_len, -1) # (batch, seq_len, head_num * v_dim)

        # 6. Linear backward
        grad_q_input = self.wq.backward(grad_q)
        grad_k_input = self.wk.backward(grad_k)
        grad_v_input = self.wv.backward(grad_v)

        # 7. input gradient
        if self.cross_attention:
            grad_x = grad_q_input  # (batch, seq_len, embed_dim)
            grad_memory = grad_k_input + grad_v_input  # (batch, seq_len, embed_dim)
            return grad_x, grad_memory
        else:
            grad_x = grad_q_input + grad_k_input + grad_v_input
            return grad_x
    
class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, ffn_dim, head_num, k_dim, v_dim, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(embed_dim, head_num, k_dim, v_dim)
        self.feed_forward = FeedForward(embed_dim, ffn_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for the encoder block.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        # Self-attention with residual connection and dropout
        attn_output = self.self_attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and dropout
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
    def backward(self, grad_output):
        """
        Backward pass for the encoder block.
        Args:
            grad_output (torch.Tensor): Gradient of the output tensor.
        Returns:
            torch.Tensor: Gradient of the input tensor.
        """
        # Backward pass through feed-forward
        grad_output = self.norm.backward(grad_output)
        grad_ff = self.feed_forward.backward(grad_output)
        grad_x = grad_output + grad_ff
        
        # Backward pass through self-attention
        grad_output = self.norm.backward(grad_x)
        grad_attn = self.self_attention.backward(grad_output)
        grad_x = grad_attn + grad_output 
        
        return grad_x

class Encoder(nn.Module):
    def __init__(self, embed_dim, ffn_dim, head_num, k_dim, v_dim, N, dropout=0.1):
        super().__init__()
        self.blocks = nn.Sequential()
        for i in range(N):
            self.blocks.add_module(f"encoder_block_{i}", EncoderBlock(embed_dim, ffn_dim, head_num, k_dim, v_dim, dropout=dropout))

    def forward(self, x):
        """
        Forward pass for the encoder.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        for block in self.blocks:
            x = block(x)
        return x    
    
    def backward(self, grad_output):
        """
        Backward pass for the encoder.
        Args:
            grad_output (torch.Tensor): Gradient of the output tensor.
        Returns:
            torch.Tensor: Gradient of the input tensor.
        """
        for block in reversed(self.blocks):
            grad_output = block.backward(grad_output)
        return grad_output

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, ffn_dim, head_num, k_dim, v_dim, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(embed_dim, head_num, k_dim, v_dim)
        self.cross_attention = MultiHeadAttention(embed_dim, head_num, k_dim, v_dim, cross_attention=True)
        self.feed_forward = FeedForward(embed_dim, ffn_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory=None):
        """
        Forward pass for the decoder block.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        if memory is None:
            raise ValueError("Memory tensor must be provided for decoder block.")
        
        # Self-attention with residual connection and dropout
        self_attn_output = self.self_attention(x, mask=generate_mask(x.size(1), x.device))
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Cross-attention with residual connection and dropout
        cross_attn_output = self.cross_attention(x, memory)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward with residual connection and dropout
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x
    
    def backward(self, grad_output, memory=None):
        """
        Backward pass for the decoder block.
        Args:
            grad_output (torch.Tensor): Gradient of the output tensor.
            memory (torch.Tensor): Memory tensor for cross-attention.
        Returns:
            (torch.Tensor, torch.Tensor) : Gradient of the input tensor.
        """
        if memory is None:
            raise ValueError("Memory tensor must be provided for decoder block.")
        
        # Backward pass through feed-forward
        grad_output = self.norm.backward(grad_output)
        grad_ff = self.feed_forward.backward(grad_output)
        grad_x = grad_output + grad_ff

        # Backward pass through cross-attention
        grad_output = self.norm.backward(grad_x)
        grad_x, grad_memory = self.cross_attention.backward(grad_output, memory)
        
        # Backward pass through self-attention
        grad_output = self.norm.backward(grad_x)
        grad_attn = self.self_attention.backward(grad_output)
        grad_x = grad_attn + grad_output 
        
        return grad_x, grad_memory

class Decoder(nn.Module):
    def __init__(self, embed_dim, ffn_dim, head_num, k_dim, v_dim, N, dropout=0.1):
        super().__init__()
        self.blocks = nn.Sequential()
        for i in range(N):
            self.blocks.add_module(f"decoder_block_{i}", DecoderBlock(embed_dim, ffn_dim, head_num, k_dim, v_dim, dropout=dropout))

    def forward(self, x, memory=None):
        """
        Forward pass for the decoder.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """

        for block in self.blocks:
            x = block(x, memory)
        return x
    
    def backward(self, grad_output, memory=None):
        """
        Backward pass for the decoder.
        Args:
            grad_output (torch.Tensor): Gradient of the output tensor.
            memory (torch.Tensor): Memory tensor for cross-attention.
        Returns:
            (torch.Tensor, torch.Tensor): Gradient of the input tensor.
        """
        if memory is None:
            raise ValueError("Memory tensor must be provided for decoder.")
        
        grad_memory_sum = torch.zeros_like(memory)
        
        for block in reversed(self.blocks):
            grad_output, grad_memory = block.backward(grad_output, memory)
            grad_memory_sum += grad_memory

        return grad_output, grad_memory_sum

class Transformer(nn.Module):
    def __init__(self,
                 device,
                 eos_token_id,
                 bos_token_id,
                 token_size = 10000, 
                 embed_dim = 512, 
                 ffn_dim = 2048, 
                 head_num = 8, 
                 k_dim = 64, 
                 v_dim = 64, 
                 N = 6, 
                 model_name = "Transformer",
                 tokenizer_name = "BytePairEncoder",
                 dropout = 0.1):
        
        super().__init__()
        
        # Store all configuration parameters for saving/loading the model.
        self.config = {
            'device': device,
            'eos_token_id': eos_token_id,
            'bos_token_id': bos_token_id,
            'token_size': token_size,
            'embed_dim': embed_dim,
            'ffn_dim': ffn_dim,
            'head_num': head_num,
            'k_dim': k_dim,
            'v_dim': v_dim,
            'N': N,
            'model_name': model_name,
            'tokenizer_name': tokenizer_name,
            'dropout': dropout
        }
        
        # Also store as individual attributes for backward compatibility
        self.device = device
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.token_size = token_size
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.head_num = head_num
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.N = N
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.dropout = dropout
        self.embedding = Embedding(token_size, embed_dim = embed_dim)
        self.positional_encoding = positional_encoder.AbsoluteSinusoidalPositionalEmbedding(embed_dim) 

        self.encoder = Encoder(embed_dim, ffn_dim, head_num, k_dim, v_dim, N, dropout=dropout)
        self.decoder = Decoder(embed_dim, ffn_dim, head_num, k_dim, v_dim, N, dropout=dropout)

        self.output_layer = nn.Linear(embed_dim, token_size)  
        self.dropout_layer = nn.Dropout(dropout)  # Dropout for embedding layer

        self.inference_method = inference_method.TransformerInference(self.decoder, eos_token_id, model=self)

    def forward(self, input_token_ids, output_token_ids, return_logits=False):
        """
        Forward pass for the transformer model. It's for training.
        If you want to use the model for inference, please use the inference method.

        This method takes tokenized input and output sequences.
        This method proceeds through the following steps:
        1. Embedding and positional encoding of input and output sequences.
        2. Applying dropout to the embeddings.
        3. Passing the input through the encoder.
        4. Passing the output through the decoder with the encoder's output as memory.
        5. Applying the output layer to get logits.
        6. Optionally returning the logits if return_logits is True.

        Args:
            input_token_ids (torch.Tensor): Input token tensor of shape (batch_size, seq_len).
            output_token_ids (torch.Tensor): Output token tensor of shape (batch_size, seq_len).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
            torch.Tensor (optional): Logits tensor of shape (batch_size, seq_len, token_size) if return_logits is True.
        """
        if input_token_ids.dim() != 2:
            raise ValueError(f"Input tensor input_token_ids must be 2-dimensional, got {input_token_ids.dim()} dimensions.")
        
        if output_token_ids.dim() != 2:
            raise ValueError(f"Output tensor output_token_ids must be 2-dimensional, got {output_token_ids.dim()} dimensions.")
        
        # Embedding and positional encoding
        input_embed = self.embedding(input_token_ids)  
        output_embed = self.embedding(output_token_ids)

        encoder_input = self.positional_encoding(input_embed)
        decoder_input = self.positional_encoding(output_embed)
        
        # Apply dropout to embeddings
        encoder_input = self.dropout_layer(encoder_input)
        decoder_input = self.dropout_layer(decoder_input)

        # Encoder
        encoder_output = self.encoder(encoder_input)

        # Decoder
        decoder_output = self.decoder(decoder_input, memory=encoder_output)

        # Output layer
        output = self.output_layer(decoder_output)  # (batch_size, seq_len, token_size)

        if return_logits:
            return output, logits
        else:
            return output

    def inference(self, input_token_ids, max_length=250, inference_method="greedy"):
        """
        Inference pass for the transformer model.
        If you want to use the model for training, please use the forward method.

        Args:
            input_token_ids (torch.Tensor): Input token tensor of shape (batch_size, seq_len).
            max_length (int): Maximum length of output sequence.
            inference_method (str): Inference method to use.
        Returns:
            torch.Tensor: Generated token IDs of shape (batch_size, generated_length).
        """
        if input_token_ids.dim() != 2:
            raise ValueError(f"Input tensor input_token_ids must be 2-dimensional, got {input_token_ids.dim()} dimensions.")

        batch_size = input_token_ids.size(0)

        # Encode input sequence to get memory
        input_embed = self.embedding(input_token_ids)
        encoder_input = self.positional_encoding(input_embed)
        # Note: During inference, we typically don't apply dropout
        # but we keep the layer for consistency
        if self.training:
            encoder_input = self.dropout_layer(encoder_input)
        memory = self.encoder(encoder_input)

        # Start with BOS token for decoder
        bos_token_ids = torch.full((batch_size, 1), self.bos_token_id, device=self.device)

        # Use the inference method to generate output
        output_tokens_ids = self.inference_method(bos_token_ids, 
                                              memory, 
                                              max_length = max_length, 
                                              inference_method = inference_method)

        return output_tokens_ids
 
    def backward(self, grad_output, memory):
        """
        Backward pass for the transformer model.
        Args:
            grad_output (torch.Tensor): Gradient of the output tensor.
        Returns:
            torch.Tensor: Gradient of the input tensor.
        """
        # Backward pass through decoder
        grad_decoder, grad_memory = self.decoder.backward(grad_output, memory)

        # Backward pass through encoder
        grad_encoder = self.encoder.backward(grad_memory)

        # Backward pass through positional encoding
        grad_positional = self.positional_encoding.backward(grad_encoder)

        # Backward pass through embedding
        grad_x = self.embedding.backward(grad_positional)

        return grad_x
