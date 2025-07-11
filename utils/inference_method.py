'''
This module provides various inference methods for the transformer model.
1. greedy search
2. beam search
3. top-k sampling
4. top-p sampling
...
'''
import torch

class TransformerInference:
    def __init__(self, decoder, eos_token_id, model=None):
        self.eos_token_id = eos_token_id
        self.decoder = decoder
        self.model = model  # Reference to the full model
        self.dict = {
            "greedy": self.greedy_search
            # "beam": self.beam_search,
            # "top_k": self.top_k_sampling,
            # "top_p": self.top_p_sampling
        }

    def __call__(self, x, memory, max_length=50, inference_method="greedy"):
        """
        Inference method for the transformer model.
        Args:
            input_token_ids (torch.Tensor): Input token tensor of shape (batch_size, seq_len).
            max_length (int): Maximum length of the output sequence.
            inference_method (str): Inference method to use. Options: "greedy", "beam", "top_k", "top_p".
        Returns:
            torch.Tensor: Output token tensor of shape (batch_size, max_length).
        """
        if inference_method not in self.dict:
            raise ValueError(f"Unsupported inference method: {inference_method}. Supported methods: {list(self.dict.keys())}")
        
        return self.dict[inference_method](x, memory, max_length)
    
    def greedy_search(self, x, memory, max_length):
        """
        Greedy search inference method.
        Args:
            x (torch.Tensor): Input token tensor of shape (batch_size, seq_len).
            memory (torch.Tensor): Memory tensor from the encoder.
            max_length (int): Maximum length of the output sequence.
        Returns:
            torch.Tensor: Output token tensor of shape (batch_size, max_length).
        """
        if self.model is None:
            raise ValueError("Model reference is required for inference")
            
        batch_size = x.size(0)
        current_seq = x  # Start with BOS token
        output_tokens_ids = []
        
        for t in range(max_length):
            # Embedding and positional encoding for current sequence
            decoder_embed = self.model.embedding(current_seq)
            decoder_input = self.model.positional_encoding(decoder_embed)
            
            # Decoder forward pass
            decoder_output = self.decoder(decoder_input, memory=memory)
            
            # Output layer to get logits
            logits = self.model.output_layer(decoder_output)  # (batch_size, seq_len, vocab_size)
            
            # Get the next token (greedy: argmax of last position)
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
            next_token = next_token_logits.argmax(dim=-1)  # (batch_size,)
            
            output_tokens_ids.append(next_token)
            
            # Add next token to current sequence
            current_seq = torch.cat([current_seq, next_token.unsqueeze(1)], dim=1)
            
            # Check for EOS token (simple version - stop if any sequence hits EOS)
            if torch.any(next_token == self.eos_token_id):
                break
        
        # Convert list of tokens to tensor
        if output_tokens_ids:
            # Stack individual tokens into a single tensor
            # output_tokens_ids: [tensor(batch_size), tensor(batch_size), ...]
            # result: tensor(batch_size, generated_length)
            generated_tokens = torch.stack(output_tokens_ids, dim=1)
        else:
            # No tokens generated - return empty tensor with correct shape
            generated_tokens = torch.empty(batch_size, 0, dtype=torch.long, device=x.device)
            
        return generated_tokens