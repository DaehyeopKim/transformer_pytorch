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
    def __init__(self, decoder, eos_token_id):
        self.eos_token_id = eos_token_id
        self.deocder = decoder
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
        
        return self.dict[inference_method](x, max_length)
    
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
        batch_size = x.size(0)
        output_tokens_ids = torch.zeros(batch_size, max_length, dtype=torch.long).to(x.device)
        
        for t in range(max_length):
            output = self.decoder(x, memory)
            next_token = output.argmax(dim=-1)[:, -1]
            output_tokens_ids[:, t] = next_token
            x = torch.cat([x, next_token.unsqueeze(1)], dim=1)
            if next_token == self.eos_token_id:
                break
        
        return output_tokens_ids