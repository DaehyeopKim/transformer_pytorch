"""
Create a simple test model for inference testing
"""
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.transformer import Transformer
from models.tokenizer import BytePairEncoder
from utils.utils import save_transformer_with_config

def create_test_model():
    """Create and save a small test model"""
    device = torch.device("cpu")  # Use CPU for simplicity
    tokenizer = BytePairEncoder(device=device)

    # Create a small model for testing
    model = Transformer(
        device=device,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        vocab_size=tokenizer.vocab_size,  # Small vocab for testing
        embed_dim=128,    # Small embedding
        ffn_dim=256,      # Small FFN
        head_num=4,       # Few heads
        k_dim=32,         # Small key dim
        v_dim=32,         # Small value dim
        N=2,              # Few layers
        dropout=0.1
    )
    
    # Initialize with random weights (in practice, this would be trained)
    model.apply(lambda m: torch.nn.init.normal_(m.weight, 0, 0.02) if hasattr(m, 'weight') else None)
    
    # Save the model
    save_transformer_with_config(model, "test_model.pth")
    
    print("✅ Test model created and saved as 'test_model.pth'")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test tokenizer
    test_text = "Hello, world!"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Tokenizer test:")
    print(f"  Original: '{test_text}'")
    print(f"  Encoded: {encoded}")
    print(f"  Decoded: '{decoded}'")
    
    return model

if __name__ == "__main__":
    create_test_model()
