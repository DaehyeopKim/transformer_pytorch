from utils import positional_encoder
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import torch
import numpy as np
import os

pe_classas = {
    "ASPE": positional_encoder.AbsoluteSinusoidalPositionalEmbedding,
    "RoPE": positional_encoder.RotaryPositionalEmbedding
}


def visualize_positional_encoding(path, pe_name, dim, max_len=5000, b = 10000.0):
    """
    Visualize the positional encoding for a given dimension and maximum length.
    Args:
        dim (int): Dimension of the positional encoding.
        max_len (int): Maximum length of the sequence.
        b (float): Base for the sinusoidal encoding.
    """
    # Create a dummy input tensor
    max_len = max_len // 10
    x = torch.zeros(1, max_len, dim)

    check_dim = [0] + list(range(dim // 4 - 1, dim, dim // 4))

    # Choose the positional encoder
    if pe_name not in pe_classas:
        raise ValueError(f"Unknown positional encoder: {pe_name}. Choose from {list(pe_classas.keys())}.")
    
    pos_encoder = pe_classas[pe_name](dim, max_len, b)
    
    # Get the positional encoding
    pe = pos_encoder(x)

    # Convert to numpy for visualization
    pe_np = pe.squeeze(0).numpy()

    # Create figure with subplots
    plt.figure(figsize=(15, 6))
    
    # Plot the positional encoding    
    plt.subplot(1, 2, 1)
    plt.imshow(pe_np.T, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title(f'Positional Encoding (dim={dim}, max_len={max_len})')
    plt.xlabel('Position')
    plt.ylabel('Dimension')

    # Plot the positional encoding for specific dimensions
    plt.subplot(1, 2, 2)
    for i in check_dim:
        plt.plot(pe_np[:, i], label=f'Dim {i}')
    plt.title(f'Positional Encoding for specific dimensions (dim={dim}, max_len={max_len})')
    plt.xlabel('Position')
    plt.ylabel('Positional Encoding Value')

    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Ensure path exists and ends with /
    os.makedirs(path, exist_ok=True)
    if not path.endswith('/'):
        path += '/'
    
    filename = f'positional_encoding_{pe_name}_dim{dim}_maxlen{max_len}.png'
    filepath = path + filename
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Positional encoding visualization saved to {filepath}")
