
"""
Positional Encoding implementation from scratch
Following "Attention is All You Need" (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):

    """
    Sinusoidal positional encoding
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    where:
        pos: position in the sequence
        i: dimension index
        d_model: model dimension
    
    Args:
        d_model: Dimension of the model
        max_len: Maximum sequence length
        dropout_p: Dropout probability
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout_p: float = 0.1):

        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout_p)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model) # pe matrix of shape (max_len, d_model)

        # create postion indices: [0, 1, 2, ..., max_len-1]
        # shape: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Create division term for the exponential
        # div_term = 1 / (10000^(2i/d_model))
        # Shape: (d_model/2, )
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices in the embedding
        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices in the embedding
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter, but part of state_dict)
        # This means it will be saved/loaded with the model but not updated during training
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added, same shape as input
        """
        # Add positional encoding to input
        # x.size(1) is the sequence length
        x = x + self.pe[:, :x.size(1), :]
        
        return self.dropout(x)

# Test the implementation
if __name__ == "__main__":
    print("=" * 80)
    print("TESTING POSITIONAL ENCODING")
    print("=" * 80)
    
    # Parameters
    batch_size = 2
    seq_len = 10
    d_model = 256
    
    # Create module
    pos_encoding = PositionalEncoding(d_model, max_len=100)
    
    # Create dummy input (token embeddings)
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Expected: ({batch_size}, {seq_len}, {d_model})")
    
    # Apply positional encoding
    output = pos_encoding(x)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected: ({batch_size}, {seq_len}, {d_model})")
    
    # Verify that positional encoding is different for each position
    print(f"\nPositional encoding sample (first position):")
    print(f"  First 10 values: {pos_encoding.pe[0, 0, :10]}")
    
    print(f"\nPositional encoding sample (fifth position):")
    print(f"  First 10 values: {pos_encoding.pe[0, 4, :10]}")
    
    # Visualize positional encoding
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract positional encoding for visualization
    pe_matrix = pos_encoding.pe[0, :50, :].numpy()  # First 50 positions
    
    plt.figure(figsize=(12, 6))
    plt.imshow(pe_matrix.T, cmap='RdBu', aspect='auto')
    plt.colorbar()
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    plt.title('Positional Encoding Visualization (First 50 positions)')
    plt.tight_layout()
    
    # Save visualization
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    import config
    
    save_path = config.VISUALIZATION_DIR / 'positional_encoding.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {save_path}")
    plt.close()
    
    print("\n" + "=" * 80)
    print("✓ POSITIONAL ENCODING TEST COMPLETE")
    print("=" * 80)



