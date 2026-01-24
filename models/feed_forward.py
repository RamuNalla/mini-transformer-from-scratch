"""
Position-wise Feed-Forward Network
Implemented from scratch following "Attention is All You Need"
"""

import torch
import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    In the paper, they use ReLU activation, but we use GELU which performs slightly better in practice (used in BERT and GPT).
    
    The FFN is applied to each position separately and identically.
    This consists of two linear transformations with an activation in between.
    
    Args:
        d_model: Model dimension
        d_ff: Hidden dimension of feed-forward network (usually 4 * d_model)
        dropout_p: Dropout probability
    """

    def __init__(self, d_model: int, d_ff: int, dropout_p: float = 0.1):

        super(PositionWiseFeedForward, self).__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        
        # First linear transformation: d_model -> d_ff
        self.linear1 = nn.Linear(d_model, d_ff)
        
        # Second linear transformation: d_ff -> d_model
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_p)
        
        # Activation function (GELU is smoother than ReLU)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # First linear transformation + activation
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Second linear transformation
        # (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        x = self.linear2(x)
        
        return x
    
# Test the implementation
if __name__ == "__main__":
    print("=" * 80)
    print("TESTING POSITION-WISE FEED-FORWARD NETWORK")
    print("=" * 80)
    
    # Parameters
    batch_size = 2
    seq_len = 10
    d_model = 256
    d_ff = 1024
    
    # Create module
    ffn = PositionWiseFeedForward(d_model, d_ff)
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Expected: ({batch_size}, {seq_len}, {d_model})")
    
    # Forward pass
    output = ffn(x)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected: ({batch_size}, {seq_len}, {d_model})")
    
    # Verify output shape matches input shape
    assert output.shape == x.shape, "Output shape should match input shape"
    print("\n✓ Shape test passed!")
    
    # Count parameters
    num_params = sum(p.numel() for p in ffn.parameters())
    print(f"\nTotal parameters: {num_params:,}")
    print(f"Expected: {d_model * d_ff + d_ff + d_ff * d_model + d_model:,}")
    # W1: d_model * d_ff + d_ff (weights + bias)
    # W2: d_ff * d_model + d_model (weights + bias)
    
    print("\n" + "=" * 80)
    print("✓ FEED-FORWARD NETWORK TEST COMPLETE")
    print("=" * 80)
