"""
Transformer Encoder Layer and Stack
Complete implementation from scratch
"""

import torch
import torch.nn as nn
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.attention import MultiHeadAttention
from models.feed_forward import PositionWiseFeedForward

class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer
    
    Architecture:
        Input
          ↓
        Multi-Head Attention
          ↓
        Add & Norm (Residual Connection + Layer Normalization)
          ↓
        Feed-Forward Network
          ↓
        Add & Norm (Residual Connection + Layer Normalization)
          ↓
        Output
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout_p: Dropout probability
    """ 

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_p: float = 0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout_p)
        
        # Position-wise feed-forward network
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout_p)
        
        # Layer normalization (applied before sublayers - Pre-LN)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Attention mask (batch_size, 1, seq_len, seq_len)
            
        Returns:
            output: Output tensor (batch_size, seq_len, d_model)
            attention_weights: Attention weights (batch_size, num_heads, seq_len, seq_len)
        """
        # Multi-head attention with residual connection and layer norm
        # Pre-LN: LayerNorm is applied before the sublayer
        attn_output, attention_weights = self.self_attention(
            self.norm1(x), self.norm1(x), self.norm1(x), mask
        )
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x, attention_weights

class TransformerEncoder(nn.Module):
    """
    Stack of N Transformer Encoder Layers
    
    Args:
        num_layers: Number of encoder layers
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout_p: Dropout probability
    """

    def __init__(self, num_layers: int, d_model: int, num_heads: int, 
                 d_ff: int, dropout_p: float = 0.1):
        super(TransformerEncoder, self).__init__()
        
        self.num_layers = num_layers
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout_p)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass through all encoder layers
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Attention mask (batch_size, 1, seq_len, seq_len)
            
        Returns:
            output: Output tensor (batch_size, seq_len, d_model)
            attention_weights_list: List of attention weights from each layer
        """
        attention_weights_list = []
        
        # Pass through each encoder layer
        for layer in self.layers:
            x, attention_weights = layer(x, mask)
            attention_weights_list.append(attention_weights)
        
        # Final layer normalization
        x = self.norm(x)
        
        return x, attention_weights_list

# Test the implementation
if __name__ == "__main__":
    print("=" * 80)
    print("TESTING TRANSFORMER ENCODER")
    print("=" * 80)
    
    # Parameters
    batch_size = 2
    seq_len = 10
    d_model = 256
    num_heads = 8
    d_ff = 1024
    num_layers = 4
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension: {d_model}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Feed-forward dim: {d_ff}")
    print(f"  Number of layers: {num_layers}")
    
    # Test single encoder layer
    print("\n" + "-" * 80)
    print("Testing Single Encoder Layer:")
    print("-" * 80)
    
    encoder_layer = TransformerEncoderLayer(d_model, num_heads, d_ff)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output, attn_weights = encoder_layer(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Attention weights shape: {attn_weights.shape}")
    
    assert output.shape == x.shape, "Output shape should match input shape"
    print("  ✓ Single layer test passed!")
    
    # Test full encoder stack
    print("\n" + "-" * 80)
    print("Testing Full Encoder Stack:")
    print("-" * 80)
    
    encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff)
    
    output, attn_weights_list = encoder(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Number of attention weight matrices: {len(attn_weights_list)}")
    print(f"  Each attention weight shape: {attn_weights_list[0].shape}")
    
    assert output.shape == x.shape, "Output shape should match input shape"
    assert len(attn_weights_list) == num_layers, f"Should have {num_layers} attention weight matrices"
    print("  ✓ Full encoder test passed!")
    
    # Test with padding mask
    print("\n" + "-" * 80)
    print("Testing with Padding Mask:")
    print("-" * 80)
    
    # Create padding mask (mask last 3 positions)
    mask = torch.ones(batch_size, 1, seq_len, seq_len)
    mask[:, :, :, -3:] = 0
    
    output_masked, attn_masked = encoder(x, mask)
    
    print(f"  Masked output shape: {output_masked.shape}")
    print(f"  Mask shape: {mask.shape}")
    print("  ✓ Mask test passed!")
    
    # Count parameters
    print("\n" + "-" * 80)
    print("Parameter Count:")
    print("-" * 80)
    
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Breakdown by component
    single_layer_params = sum(p.numel() for p in encoder_layer.parameters())
    print(f"  Parameters per layer: {single_layer_params:,}")
    print(f"  Expected total: ~{single_layer_params * num_layers:,}")
    
    print("\n" + "=" * 80)
    print("✓ TRANSFORMER ENCODER TEST COMPLETE")
    print("=" * 80)