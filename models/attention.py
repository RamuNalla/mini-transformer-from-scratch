"""
Multi-Head Attention implementation from scratch
from "Attention is All You Need" paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    Args:
        d_k: Dimension of key/query
        dropout: Dropout rate
    """ 

    def __init__(self, d_k: int, dropout_p: float = 0.1):
        super(ScaledDotProductAttention, self).__init__()

        self.d_k = d_k
        self.dropout = nn.Dropout(dropout_p)
        self.scale = math.sqrt(d_k)

    def forward(self, Q:torch.tensor, K:torch.tensor, V:torch.tensor, 
                mask: torch.tensor = None):
        
        """
        Forward pass
        
        Args:
            Q: Query tensor (batch_size, num_heads, seq_len, d_k)
            K: Key tensor (batch_size, num_heads, seq_len, d_k)
            V: Value tensor (batch_size, num_heads, seq_len, d_v)
            mask: Attention mask (batch_size, 1, seq_len, seq_len)
            
        Returns:
            output: Attention output (batch_size, num_heads, seq_len, d_v)
            attention_weights: Attention weights (batch_size, num_heads, seq_len, seq_len)
        """

        # Compute attention scores
        # QK^T: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply mask (set masked positions to large negative value)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights) 
        # Randomly zeroes out some attention weights to prevent overfitting

        # Apply attention weights to values
        output = torch.matmul(attention_weights, V)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism
    
    Instead of performing a single attention function with d_model-dimensional keys,
    values and queries, we project them h times with different, learned linear projections.
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.2):

        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads     # Dimension per head
        self.d_v = d_model // num_heads     # Value dimension per head

        # Linear projections for Q, K, V
        # Instead of h separate projections, we use single large projection
        # and split into heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model) 

        # Output projection
        self.W_O = nn.Linear(d_model, d_model)

        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention(self.d_k, dropout_p)

        self.dropout = nn.Dropout(dropout_p)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:

        """
        Split tensor into multiple heads
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Reshaped tensor (batch_size, num_heads, seq_len, d_k)
        """

        batch_size, seq_len, d_model = x.size()

        # Reshape: (batch_size, seq_len, num_heads, d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)

        # Transpose: (batch_size, num_heads, seq_len, d_k)
        x = x.transpose(1, 2)

        return x
    
    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine multiple heads back
        
        Args:
            x: Input tensor (batch_size, num_heads, seq_len, d_k)
            
        Returns:
            Combined tensor (batch_size, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.size()
        
        # Transpose: (batch_size, seq_len, num_heads, d_k)
        x = x.transpose(1, 2).contiguous()
        
        # Reshape: (batch_size, seq_len, d_model)
        x = x.view(batch_size, seq_len, self.d_model)
        
        return x

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.tensor, mask: torch.tensor = None):

        """
        Forward pass
        
        Args:
            Q: Query tensor (batch_size, seq_len, d_model)
            K: Key tensor (batch_size, seq_len, d_model)
            V: Value tensor (batch_size, seq_len, d_model)
            mask: Attention mask (batch_size, 1, seq_len, seq_len)
            
        Returns:
            output: Multi-head attention output (batch_size, seq_len, d_model)
            attention_weights: Attention weights (batch_size, num_heads, seq_len, seq_len)
        """

        batch_size = Q.size(0)

        # 1. Linear projections
        Q = self.W_Q(Q)  # (batch_size, seq_len, d_model)
        K = self.W_K(K)  # (batch_size, seq_len, d_model)
        V = self.W_V(V)  # (batch_size, seq_len, d_model)

        # 2. Split into multiple heads
        Q = self.split_heads(Q)  # (batch_size, num_heads, seq_len, d_k)
        K = self.split_heads(K)  # (batch_size, num_heads, seq_len, d_k)
        V = self.split_heads(V)  # (batch_size, num_heads, seq_len, d_v)

        # 3. Apply attention
        attn_output, attention_weights = self.attention(Q, K, V, mask)
        # attn_output: (batch_size, num_heads, seq_len, d_v)
        # attention_weights: (batch_size, num_heads, seq_len, seq_len)

         # 4. Combine heads
        attn_output = self.combine_heads(attn_output)
        # (batch_size, seq_len, d_model)

        # 5. Final linear projection
        output = self.W_O(attn_output)
        output = self.dropout(output)
        
        return output, attention_weights

# Test the implementation
if __name__ == "__main__":
    print("=" * 80)
    print("TESTING MULTI-HEAD ATTENTION")
    print("=" * 80)

    # Parameters
    batch_size = 2
    seq_len = 10
    d_model = 256
    num_heads = 8
    
    # Create module
    mha = MultiHeadAttention(d_model, num_heads)

    # Create dummy input
    Q = torch.randn(batch_size, seq_len, d_model)
    K = torch.randn(batch_size, seq_len, d_model)
    V = torch.randn(batch_size, seq_len, d_model)

    # Forward pass
    output, attention_weights = mha(Q, K, V)

    print(f"\nOutput shapes:")
    print(f"  Output: {output.shape}")
    print(f"  Attention weights: {attention_weights.shape}")
    
    print(f"\nExpected:")
    print(f"  Output: ({batch_size}, {seq_len}, {d_model})")
    print(f"  Attention: ({batch_size}, {num_heads}, {seq_len}, {seq_len})")

    # Test with mask
    print(f"\nTesting with mask...")
    mask = torch.ones(batch_size, 1, seq_len, seq_len)
    mask[:, :, :, 5:] = 0  # Mask last 5 positions
    
    output_masked, attn_masked = mha(Q, K, V, mask)
    print(f"  Masked output: {output_masked.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in mha.parameters())
    print(f"\nTotal parameters: {num_params:,}")
    
    print("\n" + "=" * 80)
    print("✓ MULTI-HEAD ATTENTION TEST COMPLETE")
    print("=" * 80)

    


    





