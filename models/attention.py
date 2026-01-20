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

    def __init__(self, d_k: int, dropout: float = 0.1):
        super(ScaledDotProductAttention, self).__init__()

        self.d_k = d_k
        self.dropout = dropout
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




