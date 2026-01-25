"""
Complete Transformer-based NER Model
Combines all components: embeddings, positional encoding, transformer encoder, classifier
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.transformer_encoder import TransformerEncoder
from models.positional_encoding import PositionalEncoding
import config

class TransformerNER(nn.Module):

    """
    Complete Transformer model for Named Entity Recognition
    
    Architecture:
        Token Embedding
          ↓
        Positional Encoding
          ↓
        Transformer Encoder (N layers)
          ↓
        Classification Head
          ↓
        NER Tags
    
    Args:
        vocab_size: Size of vocabulary
        num_labels: Number of NER tags
        d_model: Model dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        max_seq_length: Maximum sequence length
        dropout_p: Dropout probability
        pad_idx: Padding token index
    """

    def __init__(
        self,
        vocab_size: int = config.VOCAB_SIZE,
        num_labels: int = config.NUM_LABELS,
        d_model: int = config.D_MODEL,
        num_layers: int = config.NUM_LAYERS,
        num_heads: int = config.NUM_HEADS,
        d_ff: int = config.D_FF,
        max_seq_length: int = config.MAX_SEQ_LENGTH,
        dropout_p: float = config.DROPOUT,
        pad_idx: int = config.PAD_IDX
    ):
        super(TransformerNER, self).__init__()

        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.d_model = d_model
        self.pad_idx = pad_idx

        # Token embedding layer
        self.token_embedding = nn.Embedding(
            vocab_size, 
            d_model, 
            padding_idx=pad_idx
        )

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model, 
            max_seq_length, 
            dropout_p
        )

        # Transformer encoder
        self.encoder = TransformerEncoder(
            num_layers, 
            d_model, 
            num_heads, 
            d_ff, 
            dropout_p
        )

        # Dropout
        self.dropout = nn.Dropout(dropout_p)

        # Classification head for NER
        self.classifier = nn.Linear(d_model, num_labels)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)






