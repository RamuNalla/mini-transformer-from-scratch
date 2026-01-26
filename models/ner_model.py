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

    def create_padding_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Create padding mask for attention
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
        Returns:
            mask: Attention mask (batch_size, 1, seq_len, seq_len)
        """
        # Create mask where padding tokens are 0, others are 1
        # Shape: (batch_size, seq_len)
        mask = (input_ids != self.pad_idx).float()
        
        # Expand for attention: (batch_size, 1, 1, seq_len)
        mask = mask.unsqueeze(1).unsqueeze(2)
        
        # Expand to (batch_size, 1, seq_len, seq_len)
        mask = mask.expand(-1, -1, input_ids.size(1), -1)
        
        return mask

    def forward(
        self, input_ids: torch.Tensor, 
        attention_mask: torch.Tensor = None,
        return_attention: bool = False
    ):
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Optional attention mask (batch_size, 1, seq_len, seq_len)
            return_attention: Whether to return attention weights
            
        Returns:
            logits: NER tag logits (batch_size, seq_len, num_labels)
            attention_weights: (Optional) List of attention weights from each layer
        """
        # Create padding mask if not provided
        if attention_mask is None:
            attention_mask = self.create_padding_mask(input_ids)
        
        # Token embeddings
        # (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        embeddings = self.token_embedding(input_ids)
        
        # Scale embeddings (as in the paper)
        embeddings = embeddings * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        
        # Add positional encoding
        embeddings = self.positional_encoding(embeddings)
        
        # Apply dropout
        embeddings = self.dropout(embeddings)
        
        # Pass through transformer encoder
        encoder_output, attention_weights = self.encoder(embeddings, attention_mask)
        
        # Classification head
        logits = self.classifier(encoder_output)
        
        if return_attention:
            return logits, attention_weights
        
        return logits
    

    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Predict NER tags
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Optional attention mask
            
        Returns:
            predictions: Predicted tag indices (batch_size, seq_len)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1)
        
        return predictions
    
    def get_config(self):
        """Get model configuration"""
        return {
            'vocab_size': self.vocab_size,
            'num_labels': self.num_labels,
            'd_model': self.d_model,
            'num_layers': self.encoder.num_layers,
            'num_heads': config.NUM_HEADS,
            'd_ff': config.D_FF,
            'dropout': config.DROPOUT,
        }
    







