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

# Test the implementation
if __name__ == "__main__":
    print("=" * 80)
    print("TESTING COMPLETE TRANSFORMER NER MODEL")
    print("=" * 80)
    
    # Parameters
    batch_size = 2
    seq_len = 20
    vocab_size = 1000
    num_labels = 9  # CoNLL-2003 has 9 tags
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Number of labels: {num_labels}")
    print(f"  Model dimension: {config.D_MODEL}")
    print(f"  Number of layers: {config.NUM_LAYERS}")
    print(f"  Number of heads: {config.NUM_HEADS}")
    
    # Create model
    model = TransformerNER(vocab_size=vocab_size, num_labels=num_labels)
    
    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Simulate padding (last 5 tokens are padding)
    input_ids[:, -5:] = config.PAD_IDX
    
    print(f"\n" + "-" * 80)
    print("Testing Forward Pass:")
    print("-" * 80)
    
    # Forward pass
    logits = model(input_ids)
    
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Expected: ({batch_size}, {seq_len}, {num_labels})")
    
    assert logits.shape == (batch_size, seq_len, num_labels), "Output shape mismatch"
    print("  ✓ Shape test passed!")
    
    # Test with attention weights
    print(f"\n" + "-" * 80)
    print("Testing with Attention Weights:")
    print("-" * 80)
    
    logits, attention_weights = model(input_ids, return_attention=True)
    
    print(f"  Number of attention weight matrices: {len(attention_weights)}")
    print(f"  Each attention weight shape: {attention_weights[0].shape}")
    print("  ✓ Attention weights test passed!")
    
    # Test prediction
    print(f"\n" + "-" * 80)
    print("Testing Prediction:")
    print("-" * 80)
    
    predictions = model.predict(input_ids)
    
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Expected: ({batch_size}, {seq_len})")
    print(f"  Sample predictions: {predictions[0, :10]}")
    
    assert predictions.shape == (batch_size, seq_len), "Predictions shape mismatch"
    print("  ✓ Prediction test passed!")
    
    # Count parameters
    print(f"\n" + "-" * 80)
    print("Model Statistics:")
    print("-" * 80)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Get model configuration
    print(f"\n" + "-" * 80)
    print("Model Configuration:")
    print("-" * 80)
    
    model_config = model.get_config()
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print("✓ COMPLETE NER MODEL TEST PASSED")

    







