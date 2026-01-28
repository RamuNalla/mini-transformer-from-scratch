"""
Inference script for Transformer NER
Predict entities on custom text
"""

import torch
import argparse
import pickle
from typing import List, Tuple
from pathlib import Path

import config
from models.ner_model import TransformerNER


class NERPredictor:
    """
    NER prediction class
    """
    
    def __init__(self, model_path: Path, vocab_path: Path, device: torch.device = None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to saved model
            vocab_path: Path to saved vocabulary
            device: Device to run on
        """
        self.device = device if device else config.DEVICE
        
        # Load vocabulary
        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
            self.word2idx = vocab_data['word2idx']
            self.tag2idx = vocab_data['tag2idx']
            self.idx2tag = {v: k for k, v in self.tag2idx.items()}
        
        # Load model - FIXED: Add weights_only=False for PyTorch 2.6+
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.model = TransformerNER(vocab_size=len(self.word2idx))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded from {model_path}")
        print(f"✓ Vocabulary size: {len(self.word2idx)}")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization (split by whitespace and lowercase)
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return text.lower().split()
    
    def predict(self, text: str, return_tokens: bool = False) -> List[Tuple[str, str]]:
        """
        Predict entities in text
        
        Args:
            text: Input text
            return_tokens: Whether to return token-level predictions
            
        Returns:
            List of (entity, tag) tuples or list of (token, tag) tuples
        """
        # Tokenize
        tokens = self.tokenize(text)
        
        # Convert to indices
        word_ids = [
            self.word2idx.get(token, self.word2idx[config.UNK_TOKEN])
            for token in tokens
        ]
        
        # Create tensor
        input_ids = torch.tensor([word_ids], dtype=torch.long).to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model.predict(input_ids)
        
        # Convert predictions to tags
        pred_tags = [self.idx2tag[idx] for idx in predictions[0].tolist()]
        
        if return_tokens:
            # Return token-level predictions
            return list(zip(tokens, pred_tags))
        else:
            # Extract entities
            return self.extract_entities(tokens, pred_tags)
    
    def extract_entities(self, tokens: List[str], tags: List[str]) -> List[Tuple[str, str, float]]:
        """
        Extract entities from BIO tags
        
        Args:
            tokens: List of tokens
            tags: List of BIO tags
            
        Returns:
            List of (entity_text, entity_type, confidence) tuples
        """
        entities = []
        current_entity = []
        current_tag = None
        
        for i, (token, tag) in enumerate(zip(tokens, tags)):
            if tag.startswith('B-'):
                # Start of new entity
                if current_entity:
                    # Save previous entity
                    entity_text = ' '.join(current_entity)
                    entities.append((entity_text, current_tag, 1.0))
                
                # Start new entity
                current_entity = [token]
                current_tag = tag[2:]  # Remove 'B-' prefix
                
            elif tag.startswith('I-'):
                # Inside entity
                if current_tag and tag[2:] == current_tag:
                    current_entity.append(token)
                else:
                    # Tag mismatch, start new entity
                    if current_entity:
                        entity_text = ' '.join(current_entity)
                        entities.append((entity_text, current_tag, 1.0))
                    current_entity = [token]
                    current_tag = tag[2:]
            
            else:  # 'O' tag
                # End of entity
                if current_entity:
                    entity_text = ' '.join(current_entity)
                    entities.append((entity_text, current_tag, 1.0))
                    current_entity = []
                    current_tag = None
        
        # Don't forget last entity
        if current_entity:
            entity_text = ' '.join(current_entity)
            entities.append((entity_text, current_tag, 1.0))
        
        return entities
    
    def predict_with_probabilities(self, text: str) -> List[Tuple[str, str, List[float]]]:
        """
        Predict entities with token-level probabilities
        
        Args:
            text: Input text
            
        Returns:
            List of (entity_text, entity_type, probabilities) tuples
        """
        # Tokenize
        tokens = self.tokenize(text)
        
        # Convert to indices
        word_ids = [
            self.word2idx.get(token, self.word2idx[config.UNK_TOKEN])
            for token in tokens
        ]
        
        # Create tensor
        input_ids = torch.tensor([word_ids], dtype=torch.long).to(self.device)
        
        # Get logits
        with torch.no_grad():
            logits = self.model(input_ids)
            probabilities = torch.softmax(logits, dim=-1)
        
        # Get predictions and probabilities
        predictions = torch.argmax(logits, dim=-1)
        pred_tags = [self.idx2tag[idx] for idx in predictions[0].tolist()]
        probs = probabilities[0].cpu().numpy()
        
        # Extract entities with probabilities
        entities = []
        current_entity = []
        current_tag = None
        current_probs = []
        
        for i, (token, tag) in enumerate(zip(tokens, pred_tags)):
            if tag.startswith('B-'):
                if current_entity:
                    entity_text = ' '.join(current_entity)
                    avg_prob = sum(current_probs) / len(current_probs)
                    entities.append((entity_text, current_tag, avg_prob))
                
                current_entity = [token]
                current_tag = tag[2:]
                current_probs = [probs[i].max()]
                
            elif tag.startswith('I-') and current_tag:
                current_entity.append(token)
                current_probs.append(probs[i].max())
            
            else:
                if current_entity:
                    entity_text = ' '.join(current_entity)
                    avg_prob = sum(current_probs) / len(current_probs)
                    entities.append((entity_text, current_tag, avg_prob))
                    current_entity = []
                    current_tag = None
                    current_probs = []
        
        if current_entity:
            entity_text = ' '.join(current_entity)
            avg_prob = sum(current_probs) / len(current_probs)
            entities.append((entity_text, current_tag, avg_prob))
        
        return entities


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='NER Inference')
    parser.add_argument('--text', type=str, default=None,
                       help='Text to analyze')
    parser.add_argument('--model', type=str, default=str(config.MODEL_SAVE_PATH),
                       help='Path to model')
    parser.add_argument('--vocab', type=str, default=str(config.VOCAB_SAVE_PATH),
                       help='Path to vocabulary')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode')
    
    args = parser.parse_args()
    
    # Create predictor
    print("=" * 80)
    print("TRANSFORMER NER INFERENCE")
    print("=" * 80)
    
    predictor = NERPredictor(
        Path(args.model),
        Path(args.vocab)
    )
    
    if args.interactive:
        # Interactive mode
        print("\n" + "=" * 80)
        print("INTERACTIVE MODE")
        print("=" * 80)
        print("Enter text to analyze (or 'quit' to exit)")
        print("=" * 80 + "\n")
        
        while True:
            text = input("\n> ")
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not text.strip():
                continue
            
            # Predict
            entities = predictor.predict_with_probabilities(text)
            
            # Display results
            print(f"\nInput: {text}")
            
            if entities:
                print("\nEntities found:")
                for entity_text, entity_type, prob in entities:
                    print(f"  - {entity_text:30s} [{entity_type:4s}] (confidence: {prob:.3f})")
            else:
                print("\nNo entities found.")
            
            # Also show token-level predictions
            token_preds = predictor.predict(text, return_tokens=True)
            print("\nToken-level predictions:")
            for token, tag in token_preds:
                print(f"  {token:15s} -> {tag}")
    
    else:
        # Single prediction
        if args.text is None:
            # Use default examples
            examples = [
                "Apple Inc. was founded by Steve Jobs in California.",
                "Barack Obama was born in Hawaii.",
                "Google acquired YouTube in 2006.",
                "The Eiffel Tower is located in Paris, France.",
                "Microsoft CEO Satya Nadella announced new products."
            ]
        else:
            examples = [args.text]
        
        print("\n" + "=" * 80)
        print("PREDICTIONS")
        print("=" * 80)
        
        for text in examples:
            print(f"\nInput: {text}")
            
            # Predict
            entities = predictor.predict_with_probabilities(text)
            
            if entities:
                print("\nEntities found:")
                for entity_text, entity_type, prob in entities:
                    print(f"  - {entity_text:30s} [{entity_type:4s}] (confidence: {prob:.3f})")
            else:
                print("\nNo entities found.")
            
            print("-" * 80)


if __name__ == "__main__":
    main()