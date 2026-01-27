"""
Evaluation metrics for Named Entity Recognition
Uses seqeval for proper entity-level evaluation
"""

import numpy as np
from typing import List, Dict
from seqeval.metrics import (
    f1_score, 
    precision_score, 
    recall_score, 
    classification_report,
    accuracy_score
)
from seqeval.scheme import IOB2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config

def align_predictions(predictions: List[List[int]], 
                     labels: List[List[int]], 
                     lengths: List[int],
                     idx2tag: Dict[int, str] = None) -> tuple:
    """
    Align predictions with labels by removing padding
    
    Args:
        predictions: List of predicted label sequences (with padding)
        labels: List of true label sequences (with padding)
        lengths: List of actual sequence lengths
        idx2tag: Index to tag mapping
        
    Returns:
        Tuple of (aligned_predictions, aligned_labels) as tag strings
    """
    if idx2tag is None:
        idx2tag = config.IDX2TAG
    
    pred_tags = []
    true_tags = []
    
    for pred_seq, label_seq, length in zip(predictions, labels, lengths):
        # Remove padding
        pred_seq = pred_seq[:length]
        label_seq = label_seq[:length]
        
        # Convert indices to tags
        pred_tags.append([idx2tag[idx] for idx in pred_seq])
        true_tags.append([idx2tag[idx] for idx in label_seq])
    
    return pred_tags, true_tags

def compute_metrics(predictions: List[List[int]], 
                   labels: List[List[int]], 
                   lengths: List[int] = None,
                   idx2tag: Dict[int, str] = None,
                   verbose: bool = True) -> Dict[str, float]:
    """
    Compute NER evaluation metrics using seqeval
    
    This uses entity-level evaluation, meaning:
    - A prediction is correct only if BOTH the span and entity type match
    - Partial matches are considered incorrect
    
    Args:
        predictions: List of predicted label sequences
        labels: List of true label sequences
        lengths: List of actual sequence lengths (to remove padding)
        idx2tag: Index to tag mapping
        verbose: Whether to print detailed report
        
    Returns:
        Dictionary with evaluation metrics
    """
    if idx2tag is None:
        idx2tag = config.IDX2TAG
    
    # If lengths not provided, assume all sequences are full length
    if lengths is None:
        lengths = [len(pred) for pred in predictions]
    
    # Align predictions and labels (remove padding, convert to tags)
    pred_tags, true_tags = align_predictions(predictions, labels, lengths, idx2tag)
    
    # Compute metrics using seqeval
    # seqeval computes entity-level metrics (not token-level)
    metrics = {
        'f1': f1_score(true_tags, pred_tags, scheme=IOB2, mode='strict'),
        'precision': precision_score(true_tags, pred_tags, scheme=IOB2, mode='strict'),
        'recall': recall_score(true_tags, pred_tags, scheme=IOB2, mode='strict'),
        'accuracy': accuracy_score(true_tags, pred_tags),
    }
    
    # Get detailed classification report
    if verbose:
        report = classification_report(true_tags, pred_tags, scheme=IOB2, mode='strict')
        metrics['report'] = report
        print("\n" + "=" * 80)
        print("CLASSIFICATION REPORT")
        print("=" * 80)
        print(report)
    
    return metrics

def compute_entity_metrics(predictions: List[List[int]], 
                          labels: List[List[int]], 
                          lengths: List[int] = None,
                          idx2tag: Dict[int, str] = None) -> Dict[str, Dict]:
    """
    Compute per-entity-type metrics
    
    Args:
        predictions: Predicted label sequences
        labels: True label sequences
        lengths: Sequence lengths
        idx2tag: Index to tag mapping
        
    Returns:
        Dictionary with per-entity metrics
    """
    if idx2tag is None:
        idx2tag = config.IDX2TAG
    
    if lengths is None:
        lengths = [len(pred) for pred in predictions]
    
    # Align predictions
    pred_tags, true_tags = align_predictions(predictions, labels, lengths, idx2tag)
    
    # Entity types (exclude 'O' tag)
    entity_types = set()
    for tags in true_tags:
        for tag in tags:
            if tag != 'O':
                entity_type = tag.split('-')[1]  # Extract PER, ORG, LOC, MISC
                entity_types.add(entity_type)
    
    # Compute metrics for each entity type
    entity_metrics = {}
    
    for entity_type in sorted(entity_types):
        # Filter to only this entity type
        filtered_pred = []
        filtered_true = []
        
        for pred_seq, true_seq in zip(pred_tags, true_tags):
            filtered_pred_seq = []
            filtered_true_seq = []
            
            for pred_tag, true_tag in zip(pred_seq, true_seq):
                # Keep only tags for this entity type
                if true_tag.endswith(entity_type) or pred_tag.endswith(entity_type):
                    filtered_pred_seq.append(pred_tag if pred_tag.endswith(entity_type) else 'O')
                    filtered_true_seq.append(true_tag if true_tag.endswith(entity_type) else 'O')
                else:
                    filtered_pred_seq.append('O')
                    filtered_true_seq.append('O')
            
            if filtered_true_seq:  # Only add if not empty
                filtered_pred.append(filtered_pred_seq)
                filtered_true.append(filtered_true_seq)
        
        # Compute metrics for this entity type
        if filtered_true:
            entity_metrics[entity_type] = {
                'f1': f1_score(filtered_true, filtered_pred, scheme=IOB2, mode='strict'),
                'precision': precision_score(filtered_true, filtered_pred, scheme=IOB2, mode='strict'),
                'recall': recall_score(filtered_true, filtered_pred, scheme=IOB2, mode='strict'),
            }
    
    return entity_metrics

def print_metrics_summary(metrics: Dict, entity_metrics: Dict = None):
    """
    Print a formatted summary of metrics
    
    Args:
        metrics: Overall metrics dictionary
        entity_metrics: Per-entity metrics dictionary
    """
    print("\n" + "=" * 80)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 80)
    
    print("\nOverall Performance:")
    print(f"  F1 Score:    {metrics['f1']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    
    if entity_metrics:
        print("\nPer-Entity Performance:")
        print(f"  {'Entity':<10} {'F1':<8} {'Precision':<12} {'Recall':<8}")
        print("  " + "-" * 45)
        
        for entity_type, scores in sorted(entity_metrics.items()):
            print(f"  {entity_type:<10} {scores['f1']:<8.4f} "
                  f"{scores['precision']:<12.4f} {scores['recall']:<8.4f}")
    
    print("=" * 80)


# Test the metrics
if __name__ == "__main__":
    print("=" * 80)
    print("TESTING NER METRICS")
    print("=" * 80)
    
    # Create dummy predictions and labels
    # Simulate predictions for 3 sentences
    
    # Example: "John works at Google in Paris"
    # True: B-PER O O B-ORG O B-LOC
    # Pred: B-PER O O B-ORG O B-LOC (perfect match)
    
    true_labels_1 = [1, 0, 0, 3, 0, 5]  # B-PER, O, O, B-ORG, O, B-LOC
    pred_labels_1 = [1, 0, 0, 3, 0, 5]  # Perfect match
    
    # Example 2: Some errors
    true_labels_2 = [1, 2, 0, 3, 4, 0, 5]  # B-PER, I-PER, O, B-ORG, I-ORG, O, B-LOC
    pred_labels_2 = [1, 0, 0, 3, 4, 0, 5]  # Missed I-PER
    
    # Example 3: More errors
    true_labels_3 = [0, 0, 3, 4, 0]  # O, O, B-ORG, I-ORG, O
    pred_labels_3 = [0, 0, 5, 0, 0]  # Predicted LOC instead of ORG
    
    predictions = [pred_labels_1, pred_labels_2, pred_labels_3]
    labels = [true_labels_1, true_labels_2, true_labels_3]
    lengths = [len(p) for p in predictions]
    
    print("\nComputing metrics...")
    metrics = compute_metrics(predictions, labels, lengths, verbose=True)
    
    print("\nComputing per-entity metrics...")
    entity_metrics = compute_entity_metrics(predictions, labels, lengths)
    
    print_metrics_summary(metrics, entity_metrics)
    
    print("\n" + "=" * 80)
    print("✓ METRICS TEST COMPLETE")
    print("=" * 80)
