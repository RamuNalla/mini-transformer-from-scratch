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