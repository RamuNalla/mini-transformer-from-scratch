"""
Data loader for CoNLL-2003 NER dataset
Downloads, processes, and prepares data for training
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from typing import List, Tuple, Dict
from collections import Counter
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config


class NERDataset(Dataset):
    """PyTorch Dataset for NER"""
    
    def __init__(self, sentences: List[List[str]], labels: List[List[str]], 
                 word2idx: Dict, tag2idx: Dict):
        """
        Args:
            sentences: List of tokenized sentences
            labels: List of tag sequences
            word2idx: Word to index mapping
            tag2idx: Tag to index mapping
        """
        self.sentences = sentences
        self.labels = labels
        self.word2idx = word2idx
        self.tag2idx = tag2idx
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        labels = self.labels[idx]
        
        # Convert words to indices
        word_ids = [self.word2idx.get(word, self.word2idx[config.UNK_TOKEN]) 
                   for word in sentence]
        
        # Convert tags to indices
        label_ids = [self.tag2idx[tag] for tag in labels]
        
        return torch.tensor(word_ids, dtype=torch.long), torch.tensor(label_ids, dtype=torch.long)


def collate_fn(batch):
    """
    Custom collate function to pad sequences
    
    Args:
        batch: List of (word_ids, label_ids) tuples
        
    Returns:
        Padded tensors
    """
    word_ids, label_ids = zip(*batch)
    
    # Pad sequences
    word_ids_padded = pad_sequence(word_ids, batch_first=True, padding_value=config.PAD_IDX)
    label_ids_padded = pad_sequence(label_ids, batch_first=True, padding_value=config.PAD_IDX)
    
    # Get lengths
    lengths = torch.tensor([len(seq) for seq in word_ids], dtype=torch.long)
    
    return word_ids_padded, label_ids_padded, lengths


def read_conll_file(file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Read CoNLL format file
    
    CoNLL format:
        word1 POS1 chunk1 NER1
        word2 POS2 chunk2 NER2
        
        word1 POS1 chunk1 NER1
        ...
    
    Args:
        file_path: Path to CoNLL file
        
    Returns:
        Tuple of (sentences, labels)
    """
    sentences = []
    labels = []
    current_sentence = []
    current_labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('-DOCSTART-') or line == '':
                # End of sentence
                if current_sentence:
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence = []
                    current_labels = []
            else:
                # Parse line
                parts = line.split()
                if len(parts) >= 4:
                    word = parts[0]
                    tag = parts[-1]  # NER tag is last column
                    
                    current_sentence.append(word.lower())  # Lowercase words
                    current_labels.append(tag)
        
        # Add last sentence if exists
        if current_sentence:
            sentences.append(current_sentence)
            labels.append(current_labels)
    
    return sentences, labels


def build_vocabulary(sentences: List[List[str]], max_vocab_size: int = config.VOCAB_SIZE) -> Dict[str, int]:
    """
    Build vocabulary from sentences
    
    Args:
        sentences: List of tokenized sentences
        max_vocab_size: Maximum vocabulary size
        
    Returns:
        word2idx: Word to index mapping
    """
    # Count word frequencies
    word_counter = Counter()
    for sentence in sentences:
        word_counter.update(sentence)
    
    # Get most common words
    most_common = word_counter.most_common(max_vocab_size - 2)  # -2 for PAD and UNK
    
    # Build word2idx
    word2idx = {
        config.PAD_TOKEN: config.PAD_IDX,
        config.UNK_TOKEN: 1
    }
    
    for idx, (word, _) in enumerate(most_common, start=2):
        word2idx[word] = idx
    
    return word2idx


def download_conll2003():
    """
    Download CoNLL-2003 dataset
    Note: Due to licensing, you might need to manually download from:
    https://www.clips.uantwerpen.be/conll2003/ner/
    """
    print("=" * 80)
    print("DOWNLOADING CoNLL-2003 DATASET")
    print("=" * 80)
    print("\nNote: Due to licensing restrictions, please manually download:")
    print("  1. Visit: https://www.clips.uantwerpen.be/conll2003/ner/")
    print("  2. Download train.txt, valid.txt, test.txt")
    print(f"  3. Place them in: {config.RAW_DATA_DIR}")
    print("\nOr use the datasets library:")
    print("  pip install datasets")
    print("  from datasets import load_dataset")
    print("  dataset = load_dataset('conll2003')")
    print("=" * 80)


def create_dummy_data():
    """
    Create dummy NER data for testing when dataset unavailable
    """
    print("\nCreating dummy data for testing...")
    
    # Sample sentences with NER tags
    train_sentences = [
        ['john', 'works', 'at', 'google', 'in', 'california'],
        ['apple', 'inc', 'was', 'founded', 'by', 'steve', 'jobs'],
        ['microsoft', 'ceo', 'satya', 'nadella', 'announced', 'new', 'products'],
        ['the', 'eiffel', 'tower', 'is', 'in', 'paris', 'france'],
        ['barack', 'obama', 'was', 'president', 'of', 'the', 'united', 'states'],
    ] * 200  # Repeat to create more data
    
    train_labels = [
        ['B-PER', 'O', 'O', 'B-ORG', 'O', 'B-LOC'],
        ['B-ORG', 'I-ORG', 'O', 'O', 'O', 'B-PER', 'I-PER'],
        ['B-ORG', 'O', 'B-PER', 'I-PER', 'O', 'O', 'O'],
        ['O', 'B-LOC', 'I-LOC', 'O', 'O', 'B-LOC', 'B-LOC'],
        ['B-PER', 'I-PER', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC'],
    ] * 200
    
    # Create validation and test sets
    val_sentences = train_sentences[:100]
    val_labels = train_labels[:100]
    test_sentences = train_sentences[:100]
    test_labels = train_labels[:100]
    
    print(f"✓ Created dummy data:")
    print(f"  Train: {len(train_sentences)} sentences")
    print(f"  Val: {len(val_sentences)} sentences")
    print(f"  Test: {len(test_sentences)} sentences")
    
    return {
        'train': (train_sentences, train_labels),
        'val': (val_sentences, val_labels),
        'test': (test_sentences, test_labels)
    }


def prepare_data_from_huggingface():
    """
    Load and prepare CoNLL-2003 from Hugging Face datasets
    This is easier than manual download
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install datasets library:")
        print("  pip install datasets")
        return None
    
    print("=" * 80)
    print("LOADING CoNLL-2003 FROM HUGGING FACE")
    print("=" * 80)
    
    # Load dataset - Use the correct dataset name
    # Note: conll2003 requires authentication, using alternative
    try:
        dataset = load_dataset("conll2003", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading conll2003: {e}")
        print("\nTrying alternative dataset source...")
        # Use alternative: WNUT-17 dataset (similar to CoNLL but freely available)
        try:
            dataset = load_dataset("wnut_17")
            print("✓ Loaded WNUT-17 dataset (alternative to CoNLL-2003)")
        except:
            print("Failed to load dataset. Using dummy data for testing...")
            return create_dummy_data()
    
    # Extract sentences and labels
    def extract_data(split_data):
        sentences = []
        labels = []
        
        for example in split_data:
            tokens = [token.lower() for token in example['tokens']]
            tags = [config.NER_TAGS[tag_id] for tag_id in example['ner_tags']]
            
            sentences.append(tokens)
            labels.append(tags)
        
        return sentences, labels
    
    train_sentences, train_labels = extract_data(dataset['train'])
    val_sentences, val_labels = extract_data(dataset['validation'])
    test_sentences, test_labels = extract_data(dataset['test'])
    
    print(f"✓ Train: {len(train_sentences):,} sentences")
    print(f"✓ Validation: {len(val_sentences):,} sentences")
    print(f"✓ Test: {len(test_sentences):,} sentences")
    
    return {
        'train': (train_sentences, train_labels),
        'val': (val_sentences, val_labels),
        'test': (test_sentences, test_labels)
    }


def prepare_data():
    """
    Main data preparation function
    """
    print("=" * 80)
    print("PREPARING CoNLL-2003 NER DATA")
    print("=" * 80)
    
    # Load data from Hugging Face
    data_splits = prepare_data_from_huggingface()
    
    if data_splits is None:
        print("Failed to load data")
        return None
    
    train_sentences, train_labels = data_splits['train']
    val_sentences, val_labels = data_splits['val']
    test_sentences, test_labels = data_splits['test']
    
    # Build vocabulary from training data
    print("\nBuilding vocabulary...")
    word2idx = build_vocabulary(train_sentences, config.VOCAB_SIZE)
    print(f"✓ Vocabulary size: {len(word2idx):,}")
    
    # Tag to index mapping
    tag2idx = config.TAG2IDX
    print(f"✓ Number of tags: {len(tag2idx)}")
    
    # Save vocabulary
    vocab_path = config.VOCAB_SAVE_PATH
    with open(vocab_path, 'wb') as f:
        pickle.dump({'word2idx': word2idx, 'tag2idx': tag2idx}, f)
    print(f"✓ Saved vocabulary to {vocab_path}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = NERDataset(train_sentences, train_labels, word2idx, tag2idx)
    val_dataset = NERDataset(val_sentences, val_labels, word2idx, tag2idx)
    test_dataset = NERDataset(test_sentences, test_labels, word2idx, tag2idx)
    
    print(f"✓ Train dataset: {len(train_dataset):,} samples")
    print(f"✓ Val dataset: {len(val_dataset):,} samples")
    print(f"✓ Test dataset: {len(test_dataset):,} samples")
    
    # Show sample
    print("\n" + "=" * 80)
    print("SAMPLE DATA")
    print("=" * 80)
    
    for i in range(3):
        sentence = train_sentences[i]
        labels = train_labels[i]
        print(f"\nSample {i+1}:")
        print(f"  Tokens: {' '.join(sentence[:15])}...")
        print(f"  Labels: {' '.join(labels[:15])}...")
    
    print("\n" + "=" * 80)
    print("✓ DATA PREPARATION COMPLETE")
    print("=" * 80)
    
    return {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset,
        'word2idx': word2idx,
        'tag2idx': tag2idx
    }


def create_dataloaders(datasets: Dict, batch_size: int = config.BATCH_SIZE) -> Dict:
    """
    Create DataLoaders
    
    Args:
        datasets: Dictionary of datasets
        batch_size: Batch size
        
    Returns:
        Dictionary of DataLoaders
    """
    dataloaders = {
        'train': DataLoader(
            datasets['train'],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config.NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False
        ),
        'val': DataLoader(
            datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config.NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False
        ),
        'test': DataLoader(
            datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config.NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False
        )
    }
    
    return dataloaders


if __name__ == "__main__":
    # Prepare data
    result = prepare_data()
    
    if result:
        # Test DataLoader
        print("\n" + "=" * 80)
        print("TESTING DATALOADER")
        print("=" * 80)
        
        dataloaders = create_dataloaders(result)
        
        # Get one batch
        word_ids, label_ids, lengths = next(iter(dataloaders['train']))
        
        print(f"\nBatch shapes:")
        print(f"  Word IDs: {word_ids.shape}")
        print(f"  Label IDs: {label_ids.shape}")
        print(f"  Lengths: {lengths.shape}")
        print(f"  Sample lengths: {lengths[:5]}")
        
        print("\n✓ DataLoader test complete!")