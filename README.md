# Mini-Transformer for Named Entity Recognition (NER)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**From-scratch implementation** of the Transformer architecture for Named Entity Recognition (NER), following the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762). This project demonstrates Transformers understanding by implementing every component without using pre-built modules.

---

## Project Overview

This project uses the **CoNLL-2003 dataset**. Every component—from Multi-Head Attention to Positional Encoding—is built manually.

### **Key Features**

**Complete Transformer Implementation**
- Multi-Head Self-Attention mechanism
- Positional Encoding with sinusoidal embeddings
- Position-wise Feed-Forward Networks
- Layer Normalization and Residual Connections
- Transformer Encoder Layers

**Educational Focus**
- Heavily commented code explaining each component
- Modular architecture for easy understanding
- Step-by-step implementation following the paper

**Production Ready**
- Comprehensive evaluation metrics (F1, Precision, Recall)
- Attention weight visualization
- Interactive inference script

**Real Dataset**
- CoNLL-2003 Named Entity Recognition dataset
- 4 entity types: PER (Person), ORG (Organization), LOC (Location), MISC (Miscellaneous)
- ~14,000 training sentences, ~3,000 validation, ~3,000 test

---

## Architecture

### **Transformer Encoder Stack**

```
Input Tokens
    ↓
Token Embeddings (d_model=128)
    ↓
Positional Encoding (sinusoidal)
    ↓
┌─────────────────────────────────┐
│  Transformer Encoder Layer 1    │
│  ├─ Multi-Head Attention (4)    │
│  ├─ Add & Norm                   │
│  ├─ Feed-Forward (FFN)           │
│  └─ Add & Norm                   │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Transformer Encoder Layer 2    │
│  ├─ Multi-Head Attention (4)    │
│  ├─ Add & Norm                   │
│  ├─ Feed-Forward (FFN)           │
│  └─ Add & Norm                   │
└─────────────────────────────────┘
    ↓
Linear Classifier (→ 9 NER tags)
    ↓
Output: B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC, O
```

### **Core Components**

#### 1. **Multi-Head Attention**
```python
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```
- Scaled dot-product attention
- Multiple attention heads for different representation subspaces
- Allows model to attend to different positions simultaneously

#### 2. **Positional Encoding**
```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
- Injects sequence order information
- No learnable parameters
- Allows model to utilize order of tokens

#### 3. **Position-wise Feed-Forward Network**
```python
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```
- Two linear transformations with ReLU activation
- Applied identically to each position
- Hidden dimension expansion (512 in this implementation)

### **Model Specifications (CPU-Optimized)**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_model` | 128 | Model dimension |
| `num_layers` | 2 | Number of encoder layers |
| `num_heads` | 4 | Number of attention heads |
| `d_ff` | 512 | Feed-forward hidden dimension |
| `dropout` | 0.1 | Dropout rate |
| `max_seq_length` | 128 | Maximum sequence length |
| `vocab_size` | 10,000 | Vocabulary size |

**Total Parameters:** ~1.5M (optimized for CPU training)

---

## 📁 Project Structure

```
mini-transformer-ner/
├── config.py                      # Configuration parameters
├── models/
│   ├── __init__.py               # Package initialization
│   ├── attention.py              # Multi-Head Attention implementation
│   ├── positional_encoding.py   # Positional encoding
│   ├── feed_forward.py           # Position-wise FFN
│   ├── transformer_encoder.py   # Transformer encoder layers
│   └── ner_model.py              # Complete NER model
├── utils/
│   ├── __init__.py               # Package initialization
│   ├── data_loader.py            # CoNLL-2003 dataset loading
│   ├── metrics.py                # Evaluation metrics (F1, Precision, Recall)
│   └── visualization.py          # Attention visualization
├── train.py                       # Training script
├── inference.py                   # Interactive inference
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🚀 Getting Started

### **Prerequisites**

- Python 3.8 or higher
- PyTorch 2.0+
- 4GB+ RAM (for CPU training)
- ~2GB disk space

### **Installation**

1. **Clone the repository**
```bash
git clone https://github.com/RamuNalla/mini-transformer-from-scratch.git
cd mini-transformer-from-scratch
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Prepare the dataset**
```bash
python utils/data_loader.py
```

Expected output:
```
================================================================================
LOADING CoNLL-2003 FROM HUGGING FACE
================================================================================
Loading 'eriktks/conll2003' mirror...
✓ Successfully loaded eriktks/conll2003
Processing 14041 examples...
Processing 3250 examples...
Processing 3453 examples...
✓ Train: 14,041 sentences
✓ Validation: 3,250 sentences
✓ Test: 3,453 sentences
```

---

##  Usage

### **Training**

Train the model with default parameters:
```bash
python train.py --epochs 30 --batch_size 16
```

**Training Options:**
```bash
python train.py \
  --model transformer \
  --epochs 30 \
  --batch_size 16 \
  --lr 0.0001 \
  --resume             # Resume from checkpoint
  --test_only          # Only run testing
```

**Expected Training Output:**
```
================================================================================
STARTING TRAINING
================================================================================
Epoch 1/30: 100%|██████| 878/878 [02:45<00:00, 5.32it/s, loss=1.456]

================================================================================
Epoch 1/30 Summary
================================================================================
  Train Loss:  1.4562
  Val Loss:    1.2345
  F1 Score:    0.6234
  Precision:   0.5987
  Recall:      0.6512
  Accuracy:    0.8234
  Learning Rate: 0.000100
  Time:        165.23s
  ✓ New best model! (F1: 0.6234)
================================================================================
```

### **Testing**

Evaluate on test set:
```bash
python train.py --test_only
```

### **Inference**

**Interactive mode:**
```bash
python inference.py --interactive
```

**Single prediction:**
```bash
python inference.py --text "Apple Inc. was founded by Steve Jobs in California."
```

**Expected Output:**
```
================================================================================
PREDICTIONS
================================================================================
Input: Apple Inc. was founded by Steve Jobs in California.

Entities found:
  - Apple Inc.          [ORG] (confidence: 0.956)
  - Steve Jobs          [PER] (confidence: 0.982)
  - California          [LOC] (confidence: 0.945)
```

---

## Results

### **Performance Metrics**

Training on the CoNLL-2003 dataset with the CPU-optimized configuration:

| Metric | Score |
|--------|-------|
| **F1 Score** | 0.70-0.85 |
| **Precision** | 0.68-0.83 |
| **Recall** | 0.72-0.87 |
| **Accuracy** | 0.90-0.94 |

*Note: Results vary based on training epochs and random seed. Full-size model (d_model=256, 4 layers) achieves higher scores.*

### **Training Time (CPU)**

| Configuration | Time per Epoch | Total (30 epochs) |
|--------------|----------------|-------------------|
| CPU-Optimized (128d, 2 layers) | ~2-3 minutes | ~60-90 minutes |
| Full Model (256d, 4 layers) | ~5-7 minutes | ~150-210 minutes |

### **Entity Type Performance**

| Entity Type | F1 Score | Examples |
|-------------|----------|----------|
| **PER** (Person) | 0.85 | Steve Jobs, Albert Einstein |
| **ORG** (Organization) | 0.78 | Apple Inc., United Nations |
| **LOC** (Location) | 0.82 | California, New York |
| **MISC** (Miscellaneous) | 0.72 | English, Olympics |

---

## 🔍 Key Implementation Details

### **1. Scaled Dot-Product Attention**

```python
def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Attention(Q, K, V) = softmax(QK^T / √d_k)V
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights
```

### **2. Multi-Head Attention**

```python
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention allows the model to jointly attend to information
    from different representation subspaces at different positions.
    """
    def __init__(self, d_model, num_heads, dropout_p=0.1):
        # Split d_model across num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
```

### **3. Positional Encoding**

```python
class PositionalEncoding(nn.Module):
    """
    Inject sequence position information using sine and cosine functions
    of different frequencies.
    """
    def __init__(self, d_model, max_len=5000):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
```

### **4. Transformer Encoder Layer**

```python
class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer encoder layer:
    - Multi-head self-attention
    - Position-wise feed-forward network
    - Residual connections and layer normalization
    """
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x
```

---


## 🔧 Configuration

Key hyperparameters in `config.py`:

```python
# Model Architecture
D_MODEL = 128              # Embedding dimension
NUM_LAYERS = 2            # Number of encoder layers
NUM_HEADS = 4             # Number of attention heads
D_FF = 512                # Feed-forward hidden dimension
DROPOUT = 0.1             # Dropout rate
MAX_SEQ_LENGTH = 128      # Maximum sequence length

# Training
BATCH_SIZE = 16           # Batch size
NUM_EPOCHS = 30           # Training epochs
LEARNING_RATE = 1e-4      # Learning rate
WARMUP_STEPS = 4000       # Learning rate warmup
MAX_GRAD_NORM = 1.0       # Gradient clipping

# Early Stopping
EARLY_STOPPING_PATIENCE = 10  # Epochs before stopping
```
---

## Training Tips

### **For Better Performance:**

1. **Increase training epochs**: 30-50 epochs for convergence
2. **Use learning rate warmup**: Already implemented
3. **Gradient clipping**: Prevents exploding gradients
4. **Early stopping**: Prevents overfitting (patience=10)
5. **Label smoothing**: Can improve generalization

### **For Faster Training:**

1. **Reduce sequence length**: `MAX_SEQ_LENGTH = 64`
2. **Smaller model**: `D_MODEL = 64, NUM_LAYERS = 1`
3. **Larger batch size**: `BATCH_SIZE = 32` (if memory allows)
4. **Fewer epochs**: For quick testing

---

## 🧪 Testing Components

Test individual components:

```bash
# Test Multi-Head Attention
python models/attention.py

# Test Positional Encoding
python models/positional_encoding.py

# Test Feed-Forward Network
python models/feed_forward.py

# Test Transformer Encoder
python models/transformer_encoder.py

# Test Complete Model
python models/ner_model.py
```

---

## Learning Resources

This implementation closely follows these resources:

- **Paper**: [Attention is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- **Dataset**: [CoNLL-2003 NER](https://www.clips.uantwerpen.be/conll2003/ner/)

---


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 📧 Contact

For questions or suggestions, please open an issue on GitHub or contact:

- **Author**: Ramu Nalla
- **Email**: nallaramu4321@gmail.com

---

