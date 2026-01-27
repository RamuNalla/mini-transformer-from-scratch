
"""
Attention visualization utilities for Transformer NER
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
import config

def plot_attention_heatmap(
    attention_weights: np.ndarray,
    tokens: List[str],
    layer_idx: int = 0,
    head_idx: int = 0,
    save_path: Optional[Path] = None,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot attention weights as a heatmap
    
    Args:
        attention_weights: Attention weights (seq_len, seq_len) or (num_heads, seq_len, seq_len)
        tokens: List of tokens
        layer_idx: Layer index (for title)
        head_idx: Attention head index
        save_path: Path to save the figure
        title: Custom title
        
    Returns:
        Matplotlib figure
    """
    # If 3D, extract the specified head
    if len(attention_weights.shape) == 3:
        attention_weights = attention_weights[head_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(
        attention_weights,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='YlOrRd',
        annot=len(tokens) < 20,  # Show values only for short sequences
        fmt='.2f',
        cbar_kws={'label': 'Attention Weight'},
        square=True,
        linewidths=0.5,
        ax=ax,
        vmin=0,
        vmax=1
    )
    
    # Set title
    if title is None:
        title = f'Attention Weights - Layer {layer_idx}, Head {head_idx}'
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Key Tokens', fontsize=12)
    ax.set_ylabel('Query Tokens', fontsize=12)
    
    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved attention heatmap to {save_path}")
    
    return fig

def plot_multi_head_attention(
    attention_weights: np.ndarray,
    tokens: List[str],
    layer_idx: int = 0,
    num_heads_to_show: int = 4,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot multiple attention heads in a grid
    
    Args:
        attention_weights: Attention weights (num_heads, seq_len, seq_len)
        tokens: List of tokens
        layer_idx: Layer index
        num_heads_to_show: Number of attention heads to display
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    num_heads = min(num_heads_to_show, attention_weights.shape[0])
    
    # Create grid
    rows = int(np.ceil(np.sqrt(num_heads)))
    cols = int(np.ceil(num_heads / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)
    
    for idx in range(num_heads):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Plot heatmap for this head
        sns.heatmap(
            attention_weights[idx],
            xticklabels=tokens if len(tokens) < 15 else False,
            yticklabels=tokens if len(tokens) < 15 else False,
            cmap='YlOrRd',
            cbar=True,
            square=True,
            ax=ax,
            vmin=0,
            vmax=1
        )
        
        ax.set_title(f'Head {idx}', fontsize=10, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(num_heads, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    fig.suptitle(f'Multi-Head Attention - Layer {layer_idx}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved multi-head attention plot to {save_path}")
    
    return fig


def plot_attention_patterns(
    attention_weights_list: List[np.ndarray],
    tokens: List[str],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot attention patterns across all layers
    Shows average attention for each layer
    
    Args:
        attention_weights_list: List of attention weights for each layer
        tokens: List of tokens
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    num_layers = len(attention_weights_list)
    
    fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 5))
    
    if num_layers == 1:
        axes = [axes]
    
    for layer_idx, attention_weights in enumerate(attention_weights_list):
        # Average across all heads
        if len(attention_weights.shape) == 3:
            attention_avg = attention_weights.mean(axis=0)
        else:
            attention_avg = attention_weights
        
        # Plot
        sns.heatmap(
            attention_avg,
            xticklabels=tokens if len(tokens) < 15 else False,
            yticklabels=tokens if len(tokens) < 15 else False,
            cmap='YlOrRd',
            cbar=True,
            square=True,
            ax=axes[layer_idx],
            vmin=0,
            vmax=1
        )
        
        axes[layer_idx].set_title(f'Layer {layer_idx}', fontsize=12, fontweight='bold')
    
    fig.suptitle('Attention Patterns Across Layers (Average)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved attention patterns to {save_path}")
    
    return fig


def visualize_entity_attention(
    attention_weights: np.ndarray,
    tokens: List[str],
    entities: List[tuple],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Visualize attention with entity highlighting
    
    Args:
        attention_weights: Attention weights (seq_len, seq_len)
        tokens: List of tokens
        entities: List of (start_idx, end_idx, entity_type) tuples
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(
        attention_weights,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='YlOrRd',
        cbar_kws={'label': 'Attention Weight'},
        square=True,
        ax=ax,
        vmin=0,
        vmax=1
    )
    
    # Highlight entity regions
    colors = {
        'PER': 'blue',
        'ORG': 'green',
        'LOC': 'orange',
        'MISC': 'purple'
    }
    
    for start_idx, end_idx, entity_type in entities:
        color = colors.get(entity_type, 'red')
        
        # Draw rectangles around entity regions
        ax.add_patch(plt.Rectangle(
            (start_idx, start_idx), 
            end_idx - start_idx, 
            end_idx - start_idx,
            fill=False, 
            edgecolor=color, 
            linewidth=3
        ))
    
    ax.set_title('Attention Weights with Entity Highlighting', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Key Tokens', fontsize=12)
    ax.set_ylabel('Query Tokens', fontsize=12)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved entity attention visualization to {save_path}")
    
    return fig


# Test visualization
if __name__ == "__main__":
    print("=" * 80)
    print("TESTING ATTENTION VISUALIZATION")
    print("=" * 80)
    
    # Create dummy data
    seq_len = 8
    num_heads = 4
    num_layers = 2
    
    tokens = ['John', 'works', 'at', 'Google', 'in', 'Paris', '.', '<PAD>']
    
    # Create random attention weights
    np.random.seed(42)
    
    # Single head attention
    print("\nTest 1: Single head attention")
    attention = np.random.rand(seq_len, seq_len)
    attention = attention / attention.sum(axis=1, keepdims=True)
    
    fig1 = plot_attention_heatmap(
        attention,
        tokens,
        layer_idx=0,
        head_idx=0,
        save_path=config.VISUALIZATION_DIR / "test_single_head.png"
    )
    plt.close(fig1)
    print("✓ Single head visualization test passed")
    
    # Multi-head attention
    print("\nTest 2: Multi-head attention")
    multi_head_attention = np.random.rand(num_heads, seq_len, seq_len)
    for h in range(num_heads):
        multi_head_attention[h] = multi_head_attention[h] / multi_head_attention[h].sum(axis=1, keepdims=True)
    
    fig2 = plot_multi_head_attention(
        multi_head_attention,
        tokens,
        layer_idx=0,
        num_heads_to_show=4,
        save_path=config.VISUALIZATION_DIR / "test_multi_head.png"
    )
    plt.close(fig2)
    print("✓ Multi-head visualization test passed")
    
    # Attention across layers
    print("\nTest 3: Attention across layers")
    attention_list = [
        np.random.rand(num_heads, seq_len, seq_len) for _ in range(num_layers)
    ]
    
    for attn in attention_list:
        for h in range(num_heads):
            attn[h] = attn[h] / attn[h].sum(axis=1, keepdims=True)
    
    fig3 = plot_attention_patterns(
        attention_list,
        tokens,
        save_path=config.VISUALIZATION_DIR / "test_patterns.png"
    )
    plt.close(fig3)
    print("✓ Attention patterns test passed")
    
    # Entity attention
    print("\nTest 4: Entity attention")
    entities = [
        (0, 1, 'PER'),   # John
        (3, 4, 'ORG'),   # Google
        (5, 6, 'LOC'),   # Paris
    ]
    
    fig4 = visualize_entity_attention(
        attention,
        tokens,
        entities,
        save_path=config.VISUALIZATION_DIR / "test_entity_attention.png"
    )
    plt.close(fig4)
    print("✓ Entity attention test passed")
    
    print("\n" + "=" * 80)
    print("✓ ALL VISUALIZATION TESTS COMPLETE")
    print(f"✓ Saved visualizations to {config.VISUALIZATION_DIR}")
    print("=" * 80)

