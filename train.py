"""
Training script for Transformer NER model
Complete training pipeline with validation and checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import time
import json
from pathlib import Path

import config
from models.ner_model import TransformerNER
from utils.data_loader import prepare_data, create_dataloaders
from utils.metrics import compute_metrics, print_metrics_summary, compute_entity_metrics


class Trainer:
    """
    Trainer class for NER model
    """
    
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 word2idx, tag2idx, device):
        """
        Initialize trainer
        
        Args:
            model: TransformerNER model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            test_loader: Test DataLoader
            word2idx: Word to index mapping
            tag2idx: Tag to index mapping
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.device = device
        
        # Loss function (ignore padding)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=config.PAD_IDX,
            label_smoothing=config.LABEL_SMOOTHING
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=config.BETAS,
            eps=config.EPS
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Maximize F1 score
            factor=0.5,
            patience=2
        )
        
        # TensorBoard writer
        self.writer = SummaryWriter(config.LOGS_DIR / 'transformer_ner')
        
        # Training state
        self.best_f1 = 0.0
        self.patience_counter = 0
        self.global_step = 0
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch}/{config.NUM_EPOCHS}',
            leave=True
        )
        
        for batch_idx, (word_ids, label_ids, lengths) in enumerate(progress_bar):
            # Move to device
            word_ids = word_ids.to(self.device)
            label_ids = label_ids.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(word_ids)
            
            # Reshape for loss calculation
            # logits: (batch_size, seq_len, num_labels)
            # labels: (batch_size, seq_len)
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),  # (batch_size * seq_len, num_labels)
                label_ids.view(-1)  # (batch_size * seq_len)
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                config.MAX_GRAD_NORM
            )
            
            # Update weights
            self.optimizer.step()
            
            # Track loss
            batch_loss = loss.item()
            total_loss += batch_loss
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
            
            # Log to TensorBoard
            if (batch_idx + 1) % config.LOG_INTERVAL == 0:
                self.writer.add_scalar('Train/Batch_Loss', batch_loss, self.global_step)
                self.writer.add_scalar('Train/Learning_Rate', 
                                     self.optimizer.param_groups[0]['lr'], 
                                     self.global_step)
            
            self.global_step += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def evaluate(self, dataloader, split_name='val'):
        """
        Evaluate model
        
        Args:
            dataloader: DataLoader to evaluate on
            split_name: Name of split (for logging)
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_lengths = []
        
        with torch.no_grad():
            for word_ids, label_ids, lengths in tqdm(dataloader, desc=f'Evaluating {split_name}', leave=False):
                word_ids = word_ids.to(self.device)
                label_ids = label_ids.to(self.device)
                
                # Forward pass
                logits = self.model(word_ids)
                
                # Calculate loss
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    label_ids.view(-1)
                )
                total_loss += loss.item()
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                
                # Store predictions and labels
                all_predictions.extend(predictions.cpu().tolist())
                all_labels.extend(label_ids.cpu().tolist())
                all_lengths.extend(lengths.tolist())
        
        # Compute metrics
        avg_loss = total_loss / len(dataloader)
        metrics = compute_metrics(
            all_predictions,
            all_labels,
            all_lengths,
            verbose=False
        )
        
        return avg_loss, metrics
    
    def train(self):
        """
        Main training loop
        """
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        
        for epoch in range(1, config.NUM_EPOCHS + 1):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_metrics = self.evaluate(self.val_loader, 'val')
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['f1'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Metrics/F1', val_metrics['f1'], epoch)
            self.writer.add_scalar('Metrics/Precision', val_metrics['precision'], epoch)
            self.writer.add_scalar('Metrics/Recall', val_metrics['recall'], epoch)
            
            # Calculate epoch time
            epoch_time = time.time() - start_time
            
            # Print epoch summary
            print(f'\n{"=" * 80}')
            print(f'Epoch {epoch}/{config.NUM_EPOCHS} Summary')
            print(f'{"=" * 80}')
            print(f'  Train Loss:  {train_loss:.4f}')
            print(f'  Val Loss:    {val_loss:.4f}')
            print(f'  F1 Score:    {val_metrics["f1"]:.4f}')
            print(f'  Precision:   {val_metrics["precision"]:.4f}')
            print(f'  Recall:      {val_metrics["recall"]:.4f}')
            print(f'  Accuracy:    {val_metrics["accuracy"]:.4f}')
            print(f'  Learning Rate: {current_lr:.6f}')
            print(f'  Time:        {epoch_time:.2f}s')
            
            # Check if best model
            is_best = val_metrics['f1'] > self.best_f1
            
            if is_best:
                self.best_f1 = val_metrics['f1']
                self.patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_f1': self.best_f1,
                    'config': self.model.get_config(),
                }, config.MODEL_SAVE_PATH)
                
                print(f'  ✓ New best model! (F1: {self.best_f1:.4f})')
            else:
                self.patience_counter += 1
                print(f'  No improvement for {self.patience_counter} epoch(s)')
            
            # Early stopping
            if self.patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f'\n{"=" * 80}')
                print(f'Early stopping triggered after {epoch} epochs')
                print(f'Best F1 score: {self.best_f1:.4f}')
                print(f'{"=" * 80}')
                break
            
            print(f'{"=" * 80}\n')
        
        # Training complete
        self.writer.close()
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE!")
        print("=" * 80)
        print(f'✓ Best F1 score: {self.best_f1:.4f}')
        print(f'✓ Model saved to: {config.MODEL_SAVE_PATH}')
        print("=" * 80)
    
    def test(self):
        """
        Evaluate on test set
        """
        print("\n" + "=" * 80)
        print("EVALUATING ON TEST SET")
        print("=" * 80)
        
        # Load best model - FIXED: Add weights_only=False for PyTorch 2.6+
        checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate
        test_loss, test_metrics = self.evaluate(self.test_loader, 'test')
        
        # Compute per-entity metrics
        print("\nComputing detailed metrics...")
        
        all_predictions = []
        all_labels = []
        all_lengths = []
        
        self.model.eval()
        with torch.no_grad():
            for word_ids, label_ids, lengths in self.test_loader:
                word_ids = word_ids.to(self.device)
                logits = self.model(word_ids)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().tolist())
                all_labels.extend(label_ids.tolist())
                all_lengths.extend(lengths.tolist())
        
        entity_metrics = compute_entity_metrics(all_predictions, all_labels, all_lengths)
        
        # Print results
        print_metrics_summary(test_metrics, entity_metrics)
        
        # Save results
        results = {
            'test_loss': test_loss,
            'metrics': test_metrics,
            'entity_metrics': entity_metrics
        }
        
        results_path = config.MODEL_DIR / 'test_results.json'
        with open(results_path, 'w') as f:
            # Remove report string for JSON
            results_copy = results.copy()
            if 'report' in results_copy['metrics']:
                del results_copy['metrics']['report']
            json.dump(results_copy, f, indent=2)
        
        print(f"\n✓ Results saved to {results_path}")
        
        return test_metrics, entity_metrics


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train Transformer NER model')
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--test_only', action='store_true',
                       help='Only run testing')
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.batch_size != config.BATCH_SIZE:
        config.BATCH_SIZE = args.batch_size
    if args.lr != config.LEARNING_RATE:
        config.LEARNING_RATE = args.lr
    if args.epochs != config.NUM_EPOCHS:
        config.NUM_EPOCHS = args.epochs
    
    # Print configuration
    print("\n" + "=" * 80)
    print("TRANSFORMER NER TRAINING")
    print("=" * 80)
    config.print_config()
    
    # Prepare data
    print("\n" + "=" * 80)
    print("PREPARING DATA")
    print("=" * 80)
    
    data = prepare_data()
    
    if data is None:
        print("Failed to prepare data!")
        return
    
    dataloaders = create_dataloaders(data, config.BATCH_SIZE)
    
    # Create model
    print("\n" + "=" * 80)
    print("CREATING MODEL")
    print("=" * 80)
    
    model = TransformerNER(vocab_size=len(data['word2idx']))
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Create trainer
    trainer = Trainer(
        model,
        dataloaders['train'],
        dataloaders['val'],
        dataloaders['test'],
        data['word2idx'],
        data['tag2idx'],
        config.DEVICE
    )
    
    # Train or test
    if not args.test_only:
        trainer.train()
    
    # Test
    trainer.test()


if __name__ == "__main__":
    main()