"""
Training script for SAMM2D model.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.samm2d import SAMM2D, count_parameters
from data.dataset import create_dataloaders
from utils.metrics import compute_metrics
from utils.losses import FocalLoss


class Trainer:
    """Trainer class for SAMM2D model."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'logs'))
        
        # Build model
        self.model = SAMM2D(
            img_size=config['img_size'],
            patch_size=config['patch_size'],
            in_channels=config['in_channels'],
            num_classes=config['num_classes'],
            embed_dim=config['embed_dim'],
            depth=config['depth'],
            num_heads=config['num_heads'],
            mlp_ratio=config['mlp_ratio'],
            dropout=config['dropout']
        ).to(self.device)
        
        print(f"Model parameters: {count_parameters(self.model):,}")
        
        # Loss function
        if config['loss'] == 'focal':
            self.criterion = FocalLoss(alpha=config['focal_alpha'], gamma=config['focal_gamma'])
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config['scheduler_t0'],
            T_mult=config['scheduler_tmult']
        )
        
        # Load data
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            data_dir=config['data_dir'],
            labels_csv=config['labels_csv'],
            batch_size=config['batch_size'],
            img_size=config['img_size'],
            num_workers=config['num_workers'],
            train_augment=config['train_augment']
        )
        
        # Best metrics
        self.best_val_auc = 0.0
        self.best_epoch = 0
        
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            tof = batch['tof'].to(self.device)
            mra = batch['mra'].to(self.device)
            scale_orig = batch['scale_orig'].to(self.device)
            scale_down = batch['scale_down'].to(self.device)
            scale_up = batch['scale_up'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            logits = self.model(tof, mra, scale_orig, scale_down, scale_up)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['grad_clip']
            )
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
        
        # Log to tensorboard
        self.writer.add_scalar('Train/Loss', avg_loss, epoch)
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(f'Train/{metric_name}', metric_value, epoch)
        
        return avg_loss, metrics
    
    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                tof = batch['tof'].to(self.device)
                mra = batch['mra'].to(self.device)
                scale_orig = batch['scale_orig'].to(self.device)
                scale_down = batch['scale_down'].to(self.device)
                scale_up = batch['scale_up'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                logits = self.model(tof, mra, scale_orig, scale_down, scale_up)
                loss = self.criterion(logits, labels)
                
                # Track metrics
                total_loss += loss.item()
                preds = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
        
        # Log to tensorboard
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(f'Val/{metric_name}', metric_value, epoch)
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model at epoch {epoch}")
    
    def train(self):
        """Main training loop."""
        print(f"Starting training on {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.config['num_epochs']}")
            print(f"{'='*60}")
            
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)
            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Train Metrics: {train_metrics}")
            
            # Validate
            val_loss, val_metrics = self.validate(epoch)
            print(f"\nVal Loss: {val_loss:.4f}")
            print(f"Val Metrics: {val_metrics}")
            
            # Update scheduler
            self.scheduler.step()
            
            # Check if best model
            is_best = val_metrics['auc'] > self.best_val_auc
            if is_best:
                self.best_val_auc = val_metrics['auc']
                self.best_epoch = epoch
            
            # Save checkpoint
            if epoch % self.config['save_freq'] == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best=is_best)
            
            # Early stopping
            if epoch - self.best_epoch >= self.config['patience']:
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"Best epoch: {self.best_epoch} with AUC: {self.best_val_auc:.4f}")
                break
        
        print(f"\nTraining completed!")
        print(f"Best epoch: {self.best_epoch} with AUC: {self.best_val_auc:.4f}")
        
        # Close tensorboard writer
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train SAMM2D model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Create trainer and train
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
