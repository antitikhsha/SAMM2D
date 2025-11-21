"""
Evaluation script for SAMM2D model.
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.samm2d import SAMM2D
from data.dataset import create_dataloaders
from utils.metrics import compute_metrics, compute_confusion_matrix_metrics, find_optimal_threshold


class Evaluator:
    """Evaluator class for SAMM2D model."""
    
    def __init__(self, checkpoint_path, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
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
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        # Load data
        _, _, self.test_loader = create_dataloaders(
            data_dir=config['data_dir'],
            labels_csv=config['labels_csv'],
            batch_size=config['batch_size'],
            img_size=config['img_size'],
            num_workers=config['num_workers'],
            train_augment=False
        )
    
    def evaluate(self):
        """Run evaluation on test set."""
        all_preds = []
        all_labels = []
        all_filenames = []
        
        print(f"Evaluating on {len(self.test_loader.dataset)} test samples...")
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                # Move to device
                tof = batch['tof'].to(self.device)
                mra = batch['mra'].to(self.device)
                scale_orig = batch['scale_orig'].to(self.device)
                scale_down = batch['scale_down'].to(self.device)
                scale_up = batch['scale_up'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                logits = self.model(tof, mra, scale_orig, scale_down, scale_up)
                preds = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_filenames.extend(batch['filename'])
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        return all_labels, all_preds, all_filenames
    
    def plot_results(self, labels, predictions, output_dir):
        """Plot evaluation results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find optimal threshold
        optimal_threshold = find_optimal_threshold(labels, predictions)
        
        # ROC Curve
        fpr, tpr, thresholds = roc_curve(labels, predictions)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - SAMM2D', fontsize=14, weight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Confusion Matrix
        pred_binary = (predictions >= optimal_threshold).astype(int)
        cm = confusion_matrix(labels, pred_binary)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.title(f'Confusion Matrix (threshold={optimal_threshold:.3f})', 
                 fontsize=14, weight='bold')
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Prediction Distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Distribution by class
        axes[0].hist(predictions[labels == 0], bins=50, alpha=0.7, 
                    label='Negative', color='blue')
        axes[0].hist(predictions[labels == 1], bins=50, alpha=0.7, 
                    label='Positive', color='red')
        axes[0].axvline(optimal_threshold, color='green', linestyle='--', 
                       linewidth=2, label=f'Threshold={optimal_threshold:.3f}')
        axes[0].set_xlabel('Predicted Probability', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Prediction Distribution by Class', fontsize=12, weight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Threshold vs Metrics
        metrics_at_thresholds = []
        thresholds_to_test = np.linspace(0, 1, 100)
        for t in thresholds_to_test:
            metrics = compute_metrics(labels, predictions, threshold=t)
            metrics_at_thresholds.append(metrics)
        
        axes[1].plot(thresholds_to_test, 
                    [m['precision'] for m in metrics_at_thresholds],
                    label='Precision', linewidth=2)
        axes[1].plot(thresholds_to_test, 
                    [m['recall'] for m in metrics_at_thresholds],
                    label='Recall', linewidth=2)
        axes[1].plot(thresholds_to_test, 
                    [m['f1'] for m in metrics_at_thresholds],
                    label='F1-Score', linewidth=2)
        axes[1].axvline(optimal_threshold, color='green', linestyle='--', 
                       linewidth=2, label=f'Optimal={optimal_threshold:.3f}')
        axes[1].set_xlabel('Threshold', fontsize=12)
        axes[1].set_ylabel('Score', fontsize=12)
        axes[1].set_title('Metrics vs Threshold', fontsize=12, weight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Plots saved to {output_dir}")
    
    def save_results(self, labels, predictions, filenames, output_dir):
        """Save evaluation results."""
        output_dir = Path(output_dir)
        
        # Compute metrics
        optimal_threshold = find_optimal_threshold(labels, predictions)
        metrics = compute_metrics(labels, predictions, threshold=optimal_threshold)
        cm_metrics = compute_confusion_matrix_metrics(labels, predictions, threshold=optimal_threshold)
        
        # Combine all metrics
        all_metrics = {**metrics, **cm_metrics, 'optimal_threshold': float(optimal_threshold)}
        
        # Save metrics
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(all_metrics, f, indent=4)
        
        # Save predictions
        results = []
        for filename, label, pred in zip(filenames, labels, predictions):
            results.append({
                'filename': filename,
                'true_label': int(label),
                'predicted_prob': float(pred),
                'predicted_label': int(pred >= optimal_threshold)
            })
        
        with open(output_dir / 'predictions.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"AUC-ROC:        {metrics['auc']:.4f}")
        print(f"Accuracy:       {metrics['accuracy']:.4f}")
        print(f"Precision:      {metrics['precision']:.4f}")
        print(f"Recall:         {metrics['recall']:.4f}")
        print(f"F1-Score:       {metrics['f1']:.4f}")
        print(f"Sensitivity:    {metrics['sensitivity']:.4f}")
        print(f"Specificity:    {metrics['specificity']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"TP: {cm_metrics['TP']}  FP: {cm_metrics['FP']}")
        print(f"FN: {cm_metrics['FN']}  TN: {cm_metrics['TN']}")
        print(f"{'='*60}")
        
        print(f"\n✓ Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate SAMM2D model')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                       help='Output directory for results')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create evaluator
    evaluator = Evaluator(args.checkpoint, config)
    
    # Run evaluation
    labels, predictions, filenames = evaluator.evaluate()
    
    # Save results
    evaluator.save_results(labels, predictions, filenames, args.output_dir)
    
    # Plot results
    evaluator.plot_results(labels, predictions, args.output_dir)


if __name__ == "__main__":
    main()
