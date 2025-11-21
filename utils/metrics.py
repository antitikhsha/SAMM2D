"""
Evaluation metrics for aneurysm detection.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve
)


def compute_metrics(labels, predictions, threshold=0.5):
    """
    Compute all evaluation metrics.
    
    Args:
        labels: Ground truth labels (numpy array)
        predictions: Predicted probabilities (numpy array)
        threshold: Classification threshold (default: 0.5)
    
    Returns:
        Dictionary of metrics
    """
    # Binary predictions
    pred_binary = (predictions >= threshold).astype(int)
    
    # Compute metrics
    metrics = {
        'auc': roc_auc_score(labels, predictions),
        'accuracy': accuracy_score(labels, pred_binary),
        'precision': precision_score(labels, pred_binary, zero_division=0),
        'recall': recall_score(labels, pred_binary, zero_division=0),
        'f1': f1_score(labels, pred_binary, zero_division=0),
    }
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, pred_binary).ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return metrics


def find_optimal_threshold(labels, predictions):
    """
    Find optimal classification threshold using Youden's J statistic.
    
    Args:
        labels: Ground truth labels
        predictions: Predicted probabilities
    
    Returns:
        Optimal threshold
    """
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold


def compute_confusion_matrix_metrics(labels, predictions, threshold=0.5):
    """
    Compute detailed confusion matrix metrics.
    
    Returns:
        Dictionary with TP, TN, FP, FN, TPR, FPR, TNR, FNR
    """
    pred_binary = (predictions >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred_binary).ravel()
    
    total_pos = tp + fn
    total_neg = tn + fp
    
    metrics = {
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'TPR': tp / total_pos if total_pos > 0 else 0.0,  # True Positive Rate (Sensitivity)
        'FPR': fp / total_neg if total_neg > 0 else 0.0,  # False Positive Rate
        'TNR': tn / total_neg if total_neg > 0 else 0.0,  # True Negative Rate (Specificity)
        'FNR': fn / total_pos if total_pos > 0 else 0.0,  # False Negative Rate
    }
    
    return metrics


def bootstrap_confidence_interval(labels, predictions, metric_fn, n_bootstraps=1000, confidence=0.95):
    """
    Compute bootstrap confidence interval for a metric.
    
    Args:
        labels: Ground truth labels
        predictions: Predicted probabilities
        metric_fn: Function to compute metric
        n_bootstraps: Number of bootstrap samples
        confidence: Confidence level
    
    Returns:
        (mean, lower_bound, upper_bound)
    """
    n_samples = len(labels)
    scores = []
    
    for _ in range(n_bootstraps):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        boot_labels = labels[indices]
        boot_preds = predictions[indices]
        
        # Compute metric
        try:
            score = metric_fn(boot_labels, boot_preds)
            scores.append(score)
        except:
            continue
    
    scores = np.array(scores)
    alpha = 1 - confidence
    lower = np.percentile(scores, alpha/2 * 100)
    upper = np.percentile(scores, (1 - alpha/2) * 100)
    mean = np.mean(scores)
    
    return mean, lower, upper


if __name__ == "__main__":
    # Test metrics
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    predictions = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6, 0.85, 0.15])
    
    metrics = compute_metrics(labels, predictions)
    print("Metrics:", metrics)
    
    optimal_threshold = find_optimal_threshold(labels, predictions)
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    
    cm_metrics = compute_confusion_matrix_metrics(labels, predictions)
    print("Confusion matrix metrics:", cm_metrics)
