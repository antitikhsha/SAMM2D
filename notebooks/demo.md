# SAMM2D Demo Notebook

This notebook demonstrates how to use SAMM2D for aneurysm detection.

## Setup

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
import sys
sys.path.append('..')

from models.samm2d import SAMM2D, count_parameters
from data.dataset import AneurysmDataset, create_dataloaders
from scripts.train import Trainer
from scripts.inference import Predictor
from utils.visualization import visualize_attention_maps
```

## 1. Model Architecture

```python
# Create model
model = SAMM2D(
    img_size=224,
    patch_size=16,
    in_channels=1,
    num_classes=2,
    embed_dim=256,
    depth=3,
    num_heads=4,
    mlp_ratio=4,
    dropout=0.1
)

print(f"Total parameters: {count_parameters(model):,}")
print(f"Model: {model}")
```

## 2. Dataset Loading

```python
# Create dataset
dataset = AneurysmDataset(
    data_dir='../data/processed',
    labels_csv='../data/labels.csv',
    img_size=224,
    augment=True,
    split='train'
)

print(f"Dataset size: {len(dataset)}")

# Get a sample
sample = dataset[0]
print(f"TOF shape: {sample['tof'].shape}")
print(f"Label: {sample['label']}")

# Visualize sample
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
axes[0].imshow(sample['tof'].squeeze(), cmap='gray')
axes[0].set_title('TOF')
axes[1].imshow(sample['mra'].squeeze(), cmap='gray')
axes[1].set_title('MRA')
axes[2].imshow(sample['scale_orig'].squeeze(), cmap='gray')
axes[2].set_title('Original')
axes[3].imshow(sample['scale_down'].squeeze(), cmap='gray')
axes[3].set_title('Downsampled')
axes[4].imshow(sample['scale_up'].squeeze(), cmap='gray')
axes[4].set_title('Upsampled')
plt.tight_layout()
plt.show()
```

## 3. Training

```python
import json

# Load config
with open('../configs/default_config.json', 'r') as f:
    config = json.load(f)

# Create trainer
trainer = Trainer(config)

# Train model
trainer.train()
```

## 4. Evaluation

```python
from scripts.evaluate import Evaluator

# Create evaluator
evaluator = Evaluator(
    checkpoint_path='../outputs/experiment_1/best_model.pth',
    config=config
)

# Run evaluation
labels, predictions, filenames = evaluator.evaluate()

# Plot results
evaluator.plot_results(labels, predictions, '../results/evaluation')
```

## 5. Inference

```python
# Create predictor
predictor = Predictor(
    checkpoint_path='../outputs/experiment_1/best_model.pth',
    config=config
)

# Run prediction
img_path = '../data/test/case_001.npy'
probability, prediction = predictor.predict(img_path)

print(f"Probability: {probability:.4f}")
print(f"Prediction: {'Aneurysm' if prediction == 1 else 'No Aneurysm'}")
```

## 6. Visualization

```python
# Visualize attention
from utils.visualization import visualize_attention_maps

# Load image
sample = dataset[0]

# Create input tensors
tof = sample['tof'].unsqueeze(0)
mra = sample['mra'].unsqueeze(0)
orig = sample['scale_orig'].unsqueeze(0)
down = sample['scale_down'].unsqueeze(0)
up = sample['scale_up'].unsqueeze(0)

input_tensors = (tof, mra, orig, down, up)

# Visualize
visualize_attention_maps(model, input_tensors, save_path='attention_viz.png')
```

## 7. Performance Analysis

```python
from utils.metrics import compute_metrics, compute_confusion_matrix_metrics

# Compute metrics
metrics = compute_metrics(labels, predictions)

print("Performance Metrics:")
print(f"AUC-ROC:     {metrics['auc']:.4f}")
print(f"Accuracy:    {metrics['accuracy']:.4f}")
print(f"Precision:   {metrics['precision']:.4f}")
print(f"Recall:      {metrics['recall']:.4f}")
print(f"F1-Score:    {metrics['f1']:.4f}")
print(f"Sensitivity: {metrics['sensitivity']:.4f}")
print(f"Specificity: {metrics['specificity']:.4f}")

# Confusion matrix
cm_metrics = compute_confusion_matrix_metrics(labels, predictions)
print(f"\nConfusion Matrix:")
print(f"TP: {cm_metrics['TP']}  FP: {cm_metrics['FP']}")
print(f"FN: {cm_metrics['FN']}  TN: {cm_metrics['TN']}")
```

## 8. Model Deployment

```python
# Export model for deployment
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config
}, 'samm2d_deployment.pth')

print("✓ Model exported for deployment")
```

## Conclusion

This notebook demonstrated:
1. Loading and using the SAMM2D model
2. Dataset preparation and loading
3. Training the model
4. Evaluating performance
5. Running inference
6. Visualizing attention mechanisms
7. Analyzing results

For more details, see the full documentation in the repository.
