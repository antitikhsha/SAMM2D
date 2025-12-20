# SAMM2D: Scale-Aware Multi-Modal 2D Transformer for Intracranial Aneurysm Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of **SAMM2D**

> **SAMM2D: Scale-Aware Multi-Modal 2D Transformer for Intracranial Aneurysm Detection**  
> Antara Titikhsha (CMU), Divyanshu Tak (BWH, Harvard Medical School)  


##  Overview

SAMM2D is a novel transformer-based architecture for detecting intracranial aneurysms in medical imaging. The model achieves **state-of-the-art performance** (AUC 0.78) while maintaining computational efficiency with only **18.8M parameters**.

### Key Features

- **Multi-Modal Fusion**: Integrates TOF and MRA imaging sequences through cross-modal attention
- **Multi-Scale Processing**: Captures features at multiple resolutions using cross-scale attention
- **Efficient Architecture**: 18.8M parameters, ~50ms inference time per case
- **Clinical Deployment Ready**: Designed for seamless PACS integration

##  Results

| Method | AUC-ROC | Parameters | Inference Time |
|--------|---------|------------|----------------|
| ResNet18 | 0.680 | 11.7M | - |
| 3D CNN | 0.710 | 8.5M | - |
| ViT | 0.720 | 86.5M | - |
| Swin Transformer | 0.740 | 28.3M | - |
| nnU-Net | 0.750 | 31.2M | - |
| **SAMM2D (Ours)** | **0.780** | **18.8M** | **~50ms** |

##  Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/samm2d.git
cd samm2d

# Create conda environment
conda create -n samm2d python=3.8
conda activate samm2d

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

1. Download the RSNA Intracranial Aneurysm Detection dataset
2. Organize data as follows:

```
data/
├── processed/
│   ├── case_001.npy
│   ├── case_002.npy
│   └── ...
└── labels.csv
```

3. `labels.csv` format:
```csv
filename,label,split
case_001.npy,0,train
case_002.npy,1,train
case_003.npy,0,val
...
```

### Training

```bash
python scripts/train.py --config configs/default_config.json
```

Monitor training with TensorBoard:
```bash
tensorboard --logdir outputs/experiment_1/logs
```

### Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint outputs/experiment_1/best_model.pth \
    --config configs/default_config.json \
    --output_dir results/evaluation
```

### Inference

```bash
python scripts/inference.py \
    --checkpoint outputs/experiment_1/best_model.pth \
    --input data/test_images/ \
    --output results/predictions/
```

##  Project Structure

```
samm2d/
├── configs/                 # Configuration files
│   └── default_config.json
├── data/                    # Dataset utilities
│   └── dataset.py
├── models/                  # Model architectures
│   └── samm2d.py
├── scripts/                 # Training and evaluation scripts
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── utils/                   # Utility functions
│   ├── metrics.py
│   ├── losses.py
│   └── visualization.py
├── notebooks/              # Jupyter notebooks for analysis
├── requirements.txt
└── README.md
```

##  Architecture

SAMM2D consists of several key components:

1. **Patch Embedding**: Converts 2D image patches into token embeddings
2. **Modal-Specific Encoders**: Separate transformer encoders for each imaging modality
3. **Scale-Specific Encoders**: Encoders for multi-scale feature extraction
4. **Cross-Modal Attention**: Bidirectional attention between TOF and MRA modalities
5. **Cross-Scale Attention**: Multi-scale feature fusion mechanism
6. **Fusion Network**: Combines all features for final classification

##  Ablation Studies

| Configuration | AUC-ROC | Δ vs Baseline |
|--------------|---------|---------------|
| Baseline (ResNet18) | 0.680 | - |
| Single Modal (TOF only) | 0.710 | +4.4% |
| Single Scale | 0.700 | +2.9% |
| Multi-Modal (No Cross-Attn) | 0.740 | +8.8% |
| Multi-Scale (No Cross-Attn) | 0.730 | +7.4% |
| SAMM2D (No Fusion) | 0.760 | +11.8% |
| **SAMM2D (Full)** | **0.780** | **+14.7%** |

## Hyperparameters

Key hyperparameters (see `configs/default_config.json` for all parameters):

```json
{
  "embed_dim": 256,
  "depth": 3,
  "num_heads": 4,
  "dropout": 0.1,
  "batch_size": 8,
  "learning_rate": 1e-4,
  "num_epochs": 50
}
```

##  Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{titikhsha,tak,samm2d,
  title={SAMM2D: Scale-Aware Multi-Modal 2D Transformer for Intracranial Aneurysm Detection},
  author={Titiksha, Antara, Tak, Divyanshu},
  
}
```

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- RSNA for providing the Intracranial Aneurysm Detection dataset
- The PyTorch team for the excellent deep learning framework

## Dataset
Rudie, J., Calabrese, E., Ball, R., Chang, P., Chen, R., Colak, E., Correia de Verdier, M., Prevedelo, L., Richards, T., Saluja, R., Zaharchuk, G., Sho, J., & Vazirabad, M. (2025). RSNA intracranial aneurysm detection [Data set]. Kaggle. https://kaggle.com/competitions/rsna-2025-intracranial-aneurysm-detection 

## 📧 Contact

Antara Titikhsha - Carnegie Mellon University  
Email: [antitikhsha@cmu.edu]

---

**Note**: This is research code. For clinical use, please consult with medical professionals and follow appropriate regulatory guidelines.
