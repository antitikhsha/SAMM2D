"""
Inference script for SAMM2D model.
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import json
from PIL import Image
import cv2
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.samm2d import SAMM2D


class Predictor:
    """Predictor class for SAMM2D inference."""
    
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
        
        print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
        print(f"✓ Running on {self.device}")
    
    def preprocess(self, img_path):
        """Preprocess a single image."""
        # Load image
        if str(img_path).endswith('.npy'):
            img = np.load(img_path)
            if len(img.shape) == 3:
                img = img[img.shape[0] // 2]  # Take middle slice
        else:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        # Normalize to [0, 255]
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
        
        # Resize
        img = cv2.resize(img, (self.config['img_size'], self.config['img_size']))
        
        return img
    
    def create_multi_scale(self, img):
        """Create multi-scale versions."""
        h, w = img.shape[:2]
        
        # Original
        original = img.copy()
        
        # Downsampled
        downsampled = cv2.resize(img, (w // 2, h // 2))
        downsampled = cv2.resize(downsampled, (self.config['img_size'], self.config['img_size']))
        
        # Upsampled
        upsampled = cv2.resize(img, (w * 2, h * 2))
        upsampled = cv2.resize(upsampled, (self.config['img_size'], self.config['img_size']))
        
        return original, downsampled, upsampled
    
    def predict(self, img_path):
        """
        Run inference on a single image.
        
        Returns:
            probability: Probability of aneurysm presence
            prediction: Binary prediction (0 or 1)
        """
        # Preprocess
        img = self.preprocess(img_path)
        
        # Create multi-scale
        orig, down, up = self.create_multi_scale(img)
        
        # Use same for both modalities (in practice, these would be different)
        tof = img.copy()
        mra = img.copy()
        
        # Convert to tensors
        tof = torch.from_numpy(tof).float().unsqueeze(0).unsqueeze(0) / 255.0
        mra = torch.from_numpy(mra).float().unsqueeze(0).unsqueeze(0) / 255.0
        orig = torch.from_numpy(orig).float().unsqueeze(0).unsqueeze(0) / 255.0
        down = torch.from_numpy(down).float().unsqueeze(0).unsqueeze(0) / 255.0
        up = torch.from_numpy(up).float().unsqueeze(0).unsqueeze(0) / 255.0
        
        # Move to device
        tof = tof.to(self.device)
        mra = mra.to(self.device)
        orig = orig.to(self.device)
        down = down.to(self.device)
        up = up.to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(tof, mra, orig, down, up)
            probs = torch.softmax(logits, dim=1)
            prob_aneurysm = probs[0, 1].item()
            pred = int(prob_aneurysm >= 0.5)
        
        return prob_aneurysm, pred
    
    def predict_batch(self, img_paths):
        """Run inference on multiple images."""
        results = []
        
        for img_path in tqdm(img_paths, desc="Processing"):
            try:
                prob, pred = self.predict(img_path)
                results.append({
                    'filename': Path(img_path).name,
                    'probability': float(prob),
                    'prediction': int(pred),
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'filename': Path(img_path).name,
                    'probability': None,
                    'prediction': None,
                    'status': f'error: {str(e)}'
                })
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Run inference with SAMM2D')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image or directory')
    parser.add_argument('--output', type=str, default='results/predictions',
                       help='Output directory')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create predictor
    predictor = Predictor(args.checkpoint, config)
    
    # Get image paths
    input_path = Path(args.input)
    if input_path.is_dir():
        img_paths = list(input_path.glob('*.npy')) + \
                   list(input_path.glob('*.png')) + \
                   list(input_path.glob('*.jpg'))
    else:
        img_paths = [input_path]
    
    print(f"Found {len(img_paths)} images")
    
    # Run predictions
    results = predictor.predict_batch(img_paths)
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'predictions.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print summary
    successful = sum(1 for r in results if r['status'] == 'success')
    positive = sum(1 for r in results if r['status'] == 'success' and r['prediction'] == 1)
    
    print(f"\n{'='*60}")
    print("INFERENCE SUMMARY")
    print(f"{'='*60}")
    print(f"Total images:      {len(results)}")
    print(f"Successful:        {successful}")
    print(f"Positive cases:    {positive} ({positive/successful*100:.1f}%)")
    print(f"Negative cases:    {successful-positive} ({(successful-positive)/successful*100:.1f}%)")
    print(f"{'='*60}")
    print(f"\n✓ Results saved to {output_dir / 'predictions.json'}")


if __name__ == "__main__":
    main()
