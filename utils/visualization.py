"""
Grad-CAM visualization for SAMM2D model interpretability.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) for visualization.
    
    Args:
        model: PyTorch model
        target_layer: Layer to visualize (e.g., last conv layer)
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensors, class_idx=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensors: Tuple of (tof, mra, orig, down, up)
            class_idx: Target class index (None for predicted class)
        
        Returns:
            cam: Grad-CAM heatmap [H, W]
        """
        self.model.eval()
        
        # Forward pass
        tof, mra, scale_orig, scale_down, scale_up = input_tensors
        logits = self.model(tof, mra, scale_orig, scale_down, scale_up)
        
        # Get target class
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        class_score = logits[:, class_idx]
        class_score.backward()
        
        # Compute Grad-CAM
        gradients = self.gradients  # [B, C, H, W]
        activations = self.activations  # [B, C, H, W]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to input size
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Convert to numpy
        cam = cam.squeeze().cpu().numpy()
        
        return cam
    
    def visualize(self, image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Overlay Grad-CAM heatmap on image.
        
        Args:
            image: Original image [H, W] or [H, W, 3]
            cam: Grad-CAM heatmap [H, W]
            alpha: Blending factor
            colormap: OpenCV colormap
        
        Returns:
            vis: Visualization image
        """
        # Ensure image is uint8
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Convert grayscale to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Resize CAM to image size
        h, w = image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Apply colormap
        cam_colored = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), colormap)
        
        # Overlay
        vis = cv2.addWeighted(image, 1-alpha, cam_colored, alpha, 0)
        
        return vis


def visualize_attention_maps(model, input_tensors, save_path=None):
    """
    Visualize multi-modal and multi-scale attention.
    
    Args:
        model: SAMM2D model
        input_tensors: Tuple of (tof, mra, orig, down, up)
        save_path: Path to save visualization
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('SAMM2D Attention Visualization', fontsize=16, weight='bold')
    
    tof, mra, scale_orig, scale_down, scale_up = input_tensors
    
    # Convert tensors to images for display
    images = {
        'TOF': tof[0, 0].cpu().numpy(),
        'MRA': mra[0, 0].cpu().numpy(),
        'Original': scale_orig[0, 0].cpu().numpy(),
        'Downsampled': scale_down[0, 0].cpu().numpy(),
        'Upsampled': scale_up[0, 0].cpu().numpy()
    }
    
    # Display input images
    axes[0, 0].imshow(images['TOF'], cmap='gray')
    axes[0, 0].set_title('TOF Input', fontsize=12, weight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(images['MRA'], cmap='gray')
    axes[0, 1].set_title('MRA Input', fontsize=12, weight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(images['Original'], cmap='gray')
    axes[0, 2].set_title('Original Scale', fontsize=12, weight='bold')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(images['Downsampled'], cmap='gray')
    axes[1, 0].set_title('Downsampled Scale', fontsize=12, weight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(images['Upsampled'], cmap='gray')
    axes[1, 1].set_title('Upsampled Scale', fontsize=12, weight='bold')
    axes[1, 1].axis('off')
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        logits = model(*input_tensors)
        probs = torch.softmax(logits, dim=1)
        pred_class = logits.argmax(dim=1).item()
        pred_prob = probs[0, pred_class].item()
    
    # Add prediction text
    pred_text = f"Prediction: {'Aneurysm' if pred_class == 1 else 'No Aneurysm'}\n"
    pred_text += f"Confidence: {pred_prob:.2%}"
    axes[1, 2].text(0.5, 0.5, pred_text, ha='center', va='center',
                   fontsize=14, weight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    from models.samm2d import SAMM2D
    
    # Create model
    model = SAMM2D()
    model.eval()
    
    # Dummy inputs
    batch_size = 1
    tof = torch.randn(batch_size, 1, 224, 224)
    mra = torch.randn(batch_size, 1, 224, 224)
    orig = torch.randn(batch_size, 1, 224, 224)
    down = torch.randn(batch_size, 1, 224, 224)
    up = torch.randn(batch_size, 1, 224, 224)
    
    input_tensors = (tof, mra, orig, down, up)
    
    # Visualize
    visualize_attention_maps(model, input_tensors, save_path='attention_viz.png')
