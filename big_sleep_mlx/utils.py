"""
Utilities for Big Sleep MLX
Including weight conversion from PyTorch to MLX
"""
import mlx.core as mx
import numpy as np
from pathlib import Path
from typing import Dict, Any
import json


def convert_pytorch_to_mlx(pytorch_state_dict: Dict[str, Any]) -> Dict[str, mx.array]:
    """
    Convert PyTorch state dict to MLX format
    
    Args:
        pytorch_state_dict: PyTorch model state dict
    
    Returns:
        MLX-compatible state dict
    """
    mlx_state = {}
    
    for key, value in pytorch_state_dict.items():
        # Convert PyTorch tensor to numpy, then to MLX
        if hasattr(value, 'cpu'):
            # PyTorch tensor
            np_value = value.cpu().numpy()
        else:
            # Already numpy
            np_value = np.array(value)
        
        # Convert to MLX array
        mlx_state[key] = mx.array(np_value)
    
    return mlx_state


def save_mlx_weights(state_dict: Dict[str, mx.array], path: str):
    """Save MLX weights to file"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert MLX arrays to numpy for saving
    np_state = {k: np.array(v) for k, v in state_dict.items()}
    
    # Save as compressed numpy archive
    np.savez_compressed(str(path), **np_state)
    print(f"Saved MLX weights to {path}")


def load_mlx_weights(path: str) -> Dict[str, mx.array]:
    """Load MLX weights from file"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Weights file not found: {path}")
    
    # Load numpy archive
    np_state = np.load(str(path))
    
    # Convert to MLX arrays
    mlx_state = {k: mx.array(v) for k, v in np_state.items()}
    
    print(f"Loaded MLX weights from {path}")
    return mlx_state


def convert_clip_weights(pytorch_path: str, mlx_path: str):
    """
    Convert CLIP weights from PyTorch to MLX format
    
    Args:
        pytorch_path: Path to PyTorch CLIP checkpoint
        mlx_path: Path to save MLX weights
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for weight conversion. Install with: pip install torch")
    
    print(f"Loading PyTorch CLIP from {pytorch_path}...")
    checkpoint = torch.load(pytorch_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    print("Converting to MLX format...")
    mlx_state = convert_pytorch_to_mlx(state_dict)
    
    print(f"Saving MLX weights to {mlx_path}...")
    save_mlx_weights(mlx_state, mlx_path)
    
    print("Conversion complete!")


def convert_biggan_weights(pytorch_path: str, mlx_path: str):
    """
    Convert BigGAN weights from PyTorch to MLX format
    
    Args:
        pytorch_path: Path to PyTorch BigGAN checkpoint
        mlx_path: Path to save MLX weights
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for weight conversion. Install with: pip install torch")
    
    print(f"Loading PyTorch BigGAN from {pytorch_path}...")
    checkpoint = torch.load(pytorch_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    print("Converting to MLX format...")
    mlx_state = convert_pytorch_to_mlx(state_dict)
    
    print(f"Saving MLX weights to {mlx_path}...")
    save_mlx_weights(mlx_state, mlx_path)
    
    print("Conversion complete!")


def download_and_convert_models():
    """
    Download pre-trained models and convert to MLX format
    """
    print("Downloading pre-trained models...")
    print("This feature is not yet implemented.")
    print("Please manually convert PyTorch weights using:")
    print("  - convert_clip_weights(pytorch_path, mlx_path)")
    print("  - convert_biggan_weights(pytorch_path, mlx_path)")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert PyTorch weights to MLX format')
    parser.add_argument('--model', choices=['clip', 'biggan'], required=True, help='Model type')
    parser.add_argument('--pytorch-path', required=True, help='Path to PyTorch checkpoint')
    parser.add_argument('--mlx-path', required=True, help='Path to save MLX weights')
    
    args = parser.parse_args()
    
    if args.model == 'clip':
        convert_clip_weights(args.pytorch_path, args.mlx_path)
    elif args.model == 'biggan':
        convert_biggan_weights(args.pytorch_path, args.mlx_path)
