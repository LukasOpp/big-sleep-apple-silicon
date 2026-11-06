"""
Big Sleep MLX - Main implementation using MLX for Apple Silicon
"""
import os
import sys
from datetime import datetime
from pathlib import Path
import signal

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from PIL import Image
from tqdm import tqdm, trange

from big_sleep_mlx.clip_mlx import load_clip_mlx, tokenize
from big_sleep_mlx.biggan_mlx import load_biggan_mlx


# Graceful keyboard interrupt
terminate = False

def signal_handling(signum, frame):
    print('detecting keyboard interrupt, gracefully exiting')
    global terminate
    terminate = True

signal.signal(signal.SIGINT, signal_handling)


# Helper functions
def exists(val):
    return val is not None


def create_text_path(text=None, img=None, encoding=None):
    input_name = ""
    if text is not None:
        input_name += text
    if img is not None:
        if isinstance(img, str):
            img_name = "".join(img.split(".")[:-1])
            img_name = img_name.split("/")[-1]
        else:
            img_name = "PIL_img"
        input_name += "_" + img_name
    if encoding is not None:
        input_name = "your_encoding"
    return input_name.replace("-", "_").replace(",", "").replace(" ", "_").replace("|", "--").strip('-_')[:255]


def normalize_image(img):
    """Normalize image for CLIP"""
    mean = mx.array([0.48145466, 0.4578275, 0.40821073]).reshape(1, 1, 1, 3)
    std = mx.array([0.26862954, 0.26130258, 0.27577711]).reshape(1, 1, 1, 3)
    return (img - mean) / std


def rand_cutout(image, size, center_bias=False):
    """Random cutout from image"""
    _, H, W, _ = image.shape
    max_offset = W - size
    
    if center_bias:
        # Sample near center
        center = max_offset // 2
        offset_x = int(np.random.normal(center, center / 3))
        offset_y = int(np.random.normal(center, center / 3))
        offset_x = np.clip(offset_x, 0, max_offset)
        offset_y = np.clip(offset_y, 0, max_offset)
    else:
        offset_x = np.random.randint(0, max_offset + 1)
        offset_y = np.random.randint(0, max_offset + 1)
    
    cutout = image[:, offset_y:offset_y + size, offset_x:offset_x + size, :]
    return cutout


def resize_image(img, target_size):
    """Resize image using nearest neighbor"""
    # Simple nearest neighbor resize
    B, H, W, C = img.shape
    scale_h = target_size / H
    scale_w = target_size / W
    
    # Create coordinate grids
    new_h = mx.arange(target_size).reshape(-1, 1) / scale_h
    new_w = mx.arange(target_size).reshape(1, -1) / scale_w
    
    new_h = mx.clip(new_h.astype(mx.int32), 0, H - 1)
    new_w = mx.clip(new_w.astype(mx.int32), 0, W - 1)
    
    # Index into original image
    resized = img[:, new_h, new_w, :]
    return resized


class LatentsMLX(nn.Module):
    """Latent codes for BigGAN"""
    def __init__(
        self,
        num_latents: int = 15,
        num_classes: int = 1000,
        z_dim: int = 128
    ):
        super().__init__()
        self.normu = mx.random.normal((num_latents, z_dim))
        self.cls = mx.random.normal((num_latents, num_classes)) * 0.3 - 3.9

    def __call__(self):
        classes = mx.sigmoid(self.cls)
        return self.normu, classes


class BigSleepMLX(nn.Module):
    """Big Sleep model using MLX"""
    def __init__(
        self,
        num_cutouts: int = 128,
        loss_coef: float = 100,
        image_size: int = 512,
        center_bias: bool = False,
        larger_clip: bool = False
    ):
        super().__init__()
        self.loss_coef = loss_coef
        self.image_size = image_size
        self.num_cutouts = num_cutouts
        self.center_bias = center_bias
        
        # Load models
        model_name = 'ViT-L/14' if larger_clip else 'ViT-B/32'
        self.perceptor = load_clip_mlx(model_name)
        self.biggan = load_biggan_mlx(image_size)
        
        # Latent codes
        self.latents = LatentsMLX(
            num_latents=15,
            num_classes=1000,
            z_dim=128
        )

    def reset(self):
        """Reset latent codes"""
        self.latents = LatentsMLX(
            num_latents=15,
            num_classes=1000,
            z_dim=128
        )

    def generate_image(self):
        """Generate image from current latents"""
        z, classes = self.latents()
        img = self.biggan(z, classes[0], truncation=1.0)
        # Convert from [-1, 1] to [0, 1]
        img = (img + 1) / 2
        return img

    def forward_loss(self, text_embeds, text_min_embeds=None):
        """Forward pass with loss computation"""
        # Generate image
        img = self.generate_image()
        
        # Create cutouts
        pieces = []
        for _ in range(self.num_cutouts):
            size = int(self.image_size * np.clip(np.random.normal(0.8, 0.3), 0.5, 0.95))
            cutout = rand_cutout(img, size, self.center_bias)
            cutout = resize_image(cutout, 224)
            pieces.append(cutout)
        
        into = mx.concatenate(pieces, axis=0)
        into = normalize_image(into)
        
        # Get image embeddings
        image_embed = self.perceptor.encode_image(into)
        
        # Compute losses
        z, classes = self.latents()
        
        # Latent regularization
        lat_loss = mx.mean(mx.abs(1 - mx.std(z, axis=1))) + \
                   mx.mean(mx.abs(mx.mean(z, axis=1))) + \
                   4 * mx.maximum(mx.mean(z ** 2), mx.array(1.0))
        
        # Class regularization
        cls_loss = mx.mean((50 * mx.sort(classes, axis=1)[:, :999]) ** 2)
        
        # CLIP similarity loss
        results = []
        for txt_embed in text_embeds:
            sim = -self.loss_coef * mx.mean(
                mx.sum(txt_embed * image_embed, axis=-1) /
                (mx.sqrt(mx.sum(txt_embed**2, axis=-1)) * mx.sqrt(mx.sum(image_embed**2, axis=-1)))
            )
            results.append(sim)
        
        if text_min_embeds:
            for txt_min_embed in text_min_embeds:
                sim = self.loss_coef * mx.mean(
                    mx.sum(txt_min_embed * image_embed, axis=-1) /
                    (mx.sqrt(mx.sum(txt_min_embed**2, axis=-1)) * mx.sqrt(mx.sum(image_embed**2, axis=-1)))
                )
                results.append(sim)
        
        sim_loss = sum(results) / len(results)
        
        return img, (lat_loss, cls_loss, sim_loss)


class ImagineMLX:
    """Main interface for Big Sleep MLX"""
    def __init__(
        self,
        text: str = None,
        img: str = None,
        text_min: str = "",
        lr: float = 0.07,
        image_size: int = 512,
        epochs: int = 20,
        iterations: int = 1050,
        save_every: int = 50,
        save_progress: bool = False,
        save_best: bool = False,
        num_cutouts: int = 128,
        center_bias: bool = False,
        larger_clip: bool = False,
        seed: int = None
    ):
        self.text = text
        self.text_min = text_min
        self.lr = lr
        self.image_size = image_size
        self.epochs = epochs
        self.iterations = iterations
        self.save_every = save_every
        self.save_progress = save_progress
        self.save_best = save_best
        self.seed = seed
        
        if seed is not None:
            mx.random.seed(seed)
            np.random.seed(seed)
        
        # Initialize model
        self.model = BigSleepMLX(
            num_cutouts=num_cutouts,
            loss_coef=100,
            image_size=image_size,
            center_bias=center_bias,
            larger_clip=larger_clip
        )
        
        # Create optimizer
        # MLX uses different optimizer API
        self.learning_rate = lr
        
        # Encode text
        self.text_path = create_text_path(text=text, img=img)
        self.filename = Path(f'./{self.text_path}.png')
        self.encode_text(text, text_min)
        
        self.current_best_score = 0

    def encode_text(self, text, text_min=""):
        """Encode text prompts"""
        self.encoded_texts = {"max": [], "min": []}
        
        if text is not None and "|" in text:
            texts = text.split("|")
            for t in texts:
                tokens = tokenize(t)
                with mx.no_grad():
                    encoding = self.model.perceptor.encode_text(tokens)
                self.encoded_texts["max"].append(encoding)
        elif text is not None:
            tokens = tokenize(text)
            with mx.no_grad():
                encoding = self.model.perceptor.encode_text(tokens)
            self.encoded_texts["max"].append(encoding)
        
        if text_min and "|" in text_min:
            texts = text_min.split("|")
            for t in texts:
                tokens = tokenize(t)
                with mx.no_grad():
                    encoding = self.model.perceptor.encode_text(tokens)
                self.encoded_texts["min"].append(encoding)
        elif text_min:
            tokens = tokenize(text_min)
            with mx.no_grad():
                encoding = self.model.perceptor.encode_text(tokens)
            self.encoded_texts["min"].append(encoding)

    def save_image(self, img, path):
        """Save image to disk"""
        # Convert MLX array to numpy
        img_np = np.array(img[0])
        
        # Convert from [H, W, C] to [H, W, C] and scale to [0, 255]
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        
        # Save using PIL
        pil_img = Image.fromarray(img_np)
        pil_img.save(str(path))

    def train_step(self, optimizer):
        """Single training step"""
        def loss_fn():
            img, losses = self.model.forward_loss(
                self.encoded_texts["max"],
                self.encoded_texts["min"] if self.encoded_texts["min"] else None
            )
            total_loss = sum(losses)
            return total_loss, (img, losses)
        
        # Compute gradients and update
        (loss, (img, losses)), grads = mx.value_and_grad(loss_fn, has_aux=True)()
        optimizer.update(self.model, grads)
        mx.eval(self.model.parameters())
        
        return img, loss, losses

    def __call__(self):
        """Main training loop"""
        print(f'Imagining "{self.text_path}"...')
        
        # Simple optimizer (using SGD-like updates)
        class SimpleOptimizer:
            def __init__(self, learning_rate):
                self.lr = learning_rate
            
            def update(self, model, grads):
                # Update latents
                if 'latents' in grads and 'normu' in grads['latents']:
                    model.latents.normu = model.latents.normu - self.lr * grads['latents']['normu']
                if 'latents' in grads and 'cls' in grads['latents']:
                    model.latents.cls = model.latents.cls - self.lr * grads['latents']['cls']
        
        optimizer = SimpleOptimizer(self.learning_rate)
        
        image_pbar = tqdm(total=self.epochs * self.iterations // self.save_every, desc='image updates')
        
        for epoch in trange(self.epochs, desc='epochs'):
            if terminate:
                break
            
            for i in trange(self.iterations, desc='iteration', leave=False):
                if terminate:
                    break
                
                img, loss, losses = self.train_step(optimizer)
                
                if (i + 1) % self.save_every == 0:
                    self.save_image(img, self.filename)
                    image_pbar.update(1)
                    
                    if self.save_progress:
                        num = (epoch * self.iterations + i) // self.save_every
                        self.save_image(img, Path(f'./{self.text_path}.{num}.png'))
                    
                    if self.save_best:
                        score = -losses[2]  # Negative sim_loss
                        if score < self.current_best_score:
                            self.current_best_score = score
                            self.save_image(img, Path(f'./{self.text_path}.best.png'))
        
        image_pbar.close()
        print(f'Saved final image to {self.filename}')
