"""
CLI for Big Sleep MLX
"""
import fire
import random as rnd
from big_sleep_mlx import ImagineMLX
from big_sleep_mlx.version import __version__


def train(
    text=None,
    img=None,
    text_min="",
    lr=0.07,
    image_size=512,
    epochs=20,
    iterations=1050,
    save_every=50,
    overwrite=False,
    save_progress=False,
    save_best=False,
    num_cutouts=128,
    center_bias=False,
    larger_model=False,
    seed=0,
    random=False
):
    """
    Train Big Sleep MLX model
    
    Args:
        text: Text prompt(s) to generate image from (use | to separate multiple prompts)
        img: Path to image file (optional)
        text_min: Text prompt(s) to avoid (use | to separate multiple prompts)
        lr: Learning rate
        image_size: Output image size (128, 256, or 512)
        epochs: Number of epochs
        iterations: Iterations per epoch
        save_every: Save image every N iterations
        overwrite: Overwrite existing images
        save_progress: Save intermediate images
        save_best: Save best image (highest CLIP score)
        num_cutouts: Number of cutouts for CLIP evaluation
        center_bias: Use center-biased cutouts
        larger_model: Use larger CLIP model (ViT-L/14)
        seed: Random seed
        random: Use random seed
    """
    print(f'Big Sleep MLX v{__version__}')
    print('Running on Apple Silicon with MLX acceleration')
    
    if random:
        seed = rnd.randint(0, 1000000)
    
    imagine = ImagineMLX(
        text=text,
        img=img,
        text_min=text_min,
        lr=lr,
        image_size=image_size,
        epochs=epochs,
        iterations=iterations,
        save_every=save_every,
        save_progress=save_progress,
        save_best=save_best,
        num_cutouts=num_cutouts,
        center_bias=center_bias,
        larger_clip=larger_model,
        seed=seed
    )
    
    if not overwrite and imagine.filename.exists():
        answer = input('Image already exists, overwrite? (y/n) ').lower()
        if answer not in ('yes', 'y'):
            return
    
    imagine()


def main():
    fire.Fire(train)


if __name__ == '__main__':
    main()
