"""
Setup for Big Sleep MLX
Apple Silicon optimized version using MLX
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read version
version_file = Path(__file__).parent / 'big_sleep_mlx' / 'version.py'
version = {}
with open(version_file) as f:
    exec(f.read(), version)

# Read README
readme_file = Path(__file__).parent / 'big_sleep_mlx' / 'README.md'
long_description = readme_file.read_text() if readme_file.exists() else ''

setup(
    name='big-sleep-mlx',
    version=version.get('__version__', '1.0.0'),
    description='Big Sleep - Apple Silicon optimized with MLX',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Ryan Murdock, Phil Wang, MLX Contributors',
    author_email='lucidrains@gmail.com',
    url='https://github.com/LukasOpp/big-sleep-apple-silicon',
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'dream-mlx = big_sleep_mlx.cli:main',
        ],
    },
    install_requires=[
        'mlx>=0.0.5',
        'numpy>=1.20.0',
        'Pillow>=8.0.0',
        'tqdm>=4.50.0',
        'fire>=0.4.0',
        'ftfy>=6.0.0',
        'regex>=2021.0.0',
    ],
    extras_require={
        'conversion': ['torch>=1.7.1'],  # For converting PyTorch weights
    },
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: MacOS :: MacOS X',
    ],
    keywords=[
        'artificial intelligence',
        'deep learning',
        'transformers',
        'text to image',
        'generative adversarial networks',
        'apple silicon',
        'mlx',
        'm1',
        'm2',
        'm3'
    ],
)
