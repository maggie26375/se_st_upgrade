"""
Setup script for SE+ST Upgrade Package
"""

from setuptools import setup, find_packages
import os

setup(
    name="se_st_upgrade",
    version="0.1.0",
    author="maggie26375",
    description="SE+ST Upgrade with AutoTune, AdapterTune, and RL optimization",
    url="https://github.com/maggie26375/se_st_upgrade",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "anndata>=0.9.0",
        "scanpy>=1.9.0",
        "tqdm>=4.65.0",
        "h5py>=3.8.0",
        "optuna>=3.0.0",  # For AutoTune
        "peft>=0.5.0",    # For LoRA adapters
    ],
    entry_points={
        "console_scripts": [
            "se-st-train=se_st_upgrade.cli.train:main",
            "se-st-infer=se_st_upgrade.cli.infer:main",
            "se-st-adaptertune=se_st_upgrade.cli.adaptertune:main",
            "se-st-rltune=se_st_upgrade.cli.rltune:main",
            "se-st-autotune=se_st_upgrade.cli.autotune:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": [
            "configs/*.yaml",
        ],
    },
    zip_safe=False,
)
