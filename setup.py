"""
    Setup file for my_biblio.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 4.6.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""

from setuptools import setup
import sys
import subprocess

# Function to install torch
def install_torch():
    try:
        import torch
    except ImportError:
        print("Torch not found. Installing torch...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])

# Install torch before running setup
install_torch()

# Now we can safely run the setup() function
setup(
    name="trackmol",
    version="1.0.1",
    packages=["trackmol"],
    install_requires=["pandas", "numpy","scipy","scikit-learn","matplotlib","fbm","tqdm","torch"]#"umap-learn[parametric_umap]>=0.5.2","torch"]
        #"torch>=1.13.1", "torch-geometric", "torch-cluster==1.6.3", "torch-scatter",
        #"torch-sparse", "torch-spline-conv", "torchmetrics","pytorch-lightning"
    #]
)
