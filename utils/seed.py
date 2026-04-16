"""
Global seed utility for full reproducibility.

Sets seeds for: numpy, Python random, PyTorch (CPU + CUDA).
"""
import os
import random
import numpy as np

def set_global_seed(seed: int) -> None:
    """Set seed for all RNG sources used in the project."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    except ImportError:
        pass
