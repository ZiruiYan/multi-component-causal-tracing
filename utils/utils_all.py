import numpy as np
import random
import torch


def seed_everything(seed: int = 42):
    """
    Seed everything for reproducibility.
    
    Args:
        seed (int): The seed value to use for all libraries.
    """
    # Seed the built-in random module
    random.seed(seed)
    
    # Seed numpy
    np.random.seed(seed)
    
    # Seed PyTorch
    torch.manual_seed(seed)
    # Check if CUDA is available and seed it
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Check if MPS is available and seed it
    if hasattr(torch.backends, "mps"):
        torch.manual_seed(seed)  # MPS uses the same manual seed function as CPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def scheduling(initial, step, method=None, base = 2, degree = 2):
    if method == 'linear':
        return initial * (step+1)
    if method == 'exp':
        return initial * (base**step)
    if method == 'poly':
        return initial * (step ** degree)
    return None