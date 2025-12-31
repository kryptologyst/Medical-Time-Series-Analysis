"""Device management and reproducibility utilities for medical time series analysis."""

import os
import random
from typing import Optional, Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
        deterministic: Whether to use deterministic algorithms.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        # Enable deterministic algorithms
        torch.use_deterministic_algorithms(True)
        cudnn.deterministic = True
        cudnn.benchmark = False
        
        # Set environment variables for reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_device(device: Optional[str] = None, fallback_order: list[str] = None) -> torch.device:
    """Get the best available device for computation.
    
    Args:
        device: Specific device to use. If None, auto-detect.
        fallback_order: Order of devices to try if auto-detecting.
        
    Returns:
        torch.device: The selected device.
    """
    if fallback_order is None:
        fallback_order = ["cuda", "mps", "cpu"]
    
    if device is not None:
        return torch.device(device)
    
    for device_name in fallback_order:
        if device_name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif device_name == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        elif device_name == "cpu":
            return torch.device("cpu")
    
    # Fallback to CPU if nothing else works
    return torch.device("cpu")


def get_device_info() -> dict[str, Union[str, bool, int]]:
    """Get information about available devices.
    
    Returns:
        Dictionary containing device information.
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "cpu_count": os.cpu_count(),
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_current_device": torch.cuda.current_device(),
            "cuda_device_name": torch.cuda.get_device_name(),
            "cuda_memory_allocated": torch.cuda.memory_allocated(),
            "cuda_memory_reserved": torch.cuda.memory_reserved(),
        })
    
    return info


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_memory_usage(device: torch.device) -> dict[str, float]:
    """Get memory usage information for a device.
    
    Args:
        device: The device to check memory for.
        
    Returns:
        Dictionary containing memory usage information in MB.
    """
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1024**2
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "max_allocated_mb": max_allocated,
        }
    else:
        # For CPU/MPS, we can't easily get memory usage
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}
