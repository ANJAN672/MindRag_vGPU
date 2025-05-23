import os
import logging
import torch
import subprocess
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_nvidia_smi():
    """Check if NVIDIA GPU is available using nvidia-smi command"""
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0 and 'NVIDIA-SMI' in result.stdout:
            logger.info("NVIDIA GPU detected via nvidia-smi")
            return True
        return False
    except Exception:
        return False

def get_device_config():
    """
    Determine the best device configuration based on available hardware.
    Returns a dictionary with device settings.
    """
    # Force CUDA detection if environment variable is set
    force_cuda = os.environ.get("FORCE_CUDA", "").lower() in ("1", "true", "yes")
    
    config = {
        "device": "cpu",
        "device_index": -1,
        "torch_dtype": torch.float32,
        "gpu_layers": 0,
        "use_gpu": False
    }
    
    # Check if CUDA is available through PyTorch
    cuda_available = torch.cuda.is_available()
    
    # If PyTorch doesn't detect CUDA but nvidia-smi does, we have a configuration issue
    if not cuda_available and check_nvidia_smi():
        logger.warning("NVIDIA GPU detected via nvidia-smi but PyTorch can't access it.")
        logger.warning("This might be due to PyTorch being installed without CUDA support.")
        logger.warning("Try reinstalling PyTorch with CUDA support.")
        
        if force_cuda:
            logger.info("FORCE_CUDA is set, attempting to use GPU anyway")
            cuda_available = True
    
    # Check if CUDA is available
    if cuda_available:
        try:
            # Get device count and properties
            device_count = torch.cuda.device_count()
            if device_count > 0:
                # Get device properties
                device_name = torch.cuda.get_device_name(0)
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert to GB
                
                logger.info(f"Found GPU: {device_name} with {total_memory:.2f} GB memory")
                
                # Set device configuration
                config["device"] = "cuda"
                config["device_index"] = 0
                config["use_gpu"] = True
                
                # Use half precision for GPUs to save memory
                config["torch_dtype"] = torch.float16
                
                # Determine number of layers to offload to GPU based on available memory
                # For GTX 1650 with ~4GB VRAM, we need to optimize for speed
                if "1650" in device_name:
                    # Optimized setting for GTX 1650 based on testing - use more layers for speed
                    config["gpu_layers"] = 32  # Increased from 24 to 32 for better performance
                    logger.info(f"Detected GTX 1650: Using {config['gpu_layers']} GPU layers for LLM offloading")
                elif total_memory > 8:
                    # For GPUs with more memory
                    config["gpu_layers"] = 40  # Increased for better performance
                    logger.info(f"Using {config['gpu_layers']} GPU layers for LLM offloading")
                else:
                    # Default for mid-range GPUs
                    config["gpu_layers"] = 32  # Increased from 24 to 32
                    logger.info(f"Using {config['gpu_layers']} GPU layers for LLM offloading")
                
                # Check if environment variable overrides are present
                env_gpu_layers = os.getenv("GPU_LAYERS")
                if env_gpu_layers and env_gpu_layers.isdigit():
                    config["gpu_layers"] = int(env_gpu_layers)
                    logger.info(f"Overriding GPU layers from environment: {config['gpu_layers']}")
                
                return config
        except Exception as e:
            logger.warning(f"Error configuring CUDA device: {e}")
            logger.info("Falling back to CPU")
    
    logger.info("No CUDA device available, using CPU")
    return config

# Get the device configuration once at module import
device_config = get_device_config()