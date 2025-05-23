import uvicorn
import torch
import logging
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_gpu():
    """Check and log GPU availability"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"GPU {i}: {device_name} ({memory:.2f} GB)")
        logger.info("CUDA is available - GPU acceleration enabled")
        return True
    else:
        logger.warning("CUDA is not available - running in CPU-only mode")
        return False

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Check GPU availability
    has_gpu = check_gpu()
    
    # Set environment variable for GPU usage
    if has_gpu:
        os.environ["DEVICE"] = "cuda"
    
    # Get host and port from environment or use defaults
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run("app.main:app", host=host, port=port, reload=True)