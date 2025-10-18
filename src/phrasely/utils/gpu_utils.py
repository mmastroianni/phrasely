import logging

import torch

logger = logging.getLogger(__name__)

try:
    import cupy
except ImportError:
    cupy = None
    logger.warning("CuPy not available – GPU utilities limited to CPU fallback.")


def is_gpu_available() -> bool:
    if cupy is None:
        return False
    try:
        _ = cupy.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


def get_device_info() -> dict:
    """Return a dictionary with total VRAM (in GB) and device name, or fallback info."""
    info = {"name": "CPU", "total": 0.0}

    try:
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            total_gb = round(props.total_memory / (1024**3), 1)
            info = {"name": props.name, "total": total_gb}
        elif cupy.cuda.runtime.getDeviceCount() > 0:
            dev = cupy.cuda.Device()
            # attrs = dev.attributes
            total_gb = round(dev.mem_info[1] / (1024**3), 1)
            info = {"name": f"CuPy device {dev.id}", "total": total_gb}
    except Exception as e:
        logger.warning(f"Could not query GPU info: {e}")

    return info
