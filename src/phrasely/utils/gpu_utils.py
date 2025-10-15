import importlib
import logging

logger = logging.getLogger(__name__)

def is_gpu_available() -> bool:
    """Check if CuPy is available and a GPU device is accessible."""
    try:
        cp_spec = importlib.util.find_spec("cupy")
        if cp_spec is None:
            return False
        import cupy as cp
        _ = cp.cuda.Device(0)
        return True
    except Exception:
        return False


def get_device_info() -> str:
    """Return a safe, human-readable device string."""
    try:
        import cupy as cp
        device = cp.cuda.Device(0)
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        name = props.get("name", "Unnamed GPU")
        return f"GPU: {name}"
    except Exception as e:
        logger.warning(f"Could not query GPU info: {e}")
        return "CPU"
