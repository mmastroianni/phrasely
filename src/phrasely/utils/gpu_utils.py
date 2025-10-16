import logging
import os

logger = logging.getLogger(__name__)


def is_gpu_available() -> bool:
    """
    Returns True if a GPU is available and USE_GPU env var allows it.
    Falls back to False if CUDA or CuPy isn't importable or
    explicitly disabled.
    """
    use_gpu_env = os.getenv("USE_GPU", "1") == "1"

    if not use_gpu_env:
        logger.info("USE_GPU=0 → Forcing CPU mode (CI or local override).")
        return False

    try:
        import cupy  # noqa: F401

        return True
    except Exception:
        return False


def get_device_info() -> str:
    """
    Returns human-readable GPU or CPU device info.
    """
    if not is_gpu_available():
        return "CPU"

    try:
        import cupy

        dev = cupy.cuda.runtime.getDevice()
        name = cupy.cuda.runtime.getDeviceProperties(dev)["name"]
        return f"GPU: {name}"
    except Exception as e:
        logger.warning(f"Unable to query GPU device info: {e}")
        return "GPU (unverified)"
