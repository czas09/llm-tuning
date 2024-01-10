import importlib.metadata
import importlib.util


def is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def get_package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except:
        return "0.0.0"
    

_flash_attn2_available = is_package_available("flash_attn") and get_package_version("flash_attn").startswith("2")


def is_flash_attn2_available():
    return _flash_attn2_available