import ctypes
import sys
from pathlib import Path


# noinspection protected-member
def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        base_path = Path(sys._MEIPASS)
    else:
        # Resolve source assets relative to this module so launching the script
        # from another working directory does not break resource loading.
        base_path = Path(__file__).resolve().parent

    return base_path / relative_path


def get_hwnd(title: str) -> int | None:
    """Returns the HWND for an exact window title match."""
    hwnd = ctypes.windll.user32.FindWindowW(None, title)
    return hwnd or None
