"""Compatibility shim so the repo root can be dropped into ComfyUI/custom_nodes."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_PKG_DIR = Path(__file__).resolve().parent / "custom_nodes" / "Comfyui-SynthidBypass"
_PKG_INIT = _PKG_DIR / "__init__.py"
_PKG_NAME = "comfyui_synthid_bypass_pkg"

_SPEC = importlib.util.spec_from_file_location(
    _PKG_NAME,
    _PKG_INIT,
    submodule_search_locations=[str(_PKG_DIR)],
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to load Synthid-Bypass custom nodes from {_PKG_INIT}")

_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_PKG_NAME] = _MODULE
_SPEC.loader.exec_module(_MODULE)

NODE_CLASS_MAPPINGS = _MODULE.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = _MODULE.NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
