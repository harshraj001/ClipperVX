"""ClipperVX - YouTube to Shorts/Reels Automation Tool"""

__version__ = "1.0.0"
__author__ = "ClipperVX"

from .config import Config
from .orchestrator import ClipperOrchestrator

__all__ = ["Config", "ClipperOrchestrator", "__version__"]
