"""
I/O modules for pycocos.
"""

from . import eqdsk
from . import cocos
from .eqdsk import eqdsk as EQDSK
from .cocos import (
    COCOS,
    COCOSResolution,
    cocos,
    identify_cocos,
    transform_cocos,
    fromCocosNtoCocosM,
)

__all__ = [
    "eqdsk",
    "EQDSK",
    "COCOS",
    "COCOSResolution",
    "cocos",
    "identify_cocos",
    "transform_cocos",
    "fromCocosNtoCocosM",
]
