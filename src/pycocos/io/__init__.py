"""
I/O modules for pycocos.
"""

from . import eqdsk
from . import cocos
from .eqdsk import eqdsk as EQDSK
from .cocos import (
    COCOS,
    cocos,
    assign,
    transform_cocos,
    fromCocosNtoCocosM,
)

__all__ = [
    "eqdsk",
    "cocos",
    "EQDSK",
    "COCOS",
    "cocos",
    "assign",
    "transform_cocos",
    "fromCocosNtoCocosM",
]

