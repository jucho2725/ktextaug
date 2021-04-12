from typing import TYPE_CHECKING
import importlib
import os
import sys

from file_utils import _BaseLazyModule

__version__ = "0.1.5"

class _LazyModule(_BaseLazyModule):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """
    # this is strongly inspired by transformers package, transformers/src/transformers/__init__.py

    __file__ = globals()["__file__"]
    __path__ = [os.path.dirname(__file__)]

    def _get_module(self, module_name):
        return importlib.import_module("." + module_name, self.__name__)

    def __getattr__(self, name):
        # Special handling for the version, which is a constant from this module and not imported in a submodule.
        if name == "__version__":
            return __version__
        return super().__getattr__(name)



# Base objects, independent of any specific backend
_import_structure = {
    # Models
    "transformative": [
            "backtranslate",
            "noise_generate",
            "random_insert",
            "random_delete",
            "random_swap",
            "synonym_replace",
    ],
    "generative": [],
}

# Direct imports for type-checking
if TYPE_CHECKING:
    from .transformative import (
        backtranslate,
        random_delete,
        random_insert,
        random_swap,
        synonym_replace,
        noise_generate,
    )
    from .generative import *
else:
    sys.modules[__name__] = _LazyModule(__name__, _import_structure)
