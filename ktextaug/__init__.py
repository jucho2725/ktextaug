from typing import TYPE_CHECKING
import importlib
import os
import sys

from .file_utils import (
    _BaseLazyModule,
    checkout
)

# NOTE:
# this is strongly inspired by transformers package, transformers/src/transformers/__init__.py

__version__ = "0.1.9"

class _LazyModule(_BaseLazyModule):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """
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
    # Tokenizer
    "tokenization_utils": [
        "get_tokenize_fn",
    ],
    # File utils
    "file_utils": [
        "open_text",
        "save_texts",
        "make_dir_structure",
        "_BaseLazyModule"
    ],

    # Models
    "transformative": [
            "back_translate",
            "noise_add",
            "random_insert",
            "random_delete",
            "random_swap",
            "synonym_replace",
            "utils",
    ],
    "generative": [],
    "augmentation": ["TextAugmentation"],
}

# Direct imports for type-checking

if TYPE_CHECKING:
    from .tokenization_utils import get_tokenize_fn
    from .transformative import (
        back_translate,
        random_delete,
        random_insert,
        random_swap,
        synonym_replace,
        noise_add,
    )
    from .generative import *
    from .file_utils import (
        open_text,
        save_texts,
        make_dir_structure
    )
    from augmentation import TextAugmentation
else:
    sys.modules[__name__] = _LazyModule(__name__, _import_structure)
