from types import ModuleType
from typing import Any
import six

from pathlib import Path

class _BaseLazyModule(ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    # Very heavily inspired by optuna.integration._IntegrationModule
    # https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
    def __init__(self, name, import_structure):
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # Needed for autocompletion in an IDE
        self.__all__ = list(import_structure.keys()) + sum(import_structure.values(), [])

    # Needed for autocompletion in an IDE
    def __dir__(self):
        return super().__dir__() + self.__all__

    def __getattr__(self, name: str) -> Any:
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str) -> ModuleType:
        raise NotImplementedError

def open_text(fn, enc="utf-8"):
    with open(fn, "r", encoding=enc) as f:
        return [l.strip() for l in f.readlines()]

def save_texts(fname, texts):
    texts = [text.decode("utf-8") if isinstance(text, six.binary_type) else text for text in texts]

    with open(fname, "w") as f:
        for text in texts:
            f.write(f"{text}\n")

def make_dir_structure(path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

def checkout():
    import os
    print("relative path", os.path.exists("transformative/stopwords-ko.txt"))
    print("absolute path", os.path.exists("ktextaug/stopwords-ko.txt"))
