from pkg_resources import resource_filename
from types import ModuleType
from PyKomoran import Komoran
from konlpy.tag import Mecab
from transformers import BertTokenizer

import os
# try:
#     import importlib.resources as pkg_resources
# except ImportError:
#     # Try backported to PY<37 `importlib_resources`.
#     import importlib_resources as pkg_resources

class Tokenizer:
    def __init__(self, tokenizer_or_name="komoran"):
        if isinstance(tokenizer_or_name, ModuleType):
            self.tokenizer = tokenizer_or_name
            self.tokenizer_name = None
        else:
            print(tokenizer_or_name)
            assert tokenizer_or_name.lower() == "komoran" or tokenizer_or_name.lower() == "mecab", "Only 'komoran' and 'mecab' is acceptable."
            if tokenizer_or_name == "komoran":
                self.tokenizer = Komoran("STABLE")
            elif tokenizer_or_name == "mecab":
                self.tokenizer = Mecab()
            elif tokenizer_or_name == "subword":
                tokenizer = BertTokenizer(resource_filename(__package__, "vocab_noised.txt"), do_lower_case=False)
            self.tokenizer_name = tokenizer_or_name

    def tokenize(self, text):
        if self.tokenizer_name == "komoran":
            return self.tokenizer.get_morphes_by_tags(text)
        elif self.tokenizer_name == "mecab":
            return self.tokenizer.morphs(text)
        else: # self.tokenizer_name 이 None
            return self.tokenizer.tokenize(text)

    def post_process(self, tokens):
        if self.tokenizer_name == "komoran":
            return " ".join(tokens)
        elif self.tokenizer_name == "mecab":
            return " ".join(tokens)
        else: # self.tokenizer_name 이 subword 또는 moduletype
            return self.tokenizer.convert_tokens_to_string(tokens)