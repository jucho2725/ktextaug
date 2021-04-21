from pkg_resources import resource_filename
from types import ModuleType
#from PyKomoran import Komoran
#from konlpy.tag import Mecab
from transformers import BertTokenizer
from functools import partial
from transformers import *


import os
# try:
#     import importlib.resources as pkg_resources
# except ImportError:
#     # Try backported to PY<37 `importlib_resources`.
#     import importlib_resources as pkg_resources

class Tokenizer:
    def __init__(self, tokenizer_name="komoran"):
        assert (tokenizer_name.lower() == "komoran") or (tokenizer_name.lower() == "mecab")\
            or (tokenizer_name.lower() == "subword"), "Only 'komoran', 'mecab', and 'subword' is acceptable."
        if tokenizer_name == "komoran":
            self.tokenizer = Komoran("STABLE")
        elif tokenizer_name == "mecab":
            self.tokenizer = Mecab()
        elif tokenizer_name == "subword":
            self.tokenizer = BertTokenizer(resource_filename(__package__, "vocab_noised.txt"), do_lower_case=False)
        self.tokenizer_name = tokenizer_name

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


def get_tokenizer(keyword):
    return 'test'


def get_tokenize():
    return 'test'


def get_convert_tokens_to_string():
    return 'test'


def wrapping_tokenizer(self, tokenizer=None):
    if isinstance(tokenizer, str):
        tokenizer = get_tokenizer(tokenizer)

    functions = dir(tokenizer)

    if 'tokenize' not in functions:
        tokenizer.tokenize = partial(get_tokenize(tokenizer), tokenizer)

    if 'convert_tokens_to_string' not in functions:
        tokenizer.convert_tokens_to_string = partial(get_convert_tokens_to_string(tokenizer), tokenizer)

    return tokenizer

if __name__ =='__main__':
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    print(dir(tokenizer))
    tokenizer = TestTokenizer(tokenizer)
    print(dir(tokenizer))