from types import ModuleType
import os
try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

 # __package__ 는 current module 의미


def get_tokenize_fn(tokenizer_name="mecab", vocab_path=None):
    if tokenizer_name.lower() == "komoran":
        from PyKomoran import Komoran
        tokenizer = Komoran("STABLE")
        return tokenizer.get_morphes_by_tags
    elif tokenizer_name.lower() == "mecab":
        from konlpy.tag import Mecab
        tokenizer = Mecab()
        return tokenizer.morphs
    elif tokenizer_name.lower() == "subword":
        from transformers import BertTokenizer
        if vocab_path is not None:
            tokenizer = BertTokenizer(os.path.join(__package__, vocab_path), do_lower_case=False)
            return tokenizer.tokenize
        else:
            VOCABULARY = './vocab_noised.txt'
            tokenizer = BertTokenizer(os.path.join(__package__, VOCABULARY), do_lower_case=False)
            return tokenizer.tokenize
