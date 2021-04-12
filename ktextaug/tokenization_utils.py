from konlpy.tag import Mecab
from PyKomoran import Komoran
from types import ModuleType

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
            self.tokenizer_name = tokenizer_or_name

    def tokenize(self, text):
        if self.tokenizer_name == "komoran":
            self.tokenizer.get_morphes_by_tags(text)
        elif self.tokenizer_name == "mecab":
            self.tokenizer.morphs(text)
        else: # self.tokenizer_name 이 None
            self.tokenizer.tokenize(text)
