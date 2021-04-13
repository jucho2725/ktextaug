from transformers import AutoTokenizer

VOCABULARY = './basic_vocabulary.txt'

class TextAugmentation(object):
    def __init__(self):
        pass

    def generate(self, sentence, tokenizer=None, mode=[]):
        if tokenizer is None:
            tokenizer = AutoTokenizer(VOCABULARY)
