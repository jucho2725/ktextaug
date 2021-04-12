"""
Author : JungHoon, Lee
Editor : Jin Uk, Cho
Last update : 12th, Apr, 2020
"""

import random
from ..tokenization_utils import Tokenizer
from .utils import isStopword, isWord, get_synonym


def random_insert(words, n):
    f_words = [w for w in words if (not isStopword(w)) and isWord(w)]
    target = random.choices(f_words, k=n)
    for origin in target:
        new_syn = _get_word(origin)
        words.insert(random.randrange(0, len(words)) - 1, new_syn)
    return words

def _get_word(target):
    Flag = True
    counter = 0
    while Flag:
        new_syn = get_synonym(target)
        counter += 1
        if target == new_syn and counter < 30: # TO DO : dealing with a word which has tiny set of synonyms.
            pass
        elif counter >= 30:
            new_syn = target # TO DO : error cause by bs4
            Flag = False
        else:
            Flag = False
    return new_syn


if __name__ == "__main__":
    tokenizer = Tokenizer(tokenizer_or_name="komoran")
    Sample = "철수가 밥을 빨리 먹었다."
    print("Sample : ", Sample)
    print(tokenizer.tokenize(Sample))
    print(random_insert(tokenizer.tokenize(Sample), 2))
