"""
Author : MinSu, Jeong
Last update : 20th, Nov, 2020
"""

import random
from .utils import tokenize


def random_deletion(words, p):
    if len(words) == 1:
        return words

    new_words = []

    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return " ".join([words[rand_int]])

    return new_words


if __name__ == "__main__":
    sample = "철수가 밥을 빨리 먹었다."
    print("Sample : ", sample)
    print("Tokenize")
    print(tokenize(sample))
    print("Random_Deletion")
    print(random_deletion(sample, 0.3))
