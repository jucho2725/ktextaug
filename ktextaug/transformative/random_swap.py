"""
Author : JeongHyeok, Park
Last update : 20th, Nov, 2020
"""
# Random SWAP
import random
from .utils import tokenize


def random_swap(words, n):  # N-times
    new_words = words.copy()
    for _ in range(n):
        new_words = _swap_word(new_words)
    return new_words


def _swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words) - 1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = (
        new_words[random_idx_2],
        new_words[random_idx_1],
    )
    return new_words


def main():
    ex1 = "철수가 밥을 빨리 먹었다."
    tok_words = tokenize(ex1)
    print("tokenized", tok_words)
    alpha = 0.2
    swap_words = random_swap(tok_words, int(alpha * len(tok_words)))
    print("swapped", swap_words)

if __name__ == "__main__":
    main()
