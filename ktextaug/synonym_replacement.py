import random
from .utils import isStopword, isWord, get_synonym


def synonym_replacement(words, n):
    result = words[:]
    nonStop = [w for w in result if (not isStopword(w)) and isWord(w)]
    random.shuffle(nonStop)
    num_replacement = 0
    for random_word in nonStop:
        synonym = get_synonym(random_word)
        if len(synonym) >= 1:
            synonym = random.choice(list(synonym))
            new_word = [synonym if word == random_word else word for word in result]
            num_replacement += 1
        if num_replacement >= n:
            break

    sentence = " ".join(new_word)
    result = sentence.split(" ")
    return result
