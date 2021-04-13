import random
from .utils import isStopword, isWord, get_synonym


def synonym_replace(corpus, prob, tokenizer, rng, **kwargs):
    if isinstance(corpus, str):
        corpus = tokenizer.tokenize(corpus)


def synonym_replace(corpus, prob, tokenizer, rng, n_rep, **kwargs):
    # check if there is a punctuation mark
    if isinstance(corpus, str):
        corpus = tokenizer.tokenize(corpus)
    punctuations = [".", ",", ":", ";", "?", "!"]
    if corpus[-1] in punctuations:
        keep = corpus[-1]
        words = corpus[:-1].copy()
    else:
        keep = None
        words = corpus.copy()

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
        if num_replacement >= n_rep:
            break

    sentence = " ".join(new_word)
    result = sentence.split(" ")
    return result + [keep]