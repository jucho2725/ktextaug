from .utils import isStopword, isWord, get_synonym, keep_punctuation

def random_delete(text_or_words, prob, tokenize_fn, rng, **kwargs):
    if isinstance(text_or_words, str):
        words = tokenize_fn(text_or_words)
        new_words, keep = keep_punctuation(words)
    else:
        new_words, keep = keep_punctuation(text_or_words)

    if len(new_words) == 1:
        return new_words + keep

    for _ in range(int(len(new_words) * prob)):
        new_words.pop(rng.randint(0, len(words)))
    return " ".join(new_words + [keep])


def random_swap(text_or_words, tokenize_fn, rng, n_swap, **kwargs):
    if isinstance(text_or_words, str):
        words = tokenize_fn(text_or_words)
        new_words, keep = keep_punctuation(words)
    else:
        new_words, keep = keep_punctuation(text_or_words)

    for _ in range(n_swap):
        new_words = _swap_word(new_words, rng)
    return " ".join(new_words + [keep])

def _swap_word(new_words, rng):
    random_idx_1 = rng.randint(0, len(new_words))
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = rng.randint(0, len(new_words))
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = (
        new_words[random_idx_2],
        new_words[random_idx_1],
    )
    return new_words


def random_insert(text_or_words, n, tokenize_fn, rng):
    # check if there is a punctuation mark
    if isinstance(text_or_words, str):
        words = tokenize_fn(text_or_words)
        words, keep = keep_punctuation(words)
    else:
        words, keep = keep_punctuation(text_or_words)

    f_words = [w for w in words if (not isStopword(w)) and isWord(w)]
    target = rng.choices(f_words, k=n)
    for origin in target:
        new_syn = _get_word(origin)
        words.insert(rng.randrange(0, len(words)) - 1, new_syn)
    return " ".join(words + [keep])

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
