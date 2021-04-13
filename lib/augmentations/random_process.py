PUNC = [".", ",", ":", ";", "?", "!"]


def random_delete(corpus, prob, tokenizer, rng, **kwargs):
    if isinstance(corpus, str):
        corpus = tokenizer.tokenize(corpus)
    if corpus[-1] in PUNC:
        keep = corpus[-1]
        corpus = corpus[:-1].copy()
    else:
        keep = None
        corpus = corpus.copy()

    if len(corpus) == 1:
        return corpus

    new_words = []

    for word in corpus:
        r = rng.uniform(0, 1)
        if r > prob:
            new_words.append(word)

    if len(new_words) == 0:
        rand_int = rng.randint(0, len(corpus) - 1)
        return " ".join([corpus[rand_int]])

    return new_words + [keep]


def random_swap(corpus, tokenizer, rng, n_swap, **kwargs):
    """
    :param words:
    :param n:
    :return:
    """
    # check if there is a punctuation mark
    if isinstance(corpus, str):
        corpus = tokenizer.tokenize(corpus)

    if corpus[-1] in PUNC:
        keep = corpus[-1]
        new_words = corpus[:-1].copy()
    else:
        keep = None
        new_words = corpus.copy()
    for _ in range(n_swap):
        new_words = _swap_word(new_words, rng)
    return new_words + [keep]


def _swap_word(new_words, rng):
    random_idx_1 = rng.randint(0, len(new_words) - 1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = rng.randint(0, len(new_words) - 1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = (
        new_words[random_idx_2],
        new_words[random_idx_1],
    )
    return new_words
