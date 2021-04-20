from .utils import isStopword, isWord, get_synonym, keep_punctuation

def random_delete(text_or_tokens, prob, tokenize_fn, rng, **kwargs):
    if isinstance(text_or_tokens, str):
        tokens = tokenize_fn(text_or_tokens)
        tokens, keep = keep_punctuation(tokens)
    else:
        tokens, keep = keep_punctuation(text_or_tokens)
    output_tokens, length = [], len(tokens)
    tokens.append('')
    for i in range(length):
        if rng.random() > prob:
            output_tokens.append(tokens[i])
        elif tokens[i + 1].startswith('##'):
            output_tokens.append(tokens[i])
    return " ".join(output_tokens) + keep

def random_swap(text_or_tokens, tokenize_fn, rng, n_swaps, **kwargs):
    if isinstance(text_or_tokens, str):
        tokens = tokenize_fn(text_or_tokens)
        tokens, keep = keep_punctuation(tokens)
    else:
        tokens, keep = keep_punctuation(text_or_tokens)

    for i in range(n_swaps):
        r1,r2 = sorted(rng.sample(range(len(tokens)), 2))
        tokens.insert(r2, tokens.pop(r1))
        tokens.insert(r1, tokens.pop(r2 - 1))
    return " ".join(tokens) + keep


def random_insert(text_or_words, n_inserts, tokenize_fn, rng, **kwargs):
    # check if there is a punctuation mark
    if isinstance(text_or_words, str):
        words = tokenize_fn(text_or_words)
        words, keep = keep_punctuation(words)
    else:
        words, keep = keep_punctuation(text_or_words)

    f_words = [w for w in words if (not isStopword(w)) and isWord(w)]
    target = rng.choices(f_words, k=n_inserts)
    for origin in target:
        new_syn = _get_word(origin)
        words.insert(rng.randrange(0, len(words)) - 1, new_syn)
    return " ".join(words) + keep

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
