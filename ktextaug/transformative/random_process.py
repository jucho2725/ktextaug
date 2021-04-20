import string

from .utils import isStopword, isWord, get_synonym

def random_delete(text_or_tokens, prob, tokenizer, rng, **kwargs):
    if isinstance(text_or_tokens, str):
        tokens = tokenizer.tokenize(text_or_tokens)

    output_tokens, length = [], len(tokens)
    tokens.append('')
    for i in range(length):
        if rng.random() > prob and tokens[i] not in string.punctuation:
            output_tokens.append(tokens[i])
        elif tokens[i + 1].startswith('##'):
            output_tokens.append(tokens[i])
    return tokenizer.post_process(output_tokens)

def random_swap(text_or_tokens, tokenizer, rng, n_swaps, **kwargs):
    if isinstance(text_or_tokens, str):
        tokens = tokenizer.tokenize(text_or_tokens)

    for i in range(n_swaps):
        while True:
            r1,r2 = sorted(rng.sample(range(len(tokens)), 2))
            if tokens[r1] not in string.punctuation and tokens[r2] not in string.punctuation:
                break
        tokens.insert(r2, tokens.pop(r1))
        tokens.insert(r1, tokens.pop(r2 - 1))
    return tokenizer.post_process(tokens)


def random_insert(text_or_tokens, n_inserts, tokenizer, rng, **kwargs):
    # check if there is a punctuation mark
    if isinstance(text_or_tokens, str):
        words = tokenizer.tokenize(text_or_tokens)

    f_words = [w for w in words if (not isStopword(w)) and (w not in string.punctuation) and isWord(w)]
    target = rng.choices(f_words, k=n_inserts)
    for origin in target:
        new_syn = _get_word(origin)
        words.insert(rng.randrange(0, len(words)) - 1, new_syn)
    return tokenizer.post_process(words)

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
