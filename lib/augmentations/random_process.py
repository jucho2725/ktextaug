import string
from itertools import compress
from synonym_replacement import SYN_DICT


def random_delete(tokens, prob, tokenizer, rng, **kwargs):
    if isinstance(tokens, str):
        tokens = tokenizer.tokenize(tokens)

    output_tokens, length = [], len(tokens)
    tokens.append('')
    for i in range(length):
        if rng.random() > prob or tokens[i] in string.punctuation:
            output_tokens.append(tokens[i])
        elif tokens[i+1].startswith('##'):
            output_tokens.append(tokens[i])
    return output_tokens


def random_insert(tokens, tokenizer, rng, n_insert, **kwargs):
    if isinstance(tokens, str):
        output_tokens = tokenizer.tokenize(tokens)

    random_idxes = rng.sample(range(len(output_tokens)), n_insert)
    syn_words = [SYN_DICT.get_synonym(token)[0] for token in output_tokens]
    syn_words = [s for s,t in zip(syn_words, output_tokens) if s!=t]
    random_idxes = random_idxes[:len(syn_words)]

    for idx, syn_word in zip(random_idxes, syn_words):
        output_tokens.insert(idx, syn_word)

    return output_tokens


def random_swap(tokens, tokenizer, rng, n_swaps, **kwargs):
    if isinstance(tokens, str):
        tokens = tokenizer.tokenize(tokens)

    for i in range(n_swaps):
        r1,r2 = sorted(rng.sample(range(len(tokens)), 2))
        tokens.insert(r2, tokens.pop(r1))
        tokens.insert(r1, tokens.pop(r2-1))
    return tokens



if __name__ == '__main__':
    print(SYN_DICT)
