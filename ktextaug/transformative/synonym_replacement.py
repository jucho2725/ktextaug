from .utils import isStopword, isWord, get_synonym, keep_punctuation

def synonym_replace(text_or_words, tokenizer, rng, n_syns, **kwargs):
    # check if there is a punctuation mark
    if isinstance(text_or_words, str):
        words = tokenizer.tokenize(text_or_words)
        new_words, keep = keep_punctuation(words)
    else:
        new_words, keep = keep_punctuation(text_or_words)

    result = new_words[:]
    nonStop = [w for w in result if (not isStopword(w)) and isWord(w)]
    rng.shuffle(nonStop)
    num_replacement = 0
    for random_word in nonStop:
        synonym = get_synonym(random_word, rng)
        if len(synonym) >= 1:
            synonym = rng.choice(list(synonym))
            new_words = [synonym if word == random_word else word for word in result]
            num_replacement += 1
        if num_replacement >= n_syns:
            break
    return tokenizer.post_process(new_words) + keep

