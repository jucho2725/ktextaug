def synonym_replace(corpus, prob, tokenizer, rng, **kwargs):
    if isinstance(corpus, str):
        corpus = tokenizer.tokenize(corpus)
    return corpus