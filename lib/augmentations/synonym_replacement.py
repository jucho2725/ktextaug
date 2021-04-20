import json
import random


class SynDict(object):
    def __init__(self):
        self.dictionary = self.load_syn_dict()
        self.rng = random.Random()

    def load_syn_dict(self):
        with open('../../src/synonym_dictionary_kor.json', 'r') as r:
            content = json.load(r)
        return content

    def get_synonym(self, token, n_sample=1, rng=None):

        if not rng:
            rng = self.rng

        if token in self.dictionary:
            return rng.sample(self.dictionary[token], n_sample)

        return token



def synonym_replace(tokens, tokenizer, rng, n_syns, **kwargs):
    if isinstance(tokens, str):
        tokens = tokenizer.tokenize(tokens)

    new_tokens = [SYN_DICT.get_synonym(t) for t in tokens]
    diff_idxes = [i for i, (t, nt) in enumerate(zip(tokens, new_tokens)) if t != nt]
    diff_idxes = rng.sample(diff_idxes, min(n_syns, len(diff_idxes)))

    output_tokens = []
    for i in range(len(tokens)):
        if i in diff_idxes:
            output_tokens.append(new_tokens[i])
            continue
        output_tokens.append(tokens[i])

    return output_tokens


SYN_DICT = SynDict()
# Reusable object for the efficient