import random
from transformers import BertTokenizer
from lib.augmentations.random_process import random_delete, random_swap
from lib.augmentations.synonym_replacement import synonym_replace
from lib.augmentations.noise_addition import noise_add
from lib.augmentations.back_translation import back_translate
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm.auto import tqdm

VOCABULARY = 'tokenization/vocabulary/basic_vocabulary.txt'

# ['random_deletion', 'random_swap', 'synonym_replacement', 'noise_add', 'back-translation']
# TODO: tokenization wrapping for 'mecab', 'kkma' etc.



class TextAugmentation(object):
    def __init__(self):
        self.augmentions = {
            'random_deletion': random_delete,
            'random_swap': random_swap,
            'synonym_replacement': synonym_replace,
            'noise_add': noise_add,
            'back-translation':back_translate,
        }

    def generate(self, corpus, prob=0.1, tokenizer=None, do_lower_case=False, mode='random_deletion', rng=None,
                 n_swap=1, n_rep=1, noise_mode=['jamo_split', 'vowel_change', 'phonological_change'], target_language='en', num_processes=1):
        if rng is None:
            rng = random.Random()

        if tokenizer is None:
            tokenizer = BertTokenizer(VOCABULARY, do_lower_case=do_lower_case)

        if isinstance(corpus, list):
            pool = Pool(processes=num_processes)
            func = partial(self.augmentions[mode], prob=prob, tokenizer=tokenizer, rng=rng,
                           n_swap=n_swap, n_rep=n_rep, noise_mode=noise_mode, target_language=target_language)

            return [r for r in tqdm(pool.imap(func=func, iterable=corpus), total=len(corpus))]

        elif isinstance(corpus, str):
            corpus = self.augmentions[mode](corpus, prob=prob, tokenizer=tokenizer, rng=rng,
                                        n_swap=n_swap, n_rep=n_rep, noise_mode=noise_mode, target_language=target_language)
            return corpus


if __name__ == '__main__':
    sample_text = '행복한 가정은 모두가 닮았지만, 불행한 가정은 모두 저마다의 이유로 불행하다.'
    agent = TextAugmentation()
    print(agent.generate(sample_text))