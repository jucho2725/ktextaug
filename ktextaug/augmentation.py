import random

from ktextaug.transformative.back_translation import back_translate
from .transformative.noise_addition import noise_add
from .transformative.random_process import random_delete, random_swap, random_insert
from .transformative.synonym_replacement import synonym_replace
from .tokenization_utils import get_tokenize_fn

from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm.auto import tqdm

# ['random_deletion', 'random_swap', 'synonym_replacement', 'noise_add', 'back-translation']
# TODO: tokenization wrapping for 'mecab', 'kkma' etc.

class TextAugmentation(object):
    def __init__(self, tokenize_fn=None, num_processes=1,):
        self.augmentions = {
            'random_delete': random_delete,
            'random_swap': random_swap,
            'synonym_replace': synonym_replace,
            'noise_add': noise_add,
            'back_translate':back_translate,
            'random_insert': random_insert,
        }
        if tokenize_fn is None:
            self.tokenize_fn = lambda x: x.split(" ")
        elif isinstance(tokenize_fn, str):
            self.tokenize_fn = get_tokenize_fn(tokenizer_name=tokenize_fn)
        self.num_process=num_processes if num_processes != -1 else int(cpu_count() / 2)

    def generate(self, text_or_corpus, mode='back_translate', rng=None,
                 prob=0.1, n_swap=1, n_rep=1, noise_mode=['jamo_split', 'vowel_change', 'phonological_change'], target_language='en',):

        rng = random.Random() if rng is None else rng
        if mode == "back_translate":
            text_or_corpus = self.augmentions[mode](text_or_corpus, target_language=target_language)

        if isinstance(text_or_corpus, list):
            pool = Pool(processes=self.num_process)
            func = partial(self.augmentions[mode], prob=prob, tokenize_fn=self.tokenize_fn, rng=rng,
                           n_swap=n_swap, n_rep=n_rep, noise_mode=noise_mode, target_language=target_language)

            return [r for r in tqdm(pool.imap(func=func, iterable=text_or_corpus), total=len(text_or_corpus))]

        elif isinstance(text_or_corpus, str):
            text = self.augmentions[mode](text_or_corpus, prob=prob, tokenize_fn=self.tokenize_fn, rng=rng,
                                                    n_swap=n_swap, n_rep=n_rep, noise_mode=noise_mode, target_language=target_language)
            return text


if __name__ == '__main__':
    sample_text = '달리는 기차 위에 중립은 없다. 미국의 사회 운동가이자 역사학자인 하워드 진이 남긴 격언이다.'
    agent = TextAugmentation(tokenize_fn="mecab")
    print(agent.generate(sample_text))