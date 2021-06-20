import random

from ktextaug.transformative.back_translation import back_translate
from .transformative.noise_addition import noise_add
from .transformative.random_process import random_delete, random_swap, random_insert
from .transformative.synonym_replacement import synonym_replace
from .tokenization_utils import Tokenizer

from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm.auto import tqdm

from dataclasses import dataclass, field

# ['random_deletion', 'random_swap', 'synonym_replacement', 'noise_add', 'back-translation']
# TODO: tokenization wrapping for 'mecab', 'kkma' etc.

@dataclass
class KtextArguments:
    prob: float = field(default=0.1, metadata=)
    n_swaps: int = field(default=0.1, metadata=)
    n_inserts: int = field(default=0.1, metadata=)
    n_syns: int = field(default=0.1, metadata=)

    noise_mode: list = field(default=['jamo_split', 'vowel_change', 'phonological_change'],
                             metadata=)

    # back translation
    target_language: str = field(default="en", metadata=)


class TextAugmentation(object):
    def __init__(self, tokenizer=None, num_processes=1, ):
        self.augmentions = {
            'random_delete': random_delete,
            'random_swap': random_swap,
            'synonym_replace': synonym_replace,
            'noise_add': noise_add,
            'back_translate':back_translate,
            'random_insert': random_insert,
        }
        self.kwargs = KtextArguments()

        if tokenizer is None:
            self.tokenizer = Tokenizer(tokenizer_name="subword") # default
        elif isinstance(tokenizer, str):
            self.tokenizer = Tokenizer(tokenizer_name=tokenizer) # default
        else:
            self.tokenizer = tokenizer

        self.num_process=num_processes if num_processes != -1 else int(cpu_count() / 2)

    def generate(self, text_or_corpus,
                 mode='back_translate',rng=None,
                 **kwargs):
        for key, value in kwargs.items():
            self.kwargs[key] = value

        rng = random.Random() if rng is None else rng
        if mode == "back_translate":
            text_or_corpus = self.augmentions[mode](text_or_corpus, target_language=target_language)

        if isinstance(text_or_corpus, list):
            pool = Pool(processes=self.num_process)
            func = partial(self.augmentions[mode], tokenizer=self.tokenizer, rng=rng,
                           **self.kwargs)

            return [r for r in tqdm(pool.imap(func=func, iterable=text_or_corpus), total=len(text_or_corpus))]

        elif isinstance(text_or_corpus, str):
            text = self.augmentions[mode](text_or_corpus, tokenizer=self.tokenizer, rng=rng,
                                          **self.kwargs)
            return text


if __name__ == '__main__':
    sample_text = '달리는 기차 위에 중립은 없다. 미국의 사회 운동가이자 역사학자인 하워드 진이 남긴 격언이다.'
    agent = TextAugmentation(tokenizer="mecab")
    print(agent.generate(sample_text))