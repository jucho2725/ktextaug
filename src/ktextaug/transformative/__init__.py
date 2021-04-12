from src.ktextaug.transformative.backtranslation import backtranslate
from src.ktextaug.transformative.noise_generation import noise_generate
from src.ktextaug.transformative.random_insertion import random_insert
from src.ktextaug.transformative.random_swap import random_swap

__all__ = [
    "backtranslate",
    "noise_generate",
    "random_insert",
    "random_delete",
    "random_swap",
    "synonym_replace",
]

def main():  # examples
    # single
    sent = "한국말 잘 몰라요."
    print(backtranslate(sent, target_language="ja"))

if __name__ == "__main__":
    main()
