"""
Author : JinUk, Cho
Last update : 10th, Apr, 2020
"""

import six
from googletrans import Translator

translate_client = Translator()
origin = None
result = None

def backtranslate(
    text: str, source_language: str = "ko", target_language: str = "en"
) -> str:
    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")

    back = translate_client.translate(text, dest=target_language)
    result = translate_client.translate(
        back.text, dest=source_language
    )
    return result.text


def main():  # examples
    sent = "한국말 잘 몰라요."
    print(backtranslate(sent, target_language="ja"))


if __name__ == "__main__":
    main()
