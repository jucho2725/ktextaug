"""
Author : JinUk, Cho
Last update : 10th, Apr, 2020
"""

import six
from googletrans import Translator

from typing import List, Union

translate_client = Translator()
origin = None
result = None

def backtranslate(
    source: Union[str, List[str]],
    target_language: str = "en"
) -> str:
    assert isinstance(source, list) or isinstance(source, str), "Source input should be string, or list of string."
    if isinstance(source, list):
        return _backtrans_bulk(source, target_language)
    else:
        return _backtrans_single(source, target_language)


def _backtrans_single(text, target_language):
    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")
    source_language =  translate_client.detect(text).lang
    print(source_language, target_language)
    assert source_language != target_language, "Source language is deteted as same as target. You should select different target language for backtranslation."

    back = translate_client.translate(text, dest=target_language)
    result = translate_client.translate(
        back.text, dest=source_language
    )
    return result.text

def _backtrans_bulk(sents, target_language):
    text = sents[0]
    if isinstance(text, six.binary_type):
        sents = [t.decode("utf-8") for t in sents]
    source_language = translate_client.detect(text).lang
    assert source_language != target_language, "Source language is deteted as same as target. You should select different target language for backtranslation."

    back = translate_client.translate(sents, dest=target_language)
    back_sents = [b.text for b in back]
    result = translate_client.translate(
        back_sents, dest=source_language
    )
    return [r.text for r in result]

def main():  # examples
    # single
    sent = "한국말 잘 몰라요."
    print(backtranslate(sent, target_language="ja"))

    # bulk
    sents = ["한국말 잘 몰라요.", "나는 밥을 먹었다."]
    print(backtranslate(sents, target_language="ja"))

    # error
    sent = 12
    print(backtranslate(sent, target_language="ja"))

if __name__ == "__main__":
    main()
