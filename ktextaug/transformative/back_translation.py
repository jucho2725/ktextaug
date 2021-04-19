import six
from googletrans import Translator
from typing import List, Union

translate_client = Translator()

def back_translate(text, target_language='en', **kwargs):
    assert isinstance(text, list) or isinstance(text, str), "Source input should be string, or list of string."
    if isinstance(text, list):
        return _backtrans_bulk(text, target_language)
    else:
        return _backtrans_single(text, target_language)
    return text

def _backtrans_single(text, target_language):
    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")
    source_language =  translate_client.detect(text).lang
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

