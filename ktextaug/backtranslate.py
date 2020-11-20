"""
Author : JinUk, Cho
Last update : 20th, Nov, 2020
"""

import six

class BackTranslate:
    def __init__(self):
        from googletrans import Translator

        self.translate_client = Translator()
        self.origin = None
        self.result = None

    def get_translator(self):
        return self.translate_client

    def backtranslate(
        self, text: str, source_language: str = "ko", target_language: str = "en"
    ) -> str:
        if isinstance(text, six.binary_type):
            text = text.decode("utf-8")

        self.origin = text
        back = self.translate_client.translate(text, dest=target_language)
        result = self.translate_client.translate(
            back.text, dest=source_language
        )
        self.result = result
        return result.text


def main():  # examples
    bt_model = BackTranslate()
    sent = "한국말 잘 몰라요."
    print(bt_model.backtranslate(sent, target_language="ja"))


if __name__ == "__main__":
    main()
