# from googletrans import Translator
# from ktextaug.file_utils import open_text, make_dir_structure, save_texts
# from pathlib import Path

import six
import pandas as pd
from tqdm import tqdm

# from tqdm import tqdm
# from typing import

# class BackTranslate:
#     def __init__(self, translator: object):
#         self.translator = translator
#
#     def get_translator(self):
#         return
#
#     def backtranslate(self, text: str, source_language='en', target_language='ja') -> str:
#         mid = self.translator.translate(text, dest=target_language)
#         result = self.translator.translate(mid.text, dest=source_language)
#         return result.text
#
#     def saver(self, pth: str, target_language: str) -> None:
#         assert isinstance(pth, str)
#         text = open_text(pth)
#         save_path = pth.replace('sample', f'sample_{target_language}')
#         make_dir_structure(save_path)
#         if Path(save_path).exists():
#             return
#         back = self.backtranslate(text, target_language=target_language)
#         save_texts(save_path, [back])


class BackTranslate:
    def __init__(self):
        from google.cloud import translate_v2 as translate
        self.translate_client = translate.Client()
        self.Result = None
        
    def get_translator(self):
        return self.translate_client

    def backtranslate(self, text: str, source_language='ko', target_language='en') -> str:
        if isinstance(text, six.binary_type):
            text = text.decode('utf-8')

        back = self.translate_client.translate(text, target_language=target_language)
        result = self.translate_client.translate(back['translatedText'], target_language=source_language)
        self.Result = result
        return result['translatedText']
'''
def main(): # test
#     translator = Translator()
#     bt_model = BackTranslate(translator)
#     path = "../src/data/sample.txt"
#     bt_model.saver(path, target_language="ja")
   # print('test')


if __name__ == '__main__':
    #main()
'''
