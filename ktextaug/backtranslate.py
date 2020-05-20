from googletrans import Translator
from ktextaug.file_utils import open_text, make_dir_structure, save_texts
from pathlib import Path

# from tqdm import tqdm
# from typing import

class BackTranslate:
    def __init__(self, translator: object):
        self.translator = translator

    def get_translator(self):
        return

    def backtranslate(self, text: str, source_language='en', target_language='ja') -> str:
        mid = self.translator.translate(text, dest=target_language)
        result = self.translator.translate(mid.text, dest=source_language)
        return result.text

    def saver(self, pth: str, target_language: str) -> None:
        assert isinstance(pth, str)
        text = open_text(pth)
        save_path = pth.replace('sample', f'sample_{target_language}')
        make_dir_structure(save_path)
        if Path(save_path).exists():
            return
        back = self.backtranslate(text, target_language=target_language)
        save_texts(save_path, [back])

def main(): # test
    translator = Translator()
    bt_model = BackTranslate(translator)
    path = "../src/data/sample.txt"
    bt_model.saver(path, target_language="ja")

if __name__ == '__main__':
    main()
