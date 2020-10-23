
import six



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
