# from googletrans import Translator
# translator = Translator()
# result = translator.translate('안녕하세요.', dest="ja")
# print(result.text)
# print(translator.detect('이 문장은 한글로 쓰여졌습니다.'))
#

from google.cloud import translate_v3 as translate

client = translate.TranslationServiceClient()

PROJECT_ID = "backtranslate-277516"
LOCATION = "us-central1"

response = client.translate_text(
    parent=client.location_path(PROJECT_ID, LOCATION),
    contents=["Let's try the Translation API on Google Cloud Platform."],
    mime_type="text/plain",  # mime types: text/plain, text/html
    source_language_code="en-US",  # This field is optional
    target_language_code="es",
)

# Print the translation
for translation in response.translations:
    print(u"Translated text: {}".format(translation.translated_text))
