from PyKomoran import *

komoran = Komoran("EXP")
print(komoran.get_morphes_by_tags("나는 학교에서 밥을 먹었다."))
print(komoran.get_plain_text("샤인웨어에서 단체로 캡틴 마블을 관람했습니다."))

from ktextaug.file_utils import checkout
checkout()

from ktextaug.tokenization_utils import Tokenizer
tokenizer = Tokenizer(tokenizer_or_name="komoran")

from ktextaug.transformative import backtranslate

sent = "한국말 잘 몰라요."
print(type(backtranslate))
print(backtranslate(sent, target_language="ja"))

from ktextaug.file_utils import checkout
checkout()

print(__package__)
""" checkout import """
import ktextaug
print(type(ktextaug))
print(type(ktextaug.transformative))
print(type(ktextaug.backtranslate))
print(type(ktextaug.file_utils))
print(type(ktextaug.open_text))
print(type(ktextaug.transformative.utils))
print(type(ktextaug.utils.define_stopwords))
# print(type(ktextaug.define_stopwords))

""" update stopwords list """
print(len(ktextaug.utils.stopwords))
a = [1, 2, 3]
ktextaug.utils.define_stopwords(a)
print(len(ktextaug.utils.stopwords))
print(ktextaug.utils.isStopword(1))
