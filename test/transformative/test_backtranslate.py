from ktextaug.transformative import backtranslate

sent = "한국말 잘 몰라요."
print(type(backtranslate))
print(backtranslate(sent, target_language="ja"))
