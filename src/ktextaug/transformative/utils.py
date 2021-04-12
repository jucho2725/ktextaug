from pathlib import Path
from konlpy.tag import Mecab
from nltk.corpus import stopwords
import requests
from bs4 import BeautifulSoup
import random


def open_text(fn, enc="utf-8"):
    with open(fn, "r", encoding=enc) as f:
        return "".join(f.readlines())


def save_texts(fname, texts):
    with open(fname, "w") as f:
        for t in texts:
            f.write(f"{t}\n")


def make_dir_structure(path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)


def tokenize(text):
    mecab = Mecab()
    return mecab.morphs(text)


def isWord(word):
    return word.isalnum()


def isStopword(word):
    stopword = stopwords.words("korean")
    if word in stopword:
        return True
    else:
        return False


def get_synonym(word):
    relate_list = []
    res = requests.get("https://dic.daum.net/search.do?q=" + word)
    soup = BeautifulSoup(res.content, "html.parser")
    try:
        word_id = soup.find("ul", class_="list_relate")
    except AttributeError:
        return word
    if word_id == None:
        return word
    for tag in word_id.find_all("a"):
        relate_list.append(tag.text)

    return random.choice(relate_list)


if __name__ == "__main__":
    print(isStopword("아홉"))
