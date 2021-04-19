import requests
from bs4 import BeautifulSoup
import random

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

textiowrapper = pkg_resources.open_text(__package__, 'stopwords-ko.txt') # __package__ 는 current module 의미
stopwords = [w.strip() for w in textiowrapper] # list of string
PUNC = [".", ",", ":", ";", "?", "!"]

def keep_punctuation(text):
    if text[-1] in PUNC:
        keep = text[-1]
        text = text[:-1].copy()
    else:
        keep = None
        text = text.copy()
    return text, keep

def define_stopwords(new_stopwords: list) -> list:
    """
    define new stopwords list
    :param stopwords: list of string.
    :return:
    """
    global stopwords
    stopwords = new_stopwords
    print("stopwords list has been updated.")

def isWord(word):
    return word.isalnum()

def isStopword(word):
    if word in stopwords:
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