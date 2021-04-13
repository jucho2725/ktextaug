import requests
from bs4 import BeautifulSoup


def synonym_replace(corpus, tokenizer,rng, n_rep,**kwargs):
    # check if there is a punctuation mark
    if isinstance(corpus, str):
        corpus = tokenizer.tokenize(corpus)

    punctuations = [".", ",", ":", ";", "?", "!"]
    if corpus[-1] in punctuations:
        keep = corpus[-1]
        words = corpus[:-1].copy()
    else:
        keep = None
        words = corpus.copy()

    result = words[:]
    nonStop = [w for w in result if (not isStopword(w)) and isWord(w)]
    rng.shuffle(nonStop)
    num_replacement = 0
    for random_word in nonStop:
        synonym = get_synonym(random_word, rng)
        if len(synonym) >= 1:
            synonym = rng.choice(list(synonym))
            new_word = [synonym if word == random_word else word for word in result]
            num_replacement += 1
        if num_replacement >= n_rep:
            break

    sentence = " ".join(new_word)
    result = sentence.split(" ")
    return result + [keep]


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


def get_synonym(word, rng):
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

    return rng.choice(relate_list)