import requests
from bs4 import BeautifulSoup
import random
import aug_util as util


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


def random_insertion(words):
    f_words = [w for w in words if (not util.isStopword(w)) and util.isWord(w)]
    Flag = True
    counter = 0
    while Flag:
        target = random.choice(f_words)
        new_syn = get_synonym(target)
        counter += 1
        if target == new_syn:
            pass
        elif counter == 30:
            Flag = False
        else:
            Flag = False
    words.insert(random.randrange(0, len(words)) - 1, new_syn)
    return words


if __name__ == "__main__":
    Sample = "철수가 밥을 빨리 먹었다."
    print("Sample : ", Sample)
    print(util.tokenize(Sample))
    print(random_insertion(Sample))
