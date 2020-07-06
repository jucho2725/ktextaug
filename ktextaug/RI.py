import requests
from bs4 import BeautifulSoup
#import re
import random

def get_synonym(word):
    relate_list = []
    res = requests.get("https://dic.daum.net/search.do?q="+word) #OK
    soup = BeautifulSoup(res.content, 'html.parser') #Parsing
    try : 
        word_id = soup.find('ul',class_ = 'list_relate')
    except AttributeError:
        #print(word)
        return word
    if word_id == None:
        #print(word)
        return word
    for tag in word_id.find_all("a"):
        relate_list.append(tag.text)
    #print(relate_list)
    
    return random.choice(relate_list)

def random_insertion(sentence):
    words = [w for w in sentence.split(" ") if w != ""]
    target = random.choice(words)
    new_syn = get_synonym(target)
    words.insert(random.randrange(0,len(words)),new_syn)
    return " ".join(words)

if __name__ == '__main__':
    print(random_insertion("아.. 더빙 완전 짜증났어요"))