#library import
import six
import random
import ktextaug.aug_util as util
from ktextaug.aug_util import tokenize
from ktextaug.NoiseGeneration import NoiseGenerator
'''
Augmentation
1.Random Swap
2.Random Insertion
3.Random Deletion
4.Synonym Replacement
5.Back Translation
6.Noise
'''

#######################
##Random Swap         # parameter : n
#######################
def random_swap(words, n): # N-times
   new_words = words.copy()
   for _ in range(n):
      new_words = swap_word(new_words)
   return new_words

def swap_word(new_words):
   random_idx_1 = random.randint(0, len(new_words)-1)
   random_idx_2 = random_idx_1
   counter = 0
   while random_idx_2 == random_idx_1:
      random_idx_2 = random.randint(0, len(new_words)-1)
      counter += 1
      if counter > 3:
         return new_words
   new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
   return new_words

#######################
##Random Insertion    #
#######################
def random_insertion(word,n):
    results = word[:]
    for i in range(n):
        results = word_insert(results)
    return results

def word_insert(words):
    #words = util.tokenize(sentence)
    f_words = [w for w in words if (not util.isStopword(w)) and util.isWord(w)]
    Flag = True
    counter = 0
    while Flag :
        target = random.choice(f_words)
        new_syn = util.get_synonym(target)
        counter += 1
        if counter > 30:
            Flag = False
        # print(counter)
        if target == new_syn :
            pass
        else:
            Flag = False
    result = words[:]
    result.insert(random.randrange(0,len(words))-1,new_syn)
    return result
#######################
##Random Deletion     #
#######################
def random_deletion(words,p):
    #sentence = util.tokenize(sentence)
    if len(words) == 1 :
        return words
    
    new_words = []
    
    for word in words:
        r = random.uniform(0,1)
        if r > p :
            new_words.append(word)
    
    if len(new_words) == 0 :
        rand_int = random.randint(0,len(words)-1)
        return [words[rand_int]]
    
    return new_words
#######################
##Synonym Replacement #
#######################
def synonym_replacement(words,n):
    result = words[:]
    nonStop = [w for w in result if (not util.isStopword(w)) and util.isWord(w)]
    random.shuffle(nonStop)
    num_replacement = 0
    for random_word in nonStop:
        synonym = util.get_synonym(random_word)
        if len(synonym) >= 1 :
            synonym = random.choice(list(synonym))
            new_word = [synonym if word == random_word else word for word in result]
            num_replacement += 1
        if num_replacement >= n :
            break
        
    sentence = " ".join(new_word)
    result = sentence.split(" ")
    
    return result
#######################
##Back Translation    #
#######################
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
#######################
##Augmentation        #
#######################
        
def Augment(text,alpha_rs = 0.2,alpha_ri = 0.2 ,alpha_rd = 0.1 ,alpha_sr = 0.2,p_rd = 0.2,num_iter = 9):
    translator = BackTranslate()
    noise_gen = NoiseGenerator()
    words = util.tokenize(text)
    num_words = len(words)
    augmented_sentence = {}
    num_per_tech = int(num_iter/4) + 1
    n_rs = max(1,int(alpha_rs*num_words))
    n_ri = max(1,int(alpha_ri*num_words))
    n_sr = max(2,int(alpha_sr*num_words))

    #Add original words
    augmented_sentence['org']= text

    #RS
    tmp = []
    for _ in range(num_per_tech):
        a_words = random_swap(words,n_rs)
        tmp.append(a_words)
    augmented_sentence['rs']= tmp
        
    #RI
    tmp = []
    for _ in range(num_per_tech):
        a_words = random_insertion(words,n_ri)
        tmp.append(a_words)
    augmented_sentence['ri'] = tmp
        
    #RD
    tmp = []
    for _ in range(num_per_tech):
        a_words = random_deletion(words, p_rd)
        tmp.append(a_words)
    augmented_sentence['rd'] = tmp

    #SR
    tmp = []
    for _ in range(num_per_tech):
        a_words = synonym_replacement(words, n_sr)
        tmp.append(a_words)
    augmented_sentence['sr'] = tmp
        
    #Tran
    tmp = []
    a_words =translator.backtranslate(text, target_language='en')
    a_words2 =translator.backtranslate(text, target_language='ja')
    tmp.append(a_words)
    tmp.append(a_words2)
    augmented_sentence['bt'] = tmp

    #Noise
    tmp = []
    a_words = noise_gen.noise_generate1(text)
    a_words2 = noise_gen.noise_generate2(text)
    tmp.append(a_words)
    tmp.append(a_words2)
    augmented_sentence['noise'] = tmp
    
    return augmented_sentence
#######################
##    TEST            #
#######################
    
if __name__ == "__main__":
    
    text = "이 문장은 변형적 데이터 증강기법의 예시 문장입니다."
        # "가지고 있는 데이터에서 유의미한 단어 토큰만을 선별하기 위해서는 작업이 필요합니다."

    '''
    RS = random_swap(word, 3)
    RI = random_insertion(word,5)
    RD = random_deletion(word, 0.1)
    SR = synonym_replacement(word, 3)
    print("Original : ",word)
    print("Random Swap : ",RS)
    print("Random Insertion : ",RI)
    print("Random Deletion : ",RD)
    print("Synonym Replacement : ",SR)
    '''
    result = Augment(text)
    print("Original : ", text)
    print(len(result))
    print("Random Swap : ", result['rs'])
    print("Random Insertion : ", result['ri'])
    print("Random Deletion : ", result['rd'])
    print("Synonym Replacement : ", result['sr'])
    print("BackTranslation : ", result['bt'])
    print("Adding Noise : ", result['noise'])