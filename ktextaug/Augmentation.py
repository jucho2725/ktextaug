#Summarize

##Module
#from backtranslate import BackTranslate #Back Translation
import RD
import RI
import random_swap as RS
import aug_util
#from google.cloud import translate_v2 as translate

#Randon Delection


#Class
class Augmentation():
    
    def __init__(self,Tokenizer):
        #file
        self.Tokenizer = Tokenizer
        self.Token = None
        self.P = 0.3
        self.alpha = 0.2
        self.Token_len = None
        #Translation
        #self.translate_client = translate.Client()
        self.result = None
        
    def _Tokenizer(self,sentence):
        if self.Tokenizer == "Mecab" :
            self.Token = aug_util.tokenize(sentence)
            self.Token_len = len(self.Token)
            
        elif self.Tokenizer == "khaiii":
            print("Use Khaiii Tokenizer")
            
        elif self.Tokenizer == "Komoran":
            print("Use Komoran Tokenizer")
        else :
            print("We cannot use",self.Tokenizer,"Tokenizer.")
    
    
    def Replace(self,sentence):
        if self.Token == None: self._Tokenizer(sentence)
        pass
    
    def Deletion(self,sentence):
        if self.Token == None: self._Tokenizer(sentence)
        return RD.random_deletion(self.Token, self.P)
    
    def Insertion(self,sentence):
        if self.Token == None: self._Tokenizer(sentence)
        return RI.random_insertion(self.Token)
    
    def Swap(self,sentence):
        if self.Token == None: self._Tokenizer(sentence)
        return RS.random_swap(self.Token,self.N)
    
    def Translate(self):
        #BackTranslate.
        pass
    
    def Augmentation(self):
        pass
    '''
    
    '''
if __name__ == "__main__":
    sample = "철수가 밥을 빨리 먹었다."
    A = Augmentation()
    A._Tokenizer(sample)
    print(A._Tokenizer(sample))
    