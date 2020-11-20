from ktextaug.aug_util import tokenize


from transformers import ElectraTokenizer
tok_elect = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
from transformers import BertTokenizer
tok_bertmul = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

def tokenize_tr(tok, text):
    return tok.tokenize(text)
ex1 = '아 더빙.. 진ㅉㅏ ㅉㅏ증ㄴㅏㄴㅔ요 목소리'
ex2 = '야 더빙.. 진짜 쨔증나네오 목소리'
print(tokenize(ex1))
print(tokenize(ex2))

print(tokenize_tr(tok_elect, ex1))
print(tokenize_tr(tok_elect, ex2))
print(tokenize_tr(tok_bertmul, ex1))
print('|'.join(tokenize_tr(tok_bertmul, ex2)))

from ktextaug.aug_util import tokenize



