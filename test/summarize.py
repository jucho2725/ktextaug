import six
import random
import ktextaug.utils as util
from ktextaug import *

"""
Augmentation
1.Random Swap
2.Random Insertion
3.Random Deletion
4.Synonym Replacement
5.Back Translation
6.Nois
"""




#######################
##Augmentation        #
#######################


def Augment(
    text, alpha_rs=0.2, alpha_ri=0.2, alpha_rd=0.1, alpha_sr=0.2, p_rd=0.2, num_iter=9
):
    translator = BackTranslate()
    noise_gen = NoiseGenerator()
    words = util.tokenize(text)
    num_words = len(words)
    augmented_sentence = {}
    num_per_tech = int(num_iter / 4) + 1
    n_rs = max(1, int(alpha_rs * num_words))
    n_ri = max(1, int(alpha_ri * num_words))
    n_sr = max(2, int(alpha_sr * num_words))

    # Add original words
    augmented_sentence["org"] = text

    # RS
    tmp = []
    for _ in range(num_per_tech):
        a_words = random_swap(words, n_rs)
        tmp.append(a_words)
    augmented_sentence["rs"] = tmp

    # RI
    tmp = []
    for _ in range(num_per_tech):
        a_words = random_insertion(words, n_ri)
        tmp.append(a_words)
    augmented_sentence["ri"] = tmp

    # RD
    tmp = []
    for _ in range(num_per_tech):
        a_words = random_deletion(words, p_rd)
        tmp.append(a_words)
    augmented_sentence["rd"] = tmp

    # SR
    tmp = []
    for _ in range(num_per_tech):
        a_words = synonym_replacement(words, n_sr)
        tmp.append(a_words)
    augmented_sentence["sr"] = tmp

    # Tran
    tmp = []
    a_words = translator.backtranslate(text, target_language="en")
    a_words2 = translator.backtranslate(text, target_language="ja")
    tmp.append(a_words)
    tmp.append(a_words2)
    augmented_sentence["bt"] = tmp

    # Noise
    tmp = []
    a_words = noise_gen.noise_generate1(text)
    a_words2 = noise_gen.noise_generate2(text)
    tmp.append(a_words)
    tmp.append(a_words2)
    augmented_sentence["noise"] = tmp

    return augmented_sentence


#######################
##    TEST            #
#######################

if __name__ == "__main__":

    text = "이 문장은 변형적 데이터 증강기법의 예시 문장입니다."
    # "가지고 있는 데이터에서 유의미한 단어 토큰만을 선별하기 위해서는 작업이 필요합니다."

    result = Augment(text)
    print("Original : ", text)
    print(len(result))
    print("Random Swap : ", result["rs"])
    print("Random Insertion : ", result["ri"])
    print("Random Deletion : ", result["rd"])
    print("Synonym Replacement : ", result["sr"])
    print("BackTranslation : ", result["bt"])
    print("Adding Noise : ", result["noise"])
