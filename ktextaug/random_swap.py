from konlpy.tag import Mecab
import random

def tokenize(text):
   mecab = Mecab()
   return mecab.morphs(text)

def random_swap(words, n):
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

def main():
   ex = "아놔 첨엔 먼가 흥미진진할줄 알고 폰에 다운받아서남은 용량 부족하다고 경고 뜨는데도 꿋꿋이 쪼개서영화를 끝까지 본 내가 바보임 . 머이리 허무하고 씨지는 80년대 영구와 땡칠이같음? ㅜㅜ? 다 보자마자바로 삭제 ~ 폰용량 늘었다 ~"
   tok_words = tokenize(ex)
   print("tokenized")
   print(tok_words)
   alpha = 0.2
   swap_words = random_swap(tok_words, int(alpha * len(tok_words)))
   print("swapped")
   print(swap_words)
   # return random

if __name__ == "__main__":
   main()