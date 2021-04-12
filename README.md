# ktextaug


Data augmentation Toolkit for Korean text.
It provides transformative text augmentation methods.

한국어 텍스트 증강 기법을 모아둔 패키지입니다.
현재는 변형적 텍스트 증강기법만을 구현해두었으며, 생성적 텍스트 증강기법 모델 또한 추가될 예정입니다.

## Installation

### Prerequisites

* Python >= 3.6
* Beautifulsoup4>=4.6.0
* Googletrans==3.1.0a0
* Pandas>=1.0.4
* konlpy>=0.5.2
* nltk>=3.5

in command line:

```
pip install ktextaug
```

1. 현재 mecab 으로 토크나이저가 고정되어 있습니다. 
      konlpy 의 경우 자동으로 설치가 되나, mecab-ko는 따로 직접 설치해야 합니다. 
      (토크나이저를 직접 선택할 수 있도록 수정 예정 - 4월 중순)

2. 현재 nltk에 한국어 불용어 사전을 추가하여 사용중입니다. 
     한국어 불용어 사전은 다음 링크를 참고했습니다. 
     (불용어사전 및 nltk 설치 없이 사용가능하도록 수정 예정 - 4월 초)

  https://www.ranks.nl/stopwords/korean/ 

  https://bab2min.tistory.com/544

## Getting Started

ktextaug를 사용하는 간단한 예제입니다. 

```
import ktextaug

text = "이 문장은 변형적 데이터 증강기법의 예시 문장입니다."
tokenizer = bring_it_your_own # 토크나이저는 어떤 토크나이저를 사용하더라도 상관없습니다.
tokens = tokenizer.tokenize(text)
result = ktextaug.random_swap(tokens, 2) # 토큰 시퀀스 내 두 단어의 위치를 변경하는 작업(random swap)을 2회 시행합니다. 
print(result)
# ['이', '문장', '은', '예시', '적', '데이터', '기법', '증강', '의', '문장', '변형', '입니다', '.']
```

## More examples

더 자세한 사용 예시는 examples 폴더 내의 예시들을 확인해주세요.

- `summarize.py` : 각 기법을 사용한 예시를 보여줍니다.
- `multiprocessing.py` : .csv 형식의 데이터셋을 받아 증강된 데이터셋 파일을 제공해줍니다. 시간이 많이 소요되는 기법들을 multiprocessing 을 이용하처리했습니다. 

## Test it with sample data

데이터 증강기법의 성능을 확인하실 수 있도록, 매우 작은 데이터셋을 `src/data/` 에 올려두었습니다.
이 데이터는 nsmc 데이터셋의 훈련 데이터셋을 1000개 랜덤 샘플링한 결과입니다.
(출처: https://github.com/e9t/nsmc)

해당 데이터를 가지고 증강기법을 적용해서 결과의 차이를 확인해주세요!
(.csv 파일을 다루는 예시는 `multiprocessing.py` 에서 확인 가능합니다)

## Things to know

backtranslation 기법을 위해 사용되는 googletrans 패키지에 이슈가 있습니다. (아래 링크 참고)
https://github.com/ssut/py-googletrans/issues/234
해당 이슈가 해결될 때 까지 간혹 "AttributeError: 'NoneType' object has no attribute 'group'" 에러가 발생할 수 있습니다.

Update(21.04.12)
- https://github.com/ssut/py-googletrans/issues/286
- googletrans==3.1.0a0 을 설치시 문제가 해결된다고 합니다. 4월 12일 기준 테스트 완료

## Author, Contact

이 패키지는 성균관대학교 정윤경 교수님 연구실 ING-lab 에서 만들었으며, 참여한 사람들은 다음과 같습니다

박종혁, 이정훈, 전현규, 정민수, 조진욱

관련된 문의는 cju2725@gmail.com 으로 부탁드립니다.

## TO DO

1. Generative Models 추가 예정 (4월 중순)
2. tokenizer을 선언하는 식으로 사용하도록. 기본 tokenizer이 바뀌어야함 (추가설치 필요 없는 것으로)
3. bulk 에 대한 처리, multiprocessing 적용

## Acknowledgement

“이 기술은 과학기술정보통신부 및 정보통신기획평가원의 인공지능핵심인재양성사업(인공지능대학원지원(성균관대학교), No.2019-0-00421)의 연구결과로 개발한 결과물입니다.”

