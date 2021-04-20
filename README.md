# ktextaug


Data augmentation Toolkit for Korean text.
It provides transformative text augmentation methods.
We will release generative text augmentation methods (mid of April, hopefully)

한국어에 적용가능한 텍스트 증강 기법을 모아둔 패키지입니다.
현재는 변형적 텍스트 증강기법만을 구현해두었으며, 생성적 텍스트 증강기법 모델 또한 추가될 예정입니다.
transformers 패키지 내부를 참고하면서 만들고 있습니다.

현재 버젼: 0.1.9

* TextAugmentation() 을 통해 bulk, 즉 대량의 데이터를 multiprocessing 하도록 구현되었습니다.

- 노이즈가 포함된 vocab을 가진 기본 subword tokenizer과 다른 토크나이저들을 만들었습니다. 
- 기타 버그 해결 

일정
- 4월 말 : 생성 모델 추가 (속도 이슈 해결 필요)
- 5월 : 테스트 및 첫 번째 공식 릴리즈 ?

## Installation

### Prerequisites

* Python >= 3.6

* Beautifulsoup4>=4.6.0  # for synonym search
* Googletrans==3.1.0a0   # for backtranslation
  
* konlpy>=0.5.2                # for Mecab tokenizer
* PyKomoran>=0.1.5       # for Komoran tokenizer
* transformers>=2.6.0    # for subword tokenizer


command line 설치 예시:

```
pip install ktextaug
```

## Getting Started

ktextaug를 사용하는 간단한 예제입니다.

패키지 0.1.9 버젼부턴 기본적으로 TextAugmentation() 을 사용하여 처리하는 것을 권장합니다. multiprocessing 을 이용하여 대용량의 데이터를 빠르게 처리할 수 있도록 만들었습니다. 

```python
from ktextaug import TextAugmentation

sample_text = '달리는 기차 위에 중립은 없다. 미국의 사회 운동가이자 역사학자인 하워드 진이 남긴 격언이다.'
sample_texts = ['프로그램 개발이 끝나고 서비스가 진행된다.', '도움말을 보고 싶다면 --help를 입력하면 된다.']
agent = TextAugmentation(tokenizer="mecab",
                        num_processes=1) # num_process 가 -1 일시 자동으로 가능한 process의 절반으로 할당
print(agent.generate(sample_text))     # default is back_translation
print(agent.generate(sample_texts))
```

함수를 직접 불러오는 것 또한 가능합니다. 

```python
from ktextaug.tokenization_utils import Tokenizer
from ktextaug import random_swap
from random import Random

text = "이 문장은 변형적 데이터 증강기법의 예시 문장입니다."
tokenizer = Tokenizer(tokenizer_name="mecab")

result = random_swap(text_or_tokens=text,
                     tokenizer=tokenizer,
                     rng=Random(2021),
                     n_swaps=2) # random_swap 고유 파라메터. 토큰 시퀀스 내 두 단어의 위치를 변경하는 작업(random swap)을 2회 시행합니다. 
 
print(result)
>>> ['이', '문장', '은', '예시', '적', '데이터', '기법', '증강', '의', '문장', '변형', '입니다', '.']
```

## Tokenizer

ktextaug 패키지에서 제공하는 형태소 분석기(토크나이저) 모듈은 다양한 존재하는 토크나이저를 쉽게 사용할 수있도록 만들었습니다. 설치가 필요없는 기본 토크나이저로서 또는 subword tokenizer 을 만들어두었습니다. mecab, komoran  토크나이저의 경우 별도의 설치과정이 필요하니 아래 링크를 참고해주세요. 원하는 토크나이저를 사용할 수도 있습니다.

- Mecab 설치 방법 [[링크]](https://sikaleo.tistory.com/104) - fabric 으로 쉽게 설치
- PyKomoran 설치 방법 [[링크]](https://komorandocs.readthedocs.io/ko/latest/firststep/installation.html)

```python
from ktextaug.tokenization_utils import Tokenizer
from ktextaug import random_swap
from random import Random

# get_tokenize_fn 함수의 사용예시
tokenizer = Tokenizer(tokenizer_name="komoran") # 별도의 설치 과정 필요. default 인 subword 만 설피과정 필요 없음.
tokenizer = bring_it_your_own    # 토크나이저는 어떤 토크나이저를 사용하더라도 상관없습니다.

result = random_swap(text_or_tokens=text,
                     tokenizer=tokenizer, 
                     rng=Random(2021),
                     n_swaps=2)

```

## More examples

- How_to_use 에 기본적인 사용법, 노이즈 생성에 대한 상세 설명 및  예시에 대해 적혀있습니다.


## Test it with sample data(0.1.8 에서 테스트)

데이터 증강기법의 성능을 확인하실 수 있도록, 매우 작은 데이터셋을 `examples/data/` 에 올려두었습니다.
이 데이터는 nsmc 데이터셋의 훈련 데이터셋을 1000개 랜덤 샘플링한 결과입니다.
(출처: https://github.com/e9t/nsmc)

해당 데이터를 가지고 증강기법을 적용해서 결과의 차이를 확인해주세요!

## Things to know

1. 노이즈 생성은 @hkjeon13(전현규) 의 노이즈 생성을 따랐습니다
   

https://github.com/hkjeon13/noising-korean

2. 한국어 불용어 사전의 경우 다음 링크의 파일을 그대로 가져왔습니다. 
   https://github.com/stopwords-iso/stopwords-ko/blob/master/stopwords-ko.txt

## Contribution

이 패키지는 성균관대학교 정윤경 교수님 연구실 ING-lab 에서 진행한 프로젝트로 시작되었으며,
당시 참여한 사람들은 다음과 같습니다.
- 조진욱, 전현규, 박종혁, 이정훈, 정민수

아직 패키지에 부족한 부분이 많습니다.
Contributor가 되고 싶으시다면, 언제든 issue, PR, 등을 부탁드립니다 :)

Contact: cju2725@gmail.com

## TO DO

1. Generative Models 추가 예정 (4월 말)
2. synonym search 동의어 못찾을시 문제 해결
3. documentation 작성 
4. pkg_resources 성능 오버헤드 관련 이슈 
https://docs.python.org/ko/3/library/importlib.html#module-importlib.resources



