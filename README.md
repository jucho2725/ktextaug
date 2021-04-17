# ktextaug


Data augmentation Toolkit for Korean text.
It provides transformative text augmentation methods.
We will release generative text augmentation methods (mid of April, hopefully)

한국어 텍스트 증강 기법을 모아둔 패키지입니다.
현재는 변형적 텍스트 증강기법만을 구현해두었으며, 생성적 텍스트 증강기법 모델 또한 추가될 예정입니다.
transformers 패키지 내부를 참고하면서 만들고 있습니다.


현재 버젼: 0.1.8
- 패키지 내 모든 함수와 모듈을 테스트하였습니다.
- 불용어 리스트를 패키지에 내장하고 원하는 경우 바꿀 수 있도록 변경했습니다.

일정
- 4월 말 : 생성 모델 추가 (속도 이슈 해결방법 고민중)
- 5월 : 테스트 및 첫 번째 공식 릴리즈 ?
  

## Installation

### Prerequisites

* Python >= 3.6
* konlpy>=0.5.2
* PyKomoran>=0.1.5
* Beautifulsoup4>=4.6.0 # for synonym search
* Googletrans==3.1.0a0   # for backtranslation

예제를 테스트하기 위해선 pandas, parmap 이 필요할 수 있습니다.

command line 설치 예시:

```
pip install ktextaug
```

build from source(latest):
```
git clone https://github.com/jucho2725/ktextaug.git

python setup.py
```

## Getting Started

ktextaug를 사용하는 간단한 예제입니다. 

```python
from ktextaug import random_swap

text = "이 문장은 변형적 데이터 증강기법의 예시 문장입니다."
tokenizer = bring_it_your_own   # 토크나이저는 어떤 토크나이저를 사용하더라도 상관없습니다.
tokens = tokenizer.tokenize(text) 
result = random_swap(tokens, 2) # 토큰 시퀀스 내 두 단어의 위치를 변경하는 작업(random swap)을 2회 시행합니다. 
print(result)
>>> ['이', '문장', '은', '예시', '적', '데이터', '기법', '증강', '의', '문장', '변형', '입니다', '.']
```

패키지에서 제공하는 형태소 분석기(토크나이저) 모듈은 mecab 또는 komoran을 불러옵니다. 두 토크나이저 모두 별도의 설치과정이 필요하니 아래 링크를 참고해주세요. 원하는 토크나이저를 사용할 수도 있습니다.

- Mecab 설치 방법 [[링크]](https://sikaleo.tistory.com/104) - fabric 으로 쉽게 설치
- PyKomoran 설치 방법 [[링크]](https://komorandocs.readthedocs.io/ko/latest/firststep/installation.html)

```python
from ktextaug.tokenization_utils import Tokenizer

tokenizer = Tokenizer(tokenizer_or_name="komoran") # mecab

# OR you can use your own tokenizer(should be module, neither function nor object)
your_own_tokenizer = ABC # module
tokenizer = Tokenizer(tokenizer_or_name=your_own_tokenizer) 

```

## More examples

#### noise_generation 모듈 사용법

노이즈 생성은 @hkjeon13(전현규) 의 노이즈 생성을 따랐음을 밝힙니다

https://github.com/hkjeon13/noising-korean

노이즈를 생성하는 방법은 총 3가지가 구현되어 있습니다.

"jamo_split": 자모 분리(alphabet separation)에 의한 노이즈 추가 방법. 글자의 자음과 모음을 분리합니다. 단, 가독성을 위해 종성이 없으며 중성이  'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅗ' 가 아닐 경우 실행합니다(예: 안녕하세요 > 안녕ㅎㅏㅅㅔ요)

"vowel_change": 모음 변형에 의한 노이즈 추가 방법. 글자의 모음을 변형시킵니다. 단, 가독성을 위해 종성이 없으며 중성이 'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ' 일 경우 실행합니다(예: 안녕하세요 > 안녕햐세오).

"phonological_change": 음운변화에 의한 노이즈 추가 방법. 발음을 바탕으로 단어를 변형시킵니다(너무 닮았다 > 너무 달맜다).

**실행 예시**
```python
import noise_generation

text = '행복한 가정은 모두가 닮았지만, 불행한 가정은 모두 저마다의 이유로 불행하다.'
noise_generation.noise_generate(text, prob=1., option="jamo_split")
>> 행복한 ㄱㅏ정은 모두ㄱㅏ 닮았ㅈㅣ만, 불행한 ㄱㅏ정은 모두 ㅈㅓㅁㅏㄷㅏ의 ㅇㅣ유로 불행ㅎㅏㄷㅏ.
```


**변형 예시**
```python
[original]  행복한 가정은 모두가 닮았지만, 불행한 가정은 모두 저마다의 이유로 불행하다.

[jamo_split, prob=1] 행복한 ㄱㅏ정은 모두ㄱㅏ 닮았ㅈㅣ만, 불행한 ㄱㅏ정은 모두 ㅈㅓㅁㅏㄷㅏ의 ㅇㅣ유로 불행ㅎㅏㄷㅏ.

[vowel_change, prob=1] 행복한 갸정은 묘듀갸 닮았지만, 불행한 갸정은 묘듀 져먀댜의 이우료 불행햐댜.

[phonological_change, prob=1] 행복한 가정은 모두가 달맜지만, 불행한 가정은 모두 저마다의 이유로 불행하다.
```

#### 기타
- 'phonological_change' 방법은 현재 비음화, 유음화, 구개음화, 연음 등을 구현하고 있으며, 추후 확대될 예정입니다(누락된 규칙이 있을 수 있으니, 발견 시 피드백 주시면 감사하겠습니다).
- prob는 변형 가능한 글자들에 대해서 해당 확률만큼 확률적으로 실행됩니다(prob가 1이라고 해서 모든 텍스트가 변경되는 것이 아닙니다).


**더 자세한 사용 예시는 examples 폴더 내의 예시들을 확인해주세요.**

- `summarize.py` : 각 기법을 사용한 예시를 보여줍니다.
- `multiprocessing.py` : .csv 형식의 데이터셋을 받아 증강된 데이터셋 파일을 제공해줍니다. 시간이 많이 소요되는 기법들을 multiprocessing 을 이용하여 처리했습니다. 

## Test it with sample data

데이터 증강기법의 성능을 확인하실 수 있도록, 매우 작은 데이터셋을 `examples/data/` 에 올려두었습니다.
이 데이터는 nsmc 데이터셋의 훈련 데이터셋을 1000개 랜덤 샘플링한 결과입니다.
(출처: https://github.com/e9t/nsmc)

해당 데이터를 가지고 증강기법을 적용해서 결과의 차이를 확인해주세요!
(.csv 파일을 다루는 예시는 `multiprocessing.py` 에서 확인 가능합니다)

## Things to know

1. 한국어 불용어 사전의 경우 다음 링크의 파일을 그대로 가져왔습니다. 
   https://github.com/stopwords-iso/stopwords-ko/blob/master/stopwords-ko.txt

2. backtranslation 기법을 위해 사용되는 googletrans 패키지에 이슈가 있습니다. (아래 링크 참고)
   https://github.com/ssut/py-googletrans/issues/234
   해당 이슈가 해결될 때 까지 간혹 "AttributeError: 'NoneType' object has no attribute 'group'" 에러가 발생할 수 있습니다.

   Update(21.04.12) googletrans==3.1.0a0 을 설치시 문제가 해결된다고 합니다. [(링크)](https://github.com/ssut/py-googletrans/issues/286) 4월 12일 기준 테스트 완료

## Contribution

이 패키지는 성균관대학교 정윤경 교수님 연구실 ING-lab 에서 진행한 프로젝트로 시작되었으며,
당시 참여한 사람들은 다음과 같습니다.
- 조진욱, 전현규, 박종혁, 이정훈, 정민수

보다시피 아직 패키지에 부족한 부분이 많습니다.
Contributor가 되고 싶으시다면, 언제든 issue, PR, 등을 부탁드립니다 :)

Contact: cju2725@gmail.com

## TO DO

1. Generative Models 추가 예정 (4월 말)
2. 기본 tokenizer이 바뀌어야함 (추가설치 필요 없는 것으로)
3. bulk 에 대한 처리, multiprocessing 적용
4. synonym search 동의어 못찾을시 문제 해결
5. documentation 작성 

## Acknowledgement

“이 기술은 과학기술정보통신부 및 정보통신기획평가원의 인공지능핵심인재양성사업(인공지능대학원지원(성균관대학교), No.2019-0-00421)의 연구결과로 개발한 결과물입니다.”

