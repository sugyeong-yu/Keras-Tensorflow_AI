# Chapter4. RNN

- 순환신경망(recurrent neural network, RNN) : 계층의 출력이 순환하는 인공신경망
- 은닉계층의 결과가 다음계층으로 넘어갈 뿐아니라 **자기계층으로 다시 돌아온다.**
- 따라서, 시계열 정보처럼 앞뒤 신호가 서로 상관도가 있는경우, 인공신경망의 성능을 더 높일 수 있다.

## 1. RNN원리
- RNN은 신호를 순환하여 시계열신호와 같이 상호관계가 있는 신호를 처리하는 인공신경망이다.
- 그러나 단순한 방식으로 구현하게되면 학습이 잘 이루어지지 않을 수 있다.
- 출력된 신호가 계속 순환하면 활성화함수를 반복적으로 거치게 되어 경삿값을 구하기가 힘들기 때문이다.
  - 활성화 함수는 약간의 입력값이 커져도 미분값이 매우 작아진다. 
  - 이때, 순환하여 활성화함수를 거치게되면 미분값이 0에 가까워져 학습이 어렵게 되는것.

### 1.1 LSTM 구조 및 동작
- LSTM은 아래 그림과 같이 입력조절벡터, 망각벡터, 출력조절벡터를 이용해 입력과 출력신호를 gating한다.
![image](https://user-images.githubusercontent.com/70633080/111955668-2544ff00-8b2d-11eb-9972-7444c6ca678d.png)
- gating이란 신호의 양을 조정해주는 기법이다.
- 입력조절벡터는 입력신호가 tanh활성화 함수의 완전연결계층을 거친 이후의 값을 조절한다.
- 망각벡터는 과거입력의 일부를 현재입력에 반영한다.
- 출력조절벡터는 과거의 값과 수정된 입력값을 고려해 tanh활성화 함수로 gating을 수행한다.
- 이 최종 결과는 데이터 처리를 위한 tanh계층 그리고 gating을 위한 새 sigmoid계층에 다시 입력으로 들어간다.

## 2. 문장을 판별하는 LSTM구현
- 영화추천데이터베이스를 이용해 같은 사람이 영화에 대한 느낌을 서술한 글과 영화가 좋은지 나쁜지 별표등으로 판단한 결과와의 관계를 학습.
- 학습이 완료된 후 새로운 평가글을 주었을때 인공신경망이 판별결과를 예측하도록 만든다.
### 2.1 라이브러리 임포트
```
from __future__ import print_function 
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras import layers, models
```
- __future__ import print_function : 파이썬2와 3간의 호환성을 위한 것임. > python3문법으로 print를 해도 python2에서 코드가 돌아갈 수 있게된다.
- sequence : preprocessing이 제공하는 서브패키지이다.
  - pad_sequence()와 같은 sequence를 처리하는 함수를 제공한다.
- models : 케라스 모델링에 사용되는 서브패키지이다. 
- layers : 인공신경망의 계층을 만드는 서브패키지
  - Dense, Embedding, LSTM 사용가능
### 2.2 데이터준비
