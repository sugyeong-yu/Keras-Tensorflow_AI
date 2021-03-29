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
- IMDB : 케라스가 제공하는 공개데이터
  - 25000건의 영화평과 이진회된 영화평점정보(추첨=1, 비추천=0)을 담고있다.
  - 평점정보는 별점이 많은경우는 긍정, 그렇지 않으면 부정으로 나뉜정보다
- 데이터클래스 선언
```
class Data:
    def __init__(self, max_features=20000, maxlen=80):
        (x_train, y_train), (x_test, y_test) = imdb.load_data(
            num_words=max_features)
```
- 서브패키지 imdb안의 load_data()를 이용해 데이터를 불러옴
- 최대 단어 빈도를 max_feature값으로 제안함. > 여기서는 20000이 사용됨.

```
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
```
- 일반적으로 데이터셋의 문장들은 길이가 다 다르기때문에 LSTM이 처리하기 적합하도록 길이를 통일하는 작업을 진행한다.
- 문장에서 maxlen이후의 단어들은 케라스 서브패키지인 sequence에서 제공하는 pad_sequences()함수로 잘라낸다.
  - 여기서는 최대길이를 80으로 제한함.
- 문장의 길이가 maxlen보다 작을경우 부족한 부분을 0으로 채워준다.
  - value라는 변수로 채우는 값을 설정할 수 있다.

### 2.3 모델링
```
class RNN_LSTM(models.Model):
    def __init__(self, max_features, maxlen):
```
- 모델링은 ```models.Model``` 클래스를 상속해서 만든다.
- 입력층을 먼저 만들고 다음으로 임베딩계층을 포함한다.
```
x = layers.Input((maxlen,))
h = layers.Embedding(max_features, 128)(x)
```
- 최대 특징점수를 20000으로 설정하였으며 임베딩 후 출력벡터크기를 128로 설정하였다.
- 입력의 각 샘플은 80개의 원소로 된 1차원 신호열이였지만 임베딩 계층을 통과하며 128의 길이를 가지는 벡터로 바뀌면서 8 * 128로 바뀐다.
```
h = layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2)(h)
y = layers.Dense(1, activation='sigmoid')(h)
super().__init__(x, y)
```
- 노드128개로 구성된 LSTM계층을 포함한다.
- dropout, recurrent_dropout : 일반 드롭아웃과 순환 드롭아웃을 모두 20%로 설정하여 사용하였다.
- 최종적으로 출력을 sigmoid활성화 함수로 구성된 출력노드하나로 구성한다.
```
self.compile(loss='binary_crossentropy',
                     optimizer='adam', metrics=['accuracy'])
```
- 손실함수와 최적화 함수를 argument로 지정하여 모델을 컴파일한다.
- 긍정인지 부정인지에 대한 이진판별값을 출력으로 다룬다.
  - 손실함수를 binary_crossentropy로, 최적화 함수를 adam으로 설정
  - 학습기간에는 에포크마다 손실뿐만아니라 정확도도 구하도록 **metric에 accuracy를 추가.**

### 2.4 학습 및 성능평가
- 학습 및 성능 평가를 담당할 클래스를 만든다.
```
class Machine:
    def __init__(self,
                 max_features=20000,
                 maxlen=80):
        self.data = Data(max_features, maxlen)
        self.model = RNN_LSTM(max_features, maxlen)
```
- max_features=20000, maxlen=80으로 지정한다.
- max_features는 다루는 단어의 최대수이다.
  - 빈도순위가 20000등 안에 드는 단어까지만 취급한다는 뜻.
  - **keras.datasets.imdb.load_data함수에 들어가는 매개변수**이다.
- 학습과 평가를 수행하는 run()함수를 만든다.
```
    def run(self, epochs=3, batch_size=32):
        data = self.data
        model = self.model
        print('Training stage')
        print('==============')
        model.fit(data.x_train, data.y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(data.x_test, data.y_test))

        score, acc = model.evaluate(data.x_test, data.y_test,
                                    batch_size=batch_size)
        print('Test performance: accuracy={0}, loss={1}'.format(acc, score))
```
- 평가결과는 수행하는 컴퓨터마다 무작위로 선정되는 seed값의 차이로 약간 달라질 수 있다.
- 전체 코드는 [4-1.py](https://github.com/sugyeong-yu/Keras_AI/blob/main/CH4.RNN/4-1.py)에서 확인가능하다.

## 3. 시계열데이터를 예측하는 LSTM구현
- 세계항공여행승객 수의 증가에 대한 데이터를 활용해
- 이전 12월치 승객 수를 이용해 다음달의 승객 수를 예측한다.
### 3.1 패키지 임포트
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from keras import models, layers
import seaborn as sns

from keraspp import skeras
```
- model_selection : 데이터를 학습과 검증용으로 나누는 함수
- skeras : 케라스 학습결과를 그래프로 그려주는 기능을 제공하는 서브패키지
### 3.2 데이터불러오기
- Dataset class를 만든다.
```
class Dataset:
    def __init__(self, fname='international-airline-passengers.csv', D=12):
```
- 초기화 함수 파라미터로 데이터파일명과 시계열 길이가 주어진다.
- 다음으로 데이터를 불러와 학습데이터와 평가데이터를 나눈다.
```
        data_dn = load_data(fname=fname)
        X, y = get_Xy(data_dn, D=D)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42) 
```
- 다음으로는 이 클래스를 이용할때 결과를 볼 수 있도록 결과를 멤버변수에 저장한다.
```
      self.X, self.y = X, y
      self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test   
```
- load_data() 구현
```
    def load_data(fname='international-airline-passengers.csv'):
        dataset = pd.read_csv(fname, usecols=[1], engine='python', skipfooter=3)
        data = dataset.values.reshape(-1)
```
- pd.read_csv(fname,usecols=[1] : 데이터 중 승객 수에 해당하는 1번쨰 열만 불러온다. 
- 다음으로 pandas df 에서 numpy array로 변환 후 1차원모양으로 만들어준다.
- 데이터의 가로축은 time, 세로축은 passengers 이다.
```
        plt.plot(data)
        plt.xlabel('Time'); plt.ylabel('#Passengers')
        plt.title('Original Data')
        plt.show()

        # data normalize
        data_dn = (data - np.mean(data)) / np.std(data) / 5
        plt.plot(data_dn)
        plt.xlabel('Time'); plt.ylabel('Normalized #Passengers')
        plt.title('Normalized data by $E[]$ and $5\sigma$')
        plt.show()

        return data_dn
```
- 데이터를 표준화한다.
  - 표준화 방법은 평균을 제거하고 분산을 단위 값으로 만든다. 
  - 여기서는 분산의 제곱근을 5로 나눈다.
  - 따라서 데이터는 표준화된 승객 수 이다.
- 다음으로 get_Xy()를 구현한다.
```
    def get_Xy(data, D=12):
        # make X and y
        X_l = []
        y_l = []
        N = len(data) # 몇개월 가량의 승객 수
        assert N > D, "N should be larger than D, where N is len(data)"
        for ii in range(N-D-1):
            X_l.append(data[ii:ii+D])
            y_l.append(data[ii+D])
        X = np.array(X_l)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        y = np.array(y_l)
        print(X.shape, y.shape)
        return X, y
```
- 이는 D 샘플만큼의 시계열 데이터를 한칸씩 옮겨가며 시계열벡터로 만든다.
- 그리고 D+1샘플의 값을 레이블로 하였다.
- 즉,  D개월간의 승객 수 변화를 통해 다음달 승객 수를 예측할수 있는지를 알아보는 데이터셋을 만들게 된다.
### 3.3 LSTM 시계열 회귀 모델링
- 케라스에서 제공하는 모델을 이용해 만든다. 
- 먼저 입력계층을 정의한다.
```
m_x = layers.Input(shape=shape)
```
- 입력 데이터의 모양은 X.shape[1:] 이다.
  - n,h,w,c 중에 하나의 시계열 데이터에 대한 shape(h,w,c)를 사용
- 다음으로 LSTM과 출력 계층을 설정한다.
```
m_h=layers.LSTM(10)(m_x)
m_y=layers.Dense(1)(m_h)
```
- LSTM의 노드 수는 10
- 출력 계층의 노드 수는 이진판별이므로 1
- 다음으로 모델을 정의한다.
```
m=models.Model(m_x,m_y)
m.compile('adam','mean_squared_error')
m.summary()
```
### 3.4 학습하고 평가하기
- Machine class는 시계열 LSTM을 학습하고 평가하는 클래스이다.
- 클래스 선언과 초기화
```
class Machine():
    def __init__(self):
        self.data = Dataset()
        shape = self.data.X.shape[1:]
        self.model = rnn_model(shape)
```
- 데이터생성에 필요한 Dataset() 클래스의 객체를 만듬.
- LSTM입력 계층의 크기를 shape이라는 변수에 저장.
  - shape은 (12,1)이 된다.
- rnn_model()함수로 LSTM모델을 만들어 model변수에 저장.
- 다음으로 실행함수를 만든다.
```
     def run(self, epochs=400):
            d = self.data
            X_train, X_test, y_train, y_test = d.X_train, d.X_test, d.y_train, d.y_test
            X, y = d.X, d.y
            m = self.model 
```
- self.data의 내부변수들인 X_train, X_test, y_train, y_test를 선언
- 모델의 학습을 진행한다.
```
            h = m.fit(X_train, y_train, epochs=epochs, validation_data=[X_test, y_test], verbose=0)

            skeras.plot_loss(h)
            plt.title('History of training')
            plt.show()
```
- fit()으로 학습한다. 
- 학습이 완료되면 학습 곡선을 ```skkeras.plot_loss()```로 그린다.
- 학습이 얼마나 잘되었는가 평가한다.
```
             yp = m.predict(X_test)
             print('Loss:', m.evaluate(X_test, y_test))
```
- evaluate()로 학습성능을 평가한다.
- 학습이 끝난 모델을 평가데이터를 이용해 예측한 뒤 정확도를 계산.
- 원래 레이블 정보인 y_test와 예측한 정보인 yp를 그려 서로를 비교한다.
- 전체코드는 [4.2.py]()에서 확인가능하다.
