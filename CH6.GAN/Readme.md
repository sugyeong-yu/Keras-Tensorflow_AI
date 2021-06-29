# Chapter6. GAN
- GAN은 경쟁적 학습방법을 이용하는 생성형 인공지능

## 1. GAN의 원리
- GAN은 실제데이터와 비슷한 확률분포를 가지는 fake data를 만들어낸다. 
- fake data는 생성데이터 라고도 한다.
- GAN은 레이블이 없는 정보를 다루는 비지도 학습 방법이다.
- 입력데이터 : 무작위 잡음
- 출력데이터 : 입력데이터보다 높은 차원의 특정 분포도를 갖는 데이터

### 1-1. GAN의 구조
- 생성망(Generator)와 판별망(Discriminator) 으로 구성되어 있다.
- Generator : 실제데이터와 확률분포상 유사한 데이터를 만듬
- Discriminator : 실제데이터인지 생성데이터인지 구분
- Discriminator는 진짜와 가짜를 더 잘 구별하도록 학습함. 
- Generator는 Discriminator를 더 잘속이도록 학습함.
- G <-> D 이 과정을 순환하며 학습 (경쟁적 학습법)

### 1-2. GAN의 동작원리
1. G에서 주어진 데이터와 유사한 FAKE Data를 생성.
2. D는 G에서 전달된 데이터가 FAKE인지 REAL인지 구분

- G는 저차원 무작위 잡음을 입력받아 고차원의 FAKE Image를 생성한다.
- Real Image를 학습하여 실제이미지와 확률분포가 최대한 비슷하도록 FAKE Image를 만드는 것.
- 이 과정에서 D의 출력을 활용한다. ( 판별자를 더 잘 속일 수 있도록 학습하는 것 )

- D는 입력된 이미지가 Real인지 Fake인지를 구분함
- Real Image는 변하지 않지만 Fake Image는 G의 학습이 진행되멩 따라 점점 Real과 유사해진다.
- 따라서 D는 fake와 real을 판별할 수 있도록 점진적으로 학습을 진행한다.
<img src="https://user-images.githubusercontent.com/70633080/117790545-6c758380-b284-11eb-9337-51d728fd7226.png" width=50% height=50%>
<img src="https://user-images.githubusercontent.com/70633080/117790577-74cdbe80-b284-11eb-858b-7296d4eec5e7.png" width=50% height=50%>
- real data 일부를 D에 입력(배치처리)
- 미분가능한 판별함수D가 출력을 1로 하도록 학습
- 실제데이터의 확률분포와 다른 임의의 확률분포를가진 무작위 잡음을 생성
- 생성된 무작위 잡음을 미분가능함 생성함수 G에 통과시킴 
- G가 생성한 데이터를 추출
- 추출된 생성데이터를 D에 입력
- 미분가능한 판별함수D가 출력을 0으로 하도록 학습

- G학습 시 D는 학습이 되지 않도록 동결(가중치고정)하는것이 중요함.
- 최적화가 끝나고나면 이론적으로는 G의 결과와 real image를 D가 판별하지 못하게된다.
- 이를 위해서는 각 G와 D가 최적으로 구성되고 둘의 밸런스가 잘 맞아야 한다.

## 2. 확률분포 생성을 위한 완전연결계층 GAN구현
- 처음 제안된 GAN논문에 게재된 예제를 구현해본다. 
- 이 예제는 GAN으로 정규분포를 생성한다.
- 생성에 사용하는 무작위 잡음벡터 Z는 균등분포확률신호인데 출력은 정규분포확률신호이다.
### 2-1. 패키지 임포드
```
import numpy as np
import matplotlib.pyplot as plt

from keras import models
from keras.layers import Dense, Conv1D, Reshape, Flatten, Lambda
from keras.optimizers import Adam
from keras import backend as K
```
### 2-2. Data 생성
- 데이터를 생성하는 클래스
```
class Data:
  def __init__(self,mu,sigma,ni_D):
    self.real_sample=lambda n_batch: np.random.normal(mu,sigma,(n_batch,ni_D))
    self.in_sample=lambda n_batch : np.random.rand(n_batch,ni_D)
```
- GAN에는 2가지 데이터가 필요하다.
1. GAN으로 흉내내고자 하는 실제데이터
2. 실제데이터와 통계적 특성이 다른 무작위잡음 데이터
- 이 둘을 만들기 위해서는 확률변수를 생성하는 함수가 필요하다.
- 정규분포 확률변수는 numpy아래의 random.normal()함수로 생성. 이를 lambda로 만들어 반환.
- 이를 통해 추후 원하는 수만큼 확률변수를 만들 수 있다.
- argument 확률은 random.rand()를 사용해 연속균등분포로 지정한다.

### 2-3. 머신구현하기
- 머신은 데이터와 모델로 GAN을 학습하고 성능을 평가하는 인공신경망 전체를 총괄하는 객체이다.
- __init__() : 클래스 초기화함수
- run() : 실행 멤버함수
- run_epochs() : 에포크 단위 실행멤버함수
- train() : 학습진행멤버함수
- train_epoch(): 매순간 학습진행멤버함수
- train_D() : 판별망 학습멤버함수
- train_GD() : 학습용 생성망 학습멤버함수
- test_and_show() : 성능평가 및 그래프그리기 멤버함수
- print_stat() : 상황출력정적함수

- GAN이 임의의 통계특성을 지닌 정규분포를 생성하도록 평균값과 표준편차를 4와 1.25로 설정합니다.
```
class Machine():
  def __init__(self,n_batch=10,ni_D=100):
    data_mean=4
    data_stddev=1.25
    self.data=Data(data_mean,data_stddev,ni_D)
    self.gan=GAN(ni_D=ni_D, nh_D=50,nh_G=50)
    self.n_batch=n_batch
    self.n_iter_G=1
    self.n_iter_D=1
```
- D가 한번에 받아들일 확률변수 수 (ni_D)를 100개로 설정 
  - 잠재벡터(잡음)길이가 100이라는것.
- GAN을 구성하는 2가지 신경망인 G와D 은닉계층의 노드수를 모두 50으로 설정
- 배치단위를 설정(n_batch)
- G와D의 배치별 최적화 횟수결정(n_iter)
- G와D의 각 배치마다 에포크를 다르게 가져갈수도있다.
  - 기본은 한번배치가 수행될떄 D한번, G한번이다.
  - GAN을 처음제안한 논문에서는 배치별로 D를 G보다 더 많이 학습하면 최적화에 도움이 된다고 언급되어있다.
  - 이는 하이퍼파라미터로 설정가능(n_iter)
- 다음으로 머신클래스의 실행을 담당하는 run()를 만든다.
```
def run(self,n_repeat=30000//200, n_show=200,n_test=100):
  for ii in range(n_repeat):
    print('stage',ii,'(epoch: {})'.format(ii*n_show))
    self.run_epochs(n_show,n_test)
    plt.show()
```
- run_epochs는 호출될때마다 학습을 n_show번 수행한다.
```
def run_epochs(self,epochs,n_test):
  self.train(epochs)
  self.test_and_show(n_test)
```
- 이 함수는 epochs만큼 학습을 진행하는 함수를 호출 후
- 학습된 신경망에 내부 성능 평가 데이터를 넣어서 그 성능을 결과그래프로 보여주는 함수를 호출
- 다음으로 GAN의 학습을 진행하는 함수를 만든다.
```
def train(self,epochs):
  for epoch in range(epochs):
    self.train_each()
```
- 이 함수는 매 에폭마다 train_each()를 호출해 학습한다. 
- D가 약간 진화되면 G는 이에 맞추어 자신을 좀 더 진화시킨다. (때로는 D가 G보다 더 진화할 수도 아닐수도 있음)
- 예제에서는 D와G가 한번씩 학습되도록 하였다.
```
 def train_each(self):
  for it in range(self.n_iter_D):
    self.train_D()
  for it in range(self.n_iter_G):
    self.train_GD()
```
- train_GD는 D의 결과를 피드백받아 G를 학습시키는 과정이다.
- D와 G는 각각 n_iter만큼 학습 (예제에서는 1)
```
def train_D(self):
  gan=self.gan
  n_batch=self.n_batch
  data=self.data
  
  Real=data.real_sample(n_batch)
  Z=data.insample(n_batch)
```
- 실제데이터에서 n_batch만큼 샘플을 가져온다 (정규분포를 따르는 데이터)
- 입력샘플의 분포를 균등분포로 정함(Z) 

```
  GAN=gan.G.predict(Z)
  gan.D.trainable=True
  
  gan.D_train_on_batch(Real,Gen)
```
- 입력샘플Z를 G에 통과시켜 생성망의 출력으로 바꿔준다.
- D는 GD(학습용생성망)을 사용할때 학습이 되지않도록 막아두기 때문에 D를 훈련시킬때는 gan.D.trainable을 True로 바꾸고 진행해야한다.
- 그리고 D를 학습시킨다. 
- 다음은 GD(학습용 생성망)을 학습시키는 함수이다.
```
def train_GD(self):
  gan=self.gan
  n_batch=self.n_batch
  data=self.data
  Z=data.in_sample(n_batch)
  
  gan.D.trainable=False
  gan.GD_train_on_batch(Z)
```
- n_batch만큼의 임의의 분포입력 샘플을 만든다.(Z)
- 이 입력이 G에 들어가면 모든 D는 실제샘플로 착각하도록 GD_train_on_batch를 이용해 학습한다.
- GD를 학습할때에는 실제데이터를 다룰 필요가 없기 때문에 D보다는 코드가 단순하다.
- 다음으로 GAN의 성능을 평가하고 확률예측결과를 그래프로 그리는 멤버함수를 만든다.
```
def test_and_show(self,n_test):
  data=self.data
  Gen,Z=self.test(n_test)
  Real=data.real_sample(n_test)
  self.show_hist(Real,Gen,Z)
  Machine.print_stat(Real,Gen)
```
- 우선 무작위잡음Z를 G의 입력으로 얻은 fake data Gen을 출력한다.
- 실제이미지를 n_test만큼 가져와서 Real에 저장한다.
- print_stat : 데이터의 통계적 특성을 텍스트로 표시한다.
```
def show_hist(self,Real,Gen,Z):
  plt.hist(Real.reshape(-1),histtype='step',label='Real')
  plt.hist(Gen.reshape(-1),histtype='step',label='Generated')
  plt.hist(Z.reshape(-1),histtype='step',label='Input')
  plt.legend(loc=0)
```
- 학습진행경과에 대한 그래프를 그리는 함수이다.
- 통계적 특성을 plt.hist()를 사용해 표시
- plt.legend()는 그래프의 세 선들을 구분하는데 사용한다.
- 다음으로 G가 얼마나 실제데이터의 확률분포를 따르는 데이터를 만드는지 확인하기 위해 정적멤버함수를 만든다.
```
@staticmethod
def print_stat(Real,Gen):
  def stat(d):
    return (np.mean(d),np.std(d))
  print('mean and std of real',stat(Real))
  print('mean and std of fake',stat(Gen))
```
- 이함수는 클래스의 멤버변수나 멤버함수를 사용하지 않으므로 정적멤버함수로 지정하였다.
- stat()은 벡터의 평균과 분산을 계산한다.

### 2-4. GAN모델링
- 다음 순서로 GAN을 모델링한다.
1. 클래스초기화 __init__()
2. D구현함수 gen_D()
3. G구현함수 gen_G()
4. 학습용 생성망 make_GD()
5. D 학습함수 D_train_on_batch()
6. G 학습함수 G_train_on_batch()

- 초기화 함수
```
class GAN:
  def __init__(self,ni_D,nh_D,nh_G):
    self.ni_D=ni_D
    self.nh_D=nh_D
    self.nh_G=nh_G
    
    self.D=self.gen_D()
    self.G=self.gen_G()
    self.GD=self.make_GD()
```
- ni_D : 판별망의 입력길이
- nh_D : 판별망의 두 은닉계층의 노드 수
- nh_G : 생성망의 두 은닉계층의 노드 수

- 다음으로 판별망D를 구현하는 함수를 만든다.
- 입력 -> lambda (2 * ni_D) -> Dense(nh_D,ReLU) -> Dense(nh_D,ReLU) -> Dense(1,sigmoid) -> 출력
- 두 은닉계층과 출력계층은 모두 fc layer로 구성
```
def gen_D(self):
  ni_D=self.ni_D
  nh_D=self.nh_D
  D=models.Sequential()
  D.add(Lambda(add_decorate,output_shape=add_decorate_shape,input_shape=(ni_D,)))
```
- 입력신호를 변형하는 계층을 케라스에서 제공하는 람다클래스를 이용해 만들 수 있다.
- Lambda class는 계층의 동작을 처리하는 add_decorate()함수와 계층을 통과한 출력텐서의 모양을 입력받는다.
```
  D.add(Dense(nh_D,activation='relu'))
  D.add(Dense(nh_D,activation='relu'))
  D.add(Dense(1,activation='sigmoid'))
  model_compile(D)
  return D
```
- 은닉계층인 두 Dense()는 nh_D만큼의 노드로 구성되어있다.
- 다음으로 람다계층의 처리함수인 add_decorate()을 만든다.
```
def add_decorate(x):
  m=K.mean(x,axis=-1,keepdims=True)
  d=K.square(x-m)
  return K.concatenate([x,d],axis=-1)
```
- 이 합수는 입력벡터에 새로운 벡터를 추가한다. 
- 새로운 벡터는 입력벡터의 각요소에서 벡터평균을 뺀 값을 자승한 값을 가진다.(x-m)** 2 
- 벡터 추가는 K.concatenate()로 구현한다. 
- axis=-1 : 마지막차원을 서로 붙이라는  argument
- 다음으로 출력데이터의 모양을 지정하는 add_decorate_shape(input_shape)을 만든다.
```
def add_decorate_shape(input_shape):
  shape=list(input_shape)
  assert len(shape)==2
  shape[1] *= 2
  return tuple(shape)
```
- 이 함수는 input_shape을 입력받아 람다계층의 처리함수가 돌려주는 출력벡터의 크기를 설정하는 형태이다.
- 모델을 컴파일하는 model_compile()을 구현한다.
```
lr=2e-4
adam=Adam(lr=lr,beta_1=0.9,beta_2=0.999)
def model_compile(model):
  return model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])
```

- 다음으로 G를 구현하는 함수를 만든다.
- 입력 -> reshape(ni_D,1)->Conv1d(nh_Gx(1))->Conv1D(nh_Gx(1))->Conv1D(1 * (1))->Flatten->출력
```
def gen_G(self):
  ni_d=self.ni_D
  nh_G=self.nh_D
  G=models.Sequential() #(batch,ni_D)
  G.add(Reshape((ni_D,1),input_shape=(ni_D,)))
  G.add(Conv1D(nh_G,1,activation='relu'))
  G.add(Conv1D(nh_G,1,activation='sigmoid'))
  G.add(Conv1D(1,1))
  G.add(Flatten())
```
- G에 들어가는 입력데이터의 모양은 (batch,input_dim)이다.
- 1차원 Conv에 입력을 넣으려면 (batch,steps,input_dim)으로 데이터차원을 확대해야한다. 
- 차원을 확대한 후 두 1차원합성곱계층을 거쳐 확률변수의 확률분포를 바꿔준다.
- 필터수 nh_G를 늘리게되면 입력신호의 범위를 좀 더 세분화해서 처리할 수 있다. 
- 1은 커널크기를 말하며 입력벡터간 상관도를 높여주는 역할을 한다.
- activation 함수는 정말한 조정을 위해 한번은 시그모이드로 설정 
- 마지막은 출력이 하나여야 한다. 따라서 Conv1D(1,1)
- 두번쨰 합성곱 계층에 의한 변환이 끝나면 1차원벡터로 복구한다.
- 추후 판별망은 완전연결계층으로 판별하기 때문에 생성망의 출력을 1차원으로 바꾸는 것이다. 

- 케라스에서는 입력데이터의 모양을 1차원으로 지정해도 내부에서는 2차원으로 동작한다. (배치, 데이터수)
- 따라서 실제로 케라스에서는 첫번째차원을 지정하지않아도 된다.(배치)
- 다만, 고급기능인 람다계층이나 손실함수에서 K.mean(), K.square()같은 함수에서는 배치크기를 지정해야한다.
```
model_compile(G)
return(G)
```
- 모델링이 끝나면 컴파일한다.

- 학습용생성망을 구현하는 함수를 만든다.
- 이는 G의 상단에 D를 달아주어 구현한다.
- 이때, D의 가중치는 동결시켜야한다
```
def make_GD(self):
  G,D=self.G,self.D
  GD=models.Sequential()
  GD.add(G)
  GD.add(D)
  
  D.trainable=False
  model_compile(GD)
  D.trainable=True
  return GD
```
- GD에 G를 add한 후 D를 add한다. 순서상으로는 상단에 D가 위치한다. 

- 다음으로 D의 학습을 진행하는 함수를 만든다.
```
def D_train_on_batch(self,Real,Gen):
  D=self.D
  X=np.concatenate([Real,Gen],axis=0)
  y=np.array([1]*Real.shape[0]+[0]*Gen.shape[0])
  D.train_on_batch(X,y)
```
- 데이터와 레이블을 만들고 D.train_on_batch()를 이용해 학습일 진행, 손실값을 계산한다.
- train_on_batch는 fit()과는 처리하는 데이터 양이 다르다. 
- fit은 전체를 받아 배치단위로 반복학습
- train_on_batch는 배치만큼 받아 1회만 학습
- 다음으로 학습용 생성망을 학습시키는 멤버함수를 만든다.
```
def GD_train_on_batch(self,Z):
  GD=self.GD
  y=np.array([1]*Z.shape[0])
  GD.train_on_batch(Z,y)
```
- 허구값을 D에서 실제로 판별하도록 학습해야하기 때문에 목표 출력값을 모두 1로 설정했다.

## 3. 필기체를 생성하는 합성곱계층 GAN구현
- 입력한 필기체를 보고 배워 새로운 유사 필기체를 만든다. 
- GAN에 들어있는 두 신경망은 합성곱계층을 이용해 만든다. 
```
from keras.datasets import mnist
from PIL import Image
import numpy as np
import math, os
import keras.backend as K
import tensorflow as tf
```
- 이번 예시처럼 신경망의 출력이 스칼라나 벡터가 아닌 다차원일 경우 해당 차원에 맞는 손실함수가 필요하다.
- 4차원 데이터를 이용하는 손실함수를 keras backend와 tensorflow로 구현한다.
```
def mse_4d(y_true,y_pred):
  return K.mean(K.square(y_pred-y_true),axis(1,2,3))
def mse_4d_tf(y_true,y_pred):
  return tf.reduce_mean(tf.square(y_pred - y_true), axis(1,2,3))
```
- 평균제곱오차를 구했다. 4차원데이터이므로 0축을 제외하고는 평균계산이 다른 모든축에 대해 이루어지도록 하였다.

### 3-1. 합성곱계층 GAN수행
- 매개변수를 효율적으로 입력받을 수 있도록 argparse 를 사용한다.
```
import argparse
def main():
  parser=argparse.ArgumentParser()
  parser.add_argument('--batch_size',type=int,default=16)
  parser.add_argument('--epochs',type=int,default=1000)
  등
```
```
args= parser.parse_args()
train(args)
```
### 3-2. 합성곱계층 GAN모델링
```
from keras import models, layers, optimizers
class GAN(models.Sequential):
  def __init__(self,input_dim=64):
    super().__init__()
    self.input_dim=input_dim
    
    self.generator=self.GENERATOR()
    self.discriminator=self.DISCRIMINATOR()
    
    ## 학습용생성망( 생성망+판별망)
    self.add(self.generator)
    self.discriminator.trainable = False
    self.add(self.discriminator)
    self.compile_all()
   
   def compile_all():
    d_optim=optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim=optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)
    
    self.compile(loss='binary_crossentropy',optimizer=g_optim)
    self.generator.compile(loss=mse_4d,optimizer='SGD')
    self.discriminator.trainable=True
    self.discriminator.compile(loss='binary_crossentropy',optimizer=d_optim)
   
   def GENERATOR(self):
    
    input_dim=self.input_dim
    
    model=models.Sequential()
    model.add(layers.Dense(1024, activation='tanh',input_dim=input_dim)) # 입력이 1차원행렬
    model.add(layers.Dense(128 * 7 * 7 ,activation='tanh'))
    model.add(layers.BatchNormalization())
    model.add(layers.Reshape((128,7,7),input_shape=(128 * 7 * 7,)))
    model.add(layers.UpSampling2D(size=(2,2))) # (128,14,14)
    model.add(layers.Conv2D(64,(5, 5),padding='same',activation='tanh')) # (64,?,?)
    model.add(layers.UpSampling2D(size=(2,2)))
    model.add(layers.Conv2D(1,(5,5),padding='same',activation='tanh'))
    
    return model
    
   def DISCRIMINATOR(self):
    
    model=models.Sequential()
    model.add(layers.Conv2D(64,(5,5),padding='same',activation='tanh',input_shape=(1,28,28)))
