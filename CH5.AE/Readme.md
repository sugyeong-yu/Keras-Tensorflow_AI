# Chapter5. AE(오토인코더)
- 오토인코더는 비지도학습 인공지능이다.
- 앞의 ANN,DNN,CNN,RNN은 목푯값을 통해 학습하는 지도학습이다.
- 비지도 학습은 입력데이터를 가공해 목표값을 출력하는 것이 아니다.
- AE의 목적은 입력데이터의 특징점을 효율적으로 찾는 것이다.

## 1. AE원리
- 오토인코더는 입력한 데이터를 부호화한후 복호화하는 신경망이다. 
- 결과로 입력데이터의 특징점을 추출한다.
- 주로 데이터압축, 저차원화를 통한 데이터관계관찰, 배경잡음억제 등이다.
- AE는 **주성분분석(PCA)** 로 처리하는 일차원 데이터 처리방식을 딥러닝방식으로 확장한 것이다. 
- 따라서 데이터구성이 복잡하거나 대량인 경우 PCA보다 효과적이다.
- AE의 동작은 부호화과정과 복호화과정으로 이루어진다.
- 부호화과정
  - 입력계층에 들어온 다차원데이터는 차원을 줄이는 은닉계층으로 들어간다.
  - 은닉계층의 출력이 곧 부호화결과이다.
- 복호화과정
  - 은닉계층에서 출력한 부호화결과는 출력계층으로 들어간다.
  - 이때, 출력계층의 노드수는 은닉계층의 노드수보다 많다. 
  - 출력계층은 입력계층과 노드수가 동일하다.
  
## 2. 완전연결계층을 이용한 AE구현
- 필기체 숫자 MNIST로 AE를 구현해본다.
- AE를 구성하는 계층이 완전연결계층이라 가정한다.
- AE가 필기체숫자를 차원이작은데이터로 부호화한뒤 원래 이미지와 유사하게 다시 복호화 할수있는지 보는게 포인트!!

### 2.1 모델링
```
from keras import layers,models
class AE(models.Model):
  def __init__(self,x_nodes,z_dim):
    x_shape=(x_nodes,)
```
- 초기 값으로 입력노드수(x_nodes), 은닉노드수(z_dim)을 지정한다.
- AE의 입력계층, 운닉계층, 출력계층을 정의한다.
```
     x = layers.Input(shape=x_shape)
     z = layers.Dense(z_dim, activation='relu')(x)
     y = layers.Dense(x_nodes, activation='sigmoid')(z)
```
- 입력과 출력계층의 노드수를 같도록 구성하였다.
- 은닉계층의 활성화함수는 relu로 설정.
- 출력계층의 활성화함수를 sigmoid로 한 이유는 이미지의 특성을 반영하기 위해서이다.
- MNIST이미지는 검정 또는 흰색으로 구성되어있다. 만약 픽셀들이 흰색과 검정사이의 회색으로 골고루 분포하는 경우 선형함수가 더 적합하다.
- AE의 구조를 확정하고 컴파일한다.
```
    super().__init__(x, y)
    self.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
```
- 손실함수를 binary_crossentropy로 하였다. 이는 분류문제에 주로사용되지만 AE용으로도 사용했다.
- AE는 부호화와 복호화가 자동으로 수행된다.
- 부호화가 어떻게 진행되는지 알고 싶다면 은닉계층의 결과를 출력하는 별도 부호화 모델을 설정해 신경망 외부에서 부호화결과를 확인할 수 있다.
```
self.x = x
self.z = z
self.z_dim = z_dim
self.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

def Encoder(self):
        return models.Model(self.x, self.z)
```
- 부호화과정까지만 모델로 컴파일하여 이를 Return
- 외부에서 새로운 부호화데이터를 넣어 복호화 결과를 얻는방법도 제공할 수 있다.
```
def Decoder(self):
        z_shape = (self.z_dim,)
        z = layers.Input(shape=z_shape)
        y_layer = self.layers[-1]
        y = y_layer(z)
        return models.Model(z, y)
```
- 외부에 새롭게 들어오는 입력에 대응하므로 새로운 입력을 Input()으로 만든다.
- self.layers[-1]은 제일 마지막부분 즉, AE자신의 출력계층을 의미한다.

### 2.2 데이터준비
- 사용할 MNIST데이터를 케라스의 서브패키지로 가져온다.
```
from keras.datasets import mnist
(x_train,)(x_test,)=mnist.load_data()
```
- AE에서는 레이블 정보가 사용되지 않기때문에 _ 로 대체한다.
- 다음으로 0~255의 값을 갖는 이미지를 1이하가 되도록 정규화한다.
```
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255
```
- DNN 완전연결계층구조에 적합하도록 argument를 3차원에서 2차원으로 확대한다.
```
x_train=x_train.reshape((len(x_train),-1))
x_test=x_test.resape((len(x_test),-1))
```
- 데이터의 shape은 (이미지 수, 이미지의길이) 를 나타낸다.

### 2.3 완전연결계층 AE 학습
- 주요 파라미터들과 AE모델의 인스턴스를 정의한다.
```
x_nodes=784
z_dim=36
autoencoder=AEA(x_nodes,z_dim)
```
- 학습은 fit()으로 수행한다.
```
 history = autoencoder.fit(X_train, X_train,
                              epochs=10,
                              batch_size=256,
                              shuffle=True,
                              validation_data=(X_test, X_test))
```
- 입력과 출력을 모두 x_train으로 설정하여 학습한다.
- 총 10회 학습하며 1회 배치마다 데이터 256개를 process에 보내도록 설정.

### 2.4 학습 효과 분석
```
from keraspp.skeras import plot_loss,plot_acc
import matplotlib.pyplot as plt
```
- plot_loss()와 plot_acc()는 학습결과를 그래프로 표시하는 함수이다.
## 3. 합성곱 계층을 이용한 AE구현
