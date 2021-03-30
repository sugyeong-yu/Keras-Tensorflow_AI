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
- 완전연결계층을 이용한 방법은 이미지의 위치별로 각각 다르게 처리하기 때문에 비효율적이다.
- 합성곱 계층방식은 비교적 가중치수가 줄어 학습 최적화에도 용이하다.
### 3.1 합성곱AE 모델링
```
from keras import layers, models

def Conv2D(filters, kernel_size, padding='same', activation='relu'):
    return layers.Conv2D(filters, kernel_size, padding=padding, activation=activation)
```
- Conv2D(), Maxpooling2D(), Upsampling2D(), Dropout() 을 사용한다.
- 다음으로 AE의 객체를 구성한다.
```
class AE(models.Model):
    def __init__(self, org_shape=(1, 28, 28)):
        # Input
        original = layers.Input(shape=org_shape)
```
- MNIST의 경우 입력계층에 들어가는 데이터모양은 채널정보의 위치에 따라 (1,28,28) 또는 (28,28,1) 이다.
- 채널의 위치는 케라스 설정에서 할 수 있다.
- 합성곱AE는 총 7개의 합성곱계층으로 구성된다.
- 1~3 번째까지는 부호화를 위한 합성곱계층
- 4~6 번째까지는 복호화를 위한 합성곱계층
- 7 번쨰는 부호화 및 출력을 하는 합성곱 계층이다.
- 2차원 이미지를 출력으로 하기 때문에 마지막 층도 합성곱계층으로 구성한다.
- 첫번째 합성곱계층은 4개의 3 * 3 필터로 구성된다.
```
 # encoding-1
        x = Conv2D(4, (3, 3))(original)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
```
- 합성곱계산뒤 맥스풀링이 적용된다.
- 이미지의 크기는 4분의 1로 줄어들고 개수는 4배가 되어 14 * 14 이미지가 4장 출력된다.
- 두번째로 3 * 3 필터 8개를 적용한다.
```
        # encoding-2
        x = Conv2D(8, (3, 3))(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
```
- 이 계층을 지나면 7 * 7 이미지가 8장 출력된다.
- 세번째 합성곱 계층은 부호화결과를 출력한다.

```
        # encoding-3: encoding output: 7x7 pixels
        z = Conv2D(1, (7, 7))(x)
```
- 부호화가 마무리되며 28 * 28 크기의 입력이미지한장이 7 * 7 크기의 이미지한장으로 바뀌게 된다.
- 이제 복호화를 통해 7 * 7 의 부호화이미지가 28 * 28 입력이미지로 어떻게 돌아가는지 살펴본다.
- 먼저, 필터 수를 늘려 전체 이미지 공간을 늘린다.
- 같은 크기의 이미지가 16장이 되도록 한다.
- 그리고나서 이미지의 샘플링 주파수를 좌우 두배씩 올린다.
```
       # decoding-1
        y = Conv2D(16, (3, 3))(z)
        y = layers.UpSampling2D((2, 2))(y)
```
- Upsampling2D((2,2)) 는 x축과 y축 모두에 대해 샘플링 비율을 2배 높힌다.
- Conv2D() 는 부호화과정에서는 특징점을 찾아내는 역할을 했지만 복호화에서는 특징점을 복원하는 역할을 한다.
- 이 과정을 지나며 이미지는 14 * 14 크기의 16장이 된다.
- 다음으로 16장의 이미지들을 8장으로 줄이면서 화소를 더 정교화 한다.
```
        # decoding-2
        y = Conv2D(8, (3, 3))(y)
        y = layers.UpSampling2D((2, 2))(y)
```
- 8장의 28 * 28 크기의 이미지가 생성되었다.
- 한번더 합성곱을 수행하며 이미지 특징점들을 더 구체적으로 묘사한다.
- 조합된 이미지 수가 4개가 되도록 필터를 4개 사용한다.
```
        # decoding-3
        y = Conv2D(4, (3, 3))(y)
```
- 마지막으로 출력단계이다.
- 2차원 이미지가 별도 반환없이 신경망에서 바로 출력도도록 합성곱계층을 사용해 출력데이터를 만든다.
```
        # decoding & Output
        decoded = Conv2D(1, (3, 3), activation='sigmoid')(y)
```
- 출력계층의 활성화 함수는 시그모이드이다.
- MNIST는 흑백이미지이며 이미지에서 숫자가 위치하는 곳의 색상은 최댓값에 가깝다.
- 따라서 이진정보를 볼때 사용하는 시그모이드함수를 사용했다.
- 최종적으로 이미지는 원래 크기인 28 * 28 로 돌아간다.
