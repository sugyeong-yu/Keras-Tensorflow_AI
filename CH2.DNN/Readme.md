# Chapter2 DNN

- DNN : 은닉계층을 많이 쌓아서 만든 신경망
- 목차
  1. DNN개념
  2. 필기체를 분류하는 DNN
  3. 컬러이미지를 분류하는 DNN

# 1. DNN개념
- 다수의 은닉계층을 활용하면 하나를 활용할 때 보다 입력신호를 더 정교하게 처리할 수 있음.
- 전체 노드수가 늘어나 과적합이 될 수 있지만 최근 이를 효과적으로 해결하는 다양한 방식이 제시됨.
  - 배치정규화
  - early stopping
  - Drop out
## 1.1 경사도 소실문제와 ReLU 활성화 함수
1. 경사도 소실문제
  - 사용하는 활성화 함수에 따라 경사도 소실이 발생할 수 있다. 
  - 이로인해 경사하강법을 사용하는 오차역전파의 성능이 나빠질 수 있다.
2. ReLU활성화함수
  - DNN에서 경사도 소실문제를 극복하는 활성화 함수로 ReLU 등을 사용한다.
  - ReLU는 입력이 0보다 큰 구간에서는 직선함수이므로 값이 커져도 경사도를 구할 수 있다.\
  ![image](https://user-images.githubusercontent.com/70633080/109912436-68ab0b00-7cef-11eb-8244-894e12bd6590.png)

# 2. 필기체를 분류하는 DNN구현
- 데이터셋
  - 0~9까지로 구분된 필기체 숫자들의 모음
  - 5만개의 학습데이터와 1만개의 성능평가데이터
## 2.1 기본파라미터 설정
```
Nin = 784
Nh_l = [100,50]
number_of_class = 10
Nout = number_of_class
```
- 입력노드수는 입력이미지크기인 784개
- 출력노드수는 분류클래스수인 10개
- 은닉계층은 2개 (각각의 은닉노드수를 100과 50으로 지정)

## 2.2 DNN모델구현
- 객체지향방식으로 DNN모델링 구현
```
class DNN(models.Sequential):
  def __init__(self,Nin,Nh_l,Nout):
    super().__init__()
```
- 은닉계층과 출력계층은 모두 케라스의 layers패키지 아래 Dense()개체로 구성한다.
```
self.add(layers.Dense(Nh_l[0],activation='relu', input_shape=(Nin,), name='Hidden-1'))
self.add(layers.Dropout(0.2)
self.add(layers.Dense(Nh_l[1],activation='relu', name='Hidden-2'))
self.add(layers.Dropout(0.2))
self.add(layers.Dense(Nout,activation='softmax'))
```
- 두번째 은닉층은 이전 계층의 노드수를 적지않아도 된다. (케라스가 자동으로 설정)
- Dropout(p)는 p의 확률로 출력노드의 신호를 보낼지말지를 결정하는것. 
  - 이는 학습할때와 성능평가할때 다르게 동작한다. (케라스가 자동으로 처리)
```
self.compile(loss=categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```
- 분류클래스수가 2개 이상이므로 categorical_crossentropy를 사용함. 

## 2.3 학습 및 평가
- 최종 성능평가 손실과 정확도는 각각 0.099와 0.97로 ANN의 결과(0.109,0.97)와 매우 유사하다.
- 하지만 학습데이터가 작거나 복잡한 이미지에서는 일반적으로 DNN이 더 우수한 성능을 보인다고 알려져있다.

