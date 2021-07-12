# CH1 ANN 
ANN은 인공신경망을 총칭하는 용어로 단일 은닉계층의 ANN은 얕은신경망으로 부르기도 한다.
- 목차
  1. ANN원리
  2. 필기체를 구분하는 분류 ANN
  3. 시계열을 예측하는 회귀 ANN

## 1 ANN원리
- ANN이란? : 은닉계층을 포함하는 인공신경망 기술. 
- 입력계층, 은닉계층, 출력계층으로 구성
- 하나의 계층은 순서대로 입력노드, 은닉노드, 출력노드를 포함한다.

1. 입력신호벡터x에 가중치행렬 W_xh를 곱하여 은닉계층으로 보냄. ( 입력노드중 하나에는 외부정보가 아닌 상수 1이 항상들어온다. 이는 편향값 보상에 사용)
2. 은닉 계층의 각 노드들은 입력된 신호 벡터에 **활성화함수** f_n()을 적용한 결과벡터를 내보낸다. ( 비선형성을 보상하는 활성화함수로 보통 시그모이드, 하이퍼볼릭탄젠트 함수를 이용)
3. 은닉계층의 결과벡터에 새로운 가중치 행렬 W_hy를 곱한 뒤 출력계층으로 보냄
4. 출력 활성화함수인 f_y()를 적용하고 이 결과벡터 y를 신경망 외부로 최종 출력한다.( 분류는 활성화함수로 소프트맥스를 주로사용, 일반적으로 회귀에는 사용X)


## 2 분류ANN
: 입력정보를 바탕으로 해당 입력이 어느 클래스에 속하는지를 결정하는 문제
- 분류 ANN은 손실함수로 교차엔트로피를 사용한다. 
- 출력 노드값은 소프트맥스 연산으로 구한다. 

1. 필요한 패키지 load
``` from keras import layers, models ```
- layers : 각 계층을 만드는 모듈
- models : 만든 신경망 모델을 컴파일하고 학습시키는 역할
  - compile()
  - fit()
  - predict()
  - evaluate()
2. 파라미터설정
  - Nin :입력계층 노드수
  - Nh : 은닉계층 노드수
  - number_of_class : class수
  - Nout : 출력노드 수
3. 모델링
    1. 연쇄방식_함수형 (간단)\
    ``` model = models.Sequential()```
    - 연쇄방식은 모델구조를 정의하기 전 Sequential()로 모델을 초기화해야한다.
    ``` 
    model.add(layers.Dense(Nh,activation='relu', input_shape=(Nin,)))
    model.add(layers.Dense(Nout,activation='softmax')) 
    ```
    - 첫번째 add()단계에서 입력계층과 은닉계층의 형태가 동시에 정해진다.


    2. 분산방식_함수형 (복잡)\
    ``` x=layers.Input(shape=(Nin,))``` 
    - 입력계층은 layers.Input()함수로 지정한다. \
    ``` h=layers.Activation('relu')(layers.Dense(Nh)(x))```
    - 은닉계층은 layers.Dense()로 지정한다. 
    - x를 입력으로 받아들이도록 layers.Dense(Nh)(x)로 지정한다. \
    ``` y=layers.Activation('softmax')(layers.Dense(Nout)(h))```
    - 출력노드수는 클래스수로 지정한다.\
    ``` model=models.Model(x,y)```
    - 모델은 입력과 출력을 지정하여 만든다. \
    ``` model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy']) ```
    - 손실과 최적화함수를 지정한다.
    - metric은 학습이나 예측이 진행될때 성능검증을 위해 손실뿐아니라 정확도 즉, accuracy도 측정하라는 의미이다.
    
    3. 연쇄방식_객체지향형
    4. 분산방식_객체지향형
4. 


## 3 회귀ANN
: 입력값으로부터 출력값을 직접 예측하는 방법
- 정보벡터를 이용해 예측할 수 있도록 학습된다. 
- 손실함수로 보통 MSE 즉, 평균제곱오차를 사용한다.
