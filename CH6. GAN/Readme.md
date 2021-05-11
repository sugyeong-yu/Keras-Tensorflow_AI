# Chapter6. GAN
- GAN은 경쟁적 학습방법을 이용하는 생성형 인공지능

## 1. GAN의 원리
- GAN은 실제데이터와 비슷한 확률분포를 가지는 fake data를 만들어낸다. 
- fake data는 생성데이터 라고도 한다.
- GAN은 레이블이 없는 정보를 다루는 비지도 학습 방법이다.
- 입력데이터 : 무작위 잡음
- 출력데이터 : 입력데이터보다 높은 차원의 특정 분포도를 갖는 데이터

## 1-1. GAN의 구조
- 생성망(Generator)와 판별망(Discriminator) 으로 구성되어 있다.
- Generator : 실제데이터와 확률분포상 유사한 데이터를 만듬
- Discriminator : 실제데이터인지 생성데이터인지 구분
- Discriminator는 진짜와 가짜를 더 잘 구별하도록 학습함. 
- Generator는 Discriminator를 더 잘속이도록 학습함.
- G <-> D 이 과정을 순환하며 학습 (경쟁적 학습법)

## 1-2. GAN의 동작원리
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


