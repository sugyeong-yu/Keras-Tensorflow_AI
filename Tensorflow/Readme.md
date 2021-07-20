# Tensorflow
- tensorflow 1.5버전과 2.0버전에는 많은 문법의 차이가 존재한다.
## version 1.5
- 입력데이터 선언 시 placeholder 사용 
  - 입력데이터가 정해지지 않은 상태에서 선언만 해놓는 것.
```
X=tf.placeholder(tf.float32,[None,784]) # None은 배치사이즈 , None으로 설정시 텐서플로가 알아서 계산해서 적용
Y=tf.placeholder(tf.float32,[None,10])
```
- 노드 생성 시 tf.Variable사용 
```
W1=tf.Variable(tf.random_normal([784,256],stddev=0.01)) # 표준편차가 0.01인 정규분포를 가지는 임의의값으로 뉴런을 초기화
L1=tf.nn.relu(tf.matmul(X,W1))
```
- 모델 생성 시 연쇄방식은 동일함
```
W1=tf.Variable(tf.random_normal([784,256],stddev=0.01)) # 표준편차가 0.01인 정규분포를 가지는 임의의값으로 뉴런을 초기화
L1=tf.nn.relu(tf.matmul(X,W1))

W2=tf.Variable(tf.random_normal([256,256],stddev=0.01))
L2=tf.nn.relu(tf.matmul(L1,W2))

W3=tf.Variable(tf.random_normal([256,10],stddev=0.01))
model=tf.matmul(L2,W3)
```
- 모델 학습 전 Session을 생성하고 Session을 통해 동작하게함
```
init=tf.global_variables_initializer() # 가중치 초기화
sess=tf.Session()
sess.run(init)
```
- 모델 학습 시 keras나 torch와는 달리 model을 직접 사용하지 않음.
- optimizer-cost-model 연쇄방식으로 동작
```
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model,labels=Y))
optimizer=tf.train.AdamOptimizer(0.001).minimize(cost)
for epoch in 100:
  for i in batch:
    _,cost_val=sess.run([optimizer,cost],feed_dict={X:batch_x,Y:batch_y})
```
- 학습 중간 또는 학습 완료 후 모델의 출력을 확인하고 싶을경우 Session을 통해서 동작
```
print("result: ",sess.run(model,feed_dict={X:batch_x}))
```
- Session을 통해 동작하지않고 직접 사용하는 경우 Tensor형태로 출력 > 결과값을 한눈에 볼 수 없음 
 ```print("predict-",sess.run(tf.argmax(model,1),feed_dict={X:mnist.test.images}),"target-",tf.argmax(mnist.test.labels,1))``` \
![image](https://user-images.githubusercontent.com/70633080/126263666-0ea96684-b4e4-4a01-9531-af1fea5cc934.png)
