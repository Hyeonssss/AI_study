# day24_machine learning_TIL





##### Tensorflow

- open source library
- numerical computation (수치연산)
- data flow graph



- tf.constant(10, dtype=tf.float32) : 상수 생성
- tf.placeholder(dtype=tf.float32) : 변수 생성, 값을 입력 받기
- tf.placeholder(shape=[None,1], dtype=tf.float32) : None = 값이 없다는게 아니라 몇 개든 상관 없음

```python
# Tensorflw를 이용해서 Simple Linear Regression을 구현
import numpy as np
import pandas as pd
import tensorflow as tf

# 1. training data set
x_data = (np.array([1,2,3,4,5])).reshape(5,1)
t_data = (np.array([3,5,7,9,11])).reshape(5,1)  # lable

# 2. placeholder
X = tf.placeholder(shape=[None,1], dtype=tf.float32)
T = tf.placeholder(shape=[None,1], dtype=tf.float32)
# None = 값이 없다는게 아니라 몇 개든 상관 없다는 뜻

# 3. Weigth & bias ( y = Wx + b ) => y = X dot W + b
W = tf. Variable(tf.random.normal([1,1]), name='weight')
b = tf. Variable(tf.random.normal([1]), name='bias')

# 4.Hypothesis or predict model
H = tf.matmul(X,W) + b   # y = Wx + b => 2차원 행렬로 처리 => y = X dot W + b

# 5. W, b를 구하기 위해 평균제곱오차를 이용한 최소제곱법으로 loss function 정의
loss = tf.reduce_mean(tf.square(H - T))

# 6. train
train = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(loss)

# 7. session & 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())  # 초기화 (2.x 버전에선 삭제됨)

# 8. 학습을 진행
# 반복학습을 진행 ( 1 epoch: training data set 전체를 이용해 1번 학습하는 것 )
for step in range(30000):
    
    _, W_val, b_val, loss_val = sess.run([train,W,b,loss],
                                         feed_dict={X : x_data, T : t_data})
    if step % 3000 == 0:
        print(f'W : {W_val}, b : {b_val}, loss : {loss_val}')
        
# 9. 학습이 종료된 후 최적의 W와 b가 계산되고 이를 이용한 model이 완성
#    Prediction(예측)
result = sess.run(H, feed_dict={X : [[9]]})
print(f'예측값은 : {result}')
```

```
W : [[-0.16375786]], b : [0.8661644], loss : 55.86674880981445
W : [[1.9565796]], b : [1.1567607], loss : 0.004475906491279602
W : [[1.9842542]], b : [1.0568457], loss : 0.0005885750870220363
W : [[1.9942864]], b : [1.0206227], loss : 7.746553455945104e-05
W : [[1.9979235]], b : [1.0074916], loss : 1.0223895515082404e-05
W : [[1.999243]], b : [1.0027261], loss : 1.3538672192225931e-06
W : [[1.9997219]], b : [1.0010095], loss : 1.855204061484983e-07
W : [[1.9998908]], b : [1.0003908], loss : 2.7844873073945564e-08
W : [[1.9999406]], b : [1.0002079], loss : 7.925575751244196e-09
W : [[1.9999406]], b : [1.0002079], loss : 7.925575751244196e-09
예측값은 : [[18.999674]]
```