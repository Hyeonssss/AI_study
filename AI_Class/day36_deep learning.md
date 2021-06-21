## day36_deep learning



### deep learning

- 1개의 logistic regression을 node라고 표현하고

  (input layer, hidden layer, output layer)

- 굉장히 많은 로지스틱들이 계층구조를 이룸



##### AI

- DNN을 구축(deep neural network)

- "XOR" - 학습이 안돼요

  single-layer perceptron으로는 절대 안돼

  multi-layer perceptron으로 학습이 가능 -> 근데 적절한 학습 방식을 찾을 수 없어

  AI(neural network)가 망해버림

- 1974 ' 웨어보스' 획기적인 논문 발표, 관심 못 받음

  1982 다시 발표, 관심 못 받음

  1986 'Hinton'이 웨어보스의 논문 재발견

  - 오차역전파(back propagation)

- 1990년대 AI

  "Back propagation" (오차 역전파)

  만약 layer 개수가 증가하면 'Vanishing gradient' 현상 발생

  AI(neural network)의 침체기

- 이후 machine learning
  - Regression
  - KNN
  - SVM
  - Naive Bayes
  - Decision Tree
  - Neural Network

- 2006 CIFAR

  초기화, Activation Function

- 딥러닝에선 시그모이드 안씀



##### overfitting (과대적합)

- 'Training data'에 대해 너무 적합하게 학습이 된 경우

- 실제 데이터 예측에서 오히려 정확도가 떨어지는 경우

- overfitting을 방지하려면

  1. 데이터량이 많아야 함

  2. 필요없거나 중복이 되는 feature을 삭제

  3. L2 정규화 (인위적으로 W의 값을 조절)

  4. DNN에서는 'dropout'이라는 방법을 사용

     학습에 참여하는 각 layer 안의 일정 node를 사용하지 않는 기법



##### 코랩에서 MNIST구현 TF 1.15

```python
# MNIST DNN 구현 (TF 1.15)

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# raw data loading
df = pd.read_csv('/content/drive/MyDrive/machine learning colab/MNIST/MNIST/train.csv')

display(df.head(), df.shape)

# 결측치 이상치는 없어용

# 이미지 확인
img_data = df.drop('label', axis=1, inplace=False).values

# fig = plt.figure()
plt.imshow(img_data[5000].reshape(28,28), cmap='Greys', interpolation='nearest')
plt.show()

# train, test 데이터 분리
x_data_train, x_data_test, t_data_train, t_data_test = \
train_test_split(df.drop('label', axis=1, inplace=False), 
                 df['label'], test_size=0.3, random_state=0)

# nomalization
scaler = MinMaxScaler()
scaler.fit(x_data_train)
x_data_train_norm = scaler.transform(x_data_train)
x_data_test_norm = scaler.transform(x_data_test)

# tensorflow implementation
sess = tf.Session()
t_data_train_onehot = sess.run(tf.one_hot(t_data_train, depth=10))
t_data_test_onehot = sess.run(tf.one_hot(t_data_test, depth=10))

# tensorflow graph 그리기
# placeholder
X = tf.placeholder(shape=[None,784], dtype=tf.float32)
T = tf.placeholder(shape=[None,10], dtype=tf.float32)

# Weight & bias
# 히든 레이어가 세개 있는 것
# weight의 초기값을 랜덤으로 하면 안됨
# 특정한 방법을 이용해야 좋은 성능을 얻을 수 있음
# 1. Xavier initialization: 
#    입력의 개수와 출력의 개수를 이용해서 weight의 초기값을 결정하는 방식
# W = np.random.normal(num_of_input, num_of_output) / np.sqrt(num_of_input)
# W2 = tf.Variable(tf.random.normal([784,64]))
# 2. He's initialization: Xavier initialization의 확장 버전
# W = np.random.normal(num_of_input, num_of_output) / np.sqrt(num_of_input / 2)

# He's 초기법
# W2 = tf.get_variable('W2', shape=[784,64],
#                      initializer=tf.contrib.layers.variance_scaling_initializer())
W2 = tf.get_variable('W2', shape=[784,64], 
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random.normal([64]))
# layer2 = tf.sigmoid(tf.matmul(X,W2) + b2)
_layer2 = tf.nn.relu(tf.matmul(X,W2) + b2)
layer2 = tf.nn.dropout(_layer2, rate=0.3)

W3 = tf.get_variable('W3', shape=[64,32],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random.normal([32]))
_layer3 = tf.nn.relu(tf.matmul(layer2,W3) + b3)
layer3 = tf.nn.dropout(_layer3, rate=0.3)

W4 = tf.get_variable('W4', shape=[32,16],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random.normal([16]))
_layer4 = tf.nn.relu(tf.matmul(layer3,W4) + b4)
layer4 = tf.nn.dropout(_layer4, rate=0.3)

# output layer
W5 = tf.get_variable('W5', shape=[16,10],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random.normal([10]))

logit = tf.matmul(layer4,W5) + b5
H = tf.nn.softmax(logit)

# loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit,
                                                                 labels=T))

# train
train = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)

# 초기화
sess.run(tf.global_variables_initializer())

# epochs = 1000
for step in range(5000):
    _,loss_val = sess.run([train, loss], feed_dict={X:x_data_train_norm,
                                                    T:t_data_train_onehot})
    if step % 500 == 0:
        print(f'loss : {loss_val}')


```

![image-20210318150505944](md-images/image-20210318150505944.png)



- 예측

```python
# Accuracy
predict = tf.argmax(H, 1)
correct = tf.equal(predict, tf.argmax(T,1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

print(f'정확도 : {sess.run(accuracy, feed_dict={X:x_data_test_norm, \
                                                T:t_data_test_onehot})}')
# 정확도 : 0.8806349039077759

```



##### TF 2.x로 MNIST 구현

```python
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt

# raw data
df = pd.read_csv('/content/drive/MyDrive/machine learning colab/MNIST/MNIST/train.csv')

# 이미지 확인
img = df.drop('label', axis=1, inplace=False).values

plt.imshow(img[5000].reshape(28,28), cmap='Greys', interpolation='nearest')
plt.show()

# data split
x_data_train, x_data_test, t_data_train, t_data_test = \
train_test_split(df.drop('label', axis=1, inplace=False), 
                 df['label'], test_size=0.3, random_state=0)

# 정규화
scaler = MinMaxScaler()
scaler.fit(x_data_train)
x_data_train_norm = scaler.transform(x_data_train)
x_data_test_norm = scaler.transform(x_data_test)

# tensorflow로 구현
model = Sequential()
model.add(Flatten(input_shape=(784,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# compile
model.compile(optimizer=Adam(learning_rate=1e-2),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_data_train_norm,
                    t_data_train,
                    epochs=1000,
                    batch_size=512,
                    verbose=1,
                    validation_split=0.3)

test_loss, test_acc = model.evaluate(x_data_test, t_data_test)
print(f'test accuracy : {test_acc}')
```

![image-20210318170353958](md-images/image-20210318170353958.png)

```python
print(model.evaluate(x_data_test_norm, t_data_test))
```

![image-20210318170440924](md-images/image-20210318170440924.png)