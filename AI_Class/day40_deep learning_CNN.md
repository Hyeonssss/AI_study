# day40_deep learning_CNN



#### Fashion-MNIST tensorflow 구현

- back propagation: loss로 값을 계산했는데 불만족할 시 재학습하는 알고리즘

###### input layer

- 데이터의 shape을 변경, convolution 가기 위해선 4차원으로 변경해야함



###### convolution layer

- W, conv

  kernel(filter)의 크기와 filter의 개수, stride

- ReLU

  0 미만 값을 절사하는 relu함수로 activation 시킴

###### pooling layer

- max pooling, ksize, stride

  데이터 크기 값을 줄이고 특징을 강조

###### Flatten

- 1차원으로 변경, DNN의 input layer 역할을 함

###### hidden layer



###### output layer

- softmax로 activation 시키는 것이 우리의 예측값

![image-20210325101125555](md-images/image-20210325101125555.png)



##### TF2_MNIST

```python
# 사용하는 모듈 import
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.layers import MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

## 1. Raw Data Loading ##
df = pd.read_csv('/content/drive/MyDrive/Machine Learning Colab/Fashion-MNIST/fashion-mnist_train.csv')
display(df.head(), df.shape)

## 2. 결측치, 이상치 처리 ##
##    우리데이터는 결측치와 이상치가 없어요!! ##

## 3. 데이터 확인(이미지인 경우) ##
img_data = df.drop('label', axis=1, inplace=False).values

fig = plt.figure()
fig_arr = list()

for n in range(10):
    fig_arr.append(fig.add_subplot(2,5,n+1))
    fig_arr[n].imshow(img_data[n].reshape(28,28), cmap='gray')
    
plt.tight_layout()
plt.show()

## 4. Data Split(Train Data와 Test Data를 분리) ##
x_data_train, x_data_test, t_data_train, t_data_test = \
train_test_split(df.drop('label', axis=1, inplace=False),
                 df['label'],
                 test_size=0.3,
                 random_state=0)

## 5. 정규화(Normalization) ##
scaler = MinMaxScaler()
scaler.fit(x_data_train)
x_data_train_norm = scaler.transform(x_data_train)
x_data_test_norm = scaler.transform(x_data_test)

## 데이터 전처리 끝!! ## 
```

![image-20210325172904246](md-images/image-20210325172904246.png)



##### 모델 구현

```python
## Tensorflow 2.x 구현 ##

model = Sequential()

# Conv2D (필터의 개수, filter의 크기, activation)
model.add(Conv2D(filters=32,
                 kernel_size=(3,3),
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,
                 kernel_size=(3,3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,
                 kernel_size=(3,3),
                 activation='relu'))

model.add(Flatten())
model.add(Dense(units=256,
                activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=10,
                activation='softmax'))

print(model.summary())

model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

history = model.fit(x_data_train_norm.reshape(-1,28,28,1),
                    t_data_train,
                    epochs=200,
                    batch_size=100,
                    verbose=1,
                    validation_split=0.3)
```

![image-20210325172946235](md-images/image-20210325172946235.png)



##### 예측

```python
model.evaluate(x_data_test_norm.reshape(-1,28,28,1),
               t_data_test)
```

![image-20210325173021704](md-images/image-20210325173021704.png)