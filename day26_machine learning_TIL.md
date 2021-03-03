# day26_machine learning_TIL



##### Normalization (정규화)

- 비율을 이용해서  data의 scale을 조정 (0~1)
- 정규화 하는 방법
  - Min Max Normalization
  - Z-score Normalization (standardization (표준화))



###### Min-Max Normalization

- 데이터 정규화의 가장 일반적인 방법

  모든 feature의 값을 최소값 0, 최대값 1로 변환

![image-20210226092532968](md-images/image-20210226092532968.png)

- 문제점 : 이상치에 상당히 민감한 scaling 방식

  이상치(outlier) 처리가 반드시 필요



###### Standardization

> Z-score Normalization

![image-20210226100125818](md-images/image-20210226100125818.png)

- 장점: 평균과 분산을 사용하기 때문에 이상치에 크게 영향을 받지 않음
- 단점: 동일한 scale을 적용할 수 없어요



- 정규화에 대해 알게 됐으니 온도에 따른 Ozone량 학습해서 예측하는 문제를 다시 구현

  1. csv 파일 loading
  2. 결측치 제거
  3. 이상치 처리 (제거)
  4. Normalization (Min-Max)수행
  5. training data set
  6. Tensorflow를 이용해서 Machine Learning + python 구현도 포함
  7. graph를 그려서 sklearn의 결과와 비교
  8. predict해서 그 결과가 sklearn과 유사한 지 비교

  이게 Machine Learning 첫번째 수행평가

  - 해야할 일 : python 코드로 구현



##### Min-Max Scaler

- python으로 구현

```python
def min_max_normalize(lst):
    normalized = np.array([])
    
    for value in lst:
        normalized_num = (value - min(lst)) / (max(lst) - min(lst))
        normalized = np.append(normalized, np.array(normalized_num))
    
    normalized = normalized.reshape(-1,1)
    return normalized

x_normalized = min_max_normalize(x_data)
t_normalized = min_max_normalize(t_data)
```



- 라이브러리로 구현

```python
from sklearn.preprocessing import MinMaxScaler

## 정규화 처리 (Min-Max Scaler)
# Min-Max Scaler라고 불리는 객체를 생성
# 독립변수와 종속변수 각각 만들어줌

# x 변경
scaler_x = MinMaxScaler()  # 객체 생성
scaler_x.fit(training_data['Temp'].values.reshape(-1,1))
training_data['Temp'] = scaler_x.transform(training_data['Temp'].values.reshape(-1,1))

# t 변경
scaler_t = MinMaxScaler()  # 객체 생성
scaler_t.fit(training_data['Ozone'].values.reshape(-1,1))
training_data['Ozone'] = scaler_t.transform(training_data['Ozone'].values.reshape(-1,1))
display(training_data)

```





##### Simple Linear Regression



###### tensorflow로 구현

```python
# 온도에 따른 Ozone량 예측

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats          # 이상치 처리
import tensorflow as tf          # tensorflow 구현
from sklearn import linear_model # sklearn으로 simpler linear regression 
from sklearn.preprocessing import MinMaxScaler # normalization 전처리


## Raw Data Loading
df = pd.read_csv('./ozone.csv')

training_data = df[['Temp', 'Ozone']]


## 결측치처리
training_data = training_data.dropna(how='any')


## 이상치(outlier) 처리
# z-score를 이용해서 outlier 처리
zscore_threshold = 1.8

# temp에 대한 outlier(지대점)
# np.abs() : 절대값을 구해주는 함수
# ~() : boolean mask의 역의 값을 구하는 함수
tmp = ~(np.abs(stats.zscore(training_data['Temp'])) > zscore_threshold)
training_data = training_data.loc[tmp]  # 110 rows × 2 columns

# Ozone에 대한 outlier
tmp = ~(np.abs(stats.zscore(training_data['Ozone'])) > zscore_threshold)
training_data = training_data.loc[tmp]  # 103 rows × 2 columns


## 정규화 처리 (Min-Max Scaler)
# Min-Max Scaler라고 불리는 객체를 생성
# 독립변수와 종속변수 각각 만들어줌

# x 변경
scaler_x = MinMaxScaler()  # 객체 생성
scaler_x.fit(training_data['Temp'].values.reshape(-1,1))
training_data['Temp'] = scaler_x.transform(training_data['Temp'].values.reshape(-1,1))

# t 변경
scaler_t = MinMaxScaler()  # 객체 생성
scaler_t.fit(training_data['Ozone'].values.reshape(-1,1))
training_data['Ozone'] = scaler_t.transform(training_data['Ozone'].values.reshape(-1,1))
display(training_data)


## Training Data Set
x_data = training_data['Temp'].values.reshape(-1,1)
t_data = training_data['Ozone'].values.reshape(-1,1)


## placeholder
X = tf.placeholder(shape=[None,1], dtype=tf.float32)
T = tf.placeholder(shape=[None,1], dtype=tf.float32)


## Weight & bias
W = tf.Variable(tf.random.normal([1,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')


## Hypothesis
H = tf.matmul(X, W) + b


## loss function
loss = tf.reduce_mean(tf.square(H-T))


## train
train = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss)


## session, 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())


## 학습
for step in range(300000):
    _, W_val, b_val, loss_val = sess.run([train, W, b, loss],
                                         feed_dict={X: x_data, T: t_data})
    
    if step % 30000 == 0:
        print(f'W: {W_val}, b: {b_val}, loss: {loss_val}')
        
```

![image-20210226171040255](md-images/image-20210226171040255.png)



###### sklearn으로 구현

```python
# sklearn을 이용해서 구현
## Training Data Set
x_data = training_data['Temp'].values.reshape(-1,1)
t_data = training_data['Ozone'].values.reshape(-1,1)

model = linear_model.LinearRegression()
model.fit(x_data, t_data)

print(f'w: {model.coef_}, b: {model.intercept_}')

```

![image-20210226171116548](md-images/image-20210226171116548.png)



###### Graph로 비교

```python
# Graph로 비교
fig = plt.figure()
fig_tensorflow = fig.add_subplot(1,2,1)
fig_sklearn = fig.add_subplot(1,2,2)

fig_tensorflow.set_title('Tensorflow Graph')
fig_sklearn.set_title('sklearn Graph')

fig_tensorflow.scatter(x_data, t_data)
fig_tensorflow.plot(x_data, x_data*W_val.ravel() + b_val, color='r')

fig_sklearn.scatter(x_data, t_data)
fig_sklearn.plot(x_data, x_data*model.coef_.ravel() + model.intercept_, color='b')


fig.tight_layout()
plt.show()

```

![image-20210226171158344](md-images/image-20210226171158344.png)



###### 예측

```python
# prediction
# 온도가 62도 일때 Ozone량은 얼마일까

# sklearn을 이용
sklearn_result = model.predict([[62]])
print(f'sklearn_result : {sklearn_result}')     # [[49.22229492]]

# tensorflow를 이용
tensorflow_result = sess.run(H, feed_dict={X:[[62]]})
print(f'tensorflow : {tensorflow_result}')

```

![image-20210226171319623](md-images/image-20210226171319623.png)



###### python으로 구현

```python
# Simple Linear Regression을 python으로 구현

import numpy as np
import pandas as pd
from scipy import stats          # 이상치 처리
from sklearn.preprocessing import MinMaxScaler # normalization 전처리


## Raw Data Loading
df = pd.read_csv('./ozone.csv')

training_data = df[['Temp', 'Ozone']]


## 결측치처리
training_data = training_data.dropna(how='any')


## 이상치(outlier) 처리
# z-score를 이용해서 outlier 처리
zscore_threshold = 1.8

# temp에 대한 outlier(지대점)
# np.abs() : 절대값을 구해주는 함수
# ~() : boolean mask의 역의 값을 구하는 함수
tmp = ~(np.abs(stats.zscore(training_data['Temp'])) > zscore_threshold)
training_data = training_data.loc[tmp]  # 110 rows × 2 columns

# Ozone에 대한 outlier
tmp = ~(np.abs(stats.zscore(training_data['Ozone'])) > zscore_threshold)
training_data = training_data.loc[tmp]  # 103 rows × 2 columns


## 정규화 처리 (Min-Max Scaler)
# Min-Max Scaler라고 불리는 객체를 생성
# 독립변수와 종속변수 각각 만들어줌

# x 변경
scaler_x = MinMaxScaler()  # 객체 생성
scaler_x.fit(training_data['Temp'].values.reshape(-1,1))
training_data['Temp'] = scaler_x.transform(training_data['Temp'].values.reshape(-1,1))

# t 변경
scaler_t = MinMaxScaler()  # 객체 생성
scaler_t.fit(training_data['Ozone'].values.reshape(-1,1))
training_data['Ozone'] = scaler_t.transform(training_data['Ozone'].values.reshape(-1,1))
display(training_data)


## Training Data Set
x_data = training_data['Temp'].values.reshape(-1,1)
t_data = training_data['Ozone'].values.reshape(-1,1)


# 2. Weight & bias
W = np.random.rand(1,1)
b = np.random.rand(1)


# 3. Hypothesis
def predict(x):
    
    y = np.dot(x,W) + b
    
    return y


# 4. loss function
def loss_func(input_obj):
    # input_obj : [W, b]
    
    input_W = input_obj[0]
    input_b = input_obj[1]
    
    y = np.dot(x_data, input_W) + input_b
    
    return np.mean(np.power((t_data - y), 2))


# 5. 편미분을 위한 함수
def numerical_derivative(f, x):
    # f: 미분하려고 하는 다변수 함수
    # x: 모든 값을 포함하는 numpy array  ex) f'(1.0, 2.0) = (8.0, 15.0)
    delta_x = 1e-4
    derivative_x = np.zeros_like(x)     # [[0 0]
                                        #  [0 0]]
    it = np.nditer(x, flags=['multi_index'])
    
    while not it.finished:
        
        idx = it.multi_index # 현재의 iterator의 index를 추출 => tuple형태 
        
        tmp = x[idx] # 현재 index의 값을 잠시 보존,
                     # delta_x를 이용한 값으로 ndarray를 수정한 후 편미분
                     # 함수값을 계산한 후 원상복구를 해줘야 다음 독립변수에
                     # 대한 편미분을 정상적으로 수행 가능
        x[idx] = tmp + delta_x
        fx_plus_delta = f(x)    # f([1.00001, 2.0]) => f(x + delta_x)
        
        x[idx] = tmp - delta_x
        fx_minus_delta = f(x)    # f([0.99999, 2.0]) => f(x - delta_x)
        
        derivative_x[idx] = (fx_plus_delta - fx_minus_delta) / (2 * delta_x)
        x[idx] = tmp
        
        it.iternext()
    
    return derivative_x


# 6. learning rate 설정
learning_rate = 1e-4


# 7. 학습을 진행
for step in range(300000):
    # ravel() : 무조건 1차원 벡터로 만들어버리는 함수
    input_param = np.concatenate((W.ravel(), b.ravel()), axis=0) # [W b]
    derivative_result = learning_rate * numerical_derivative(loss_func,input_param)
    
    W = W - derivative_result[:1].reshape(1,1)  # W 갱신
    b = b - derivative_result[1:]               # b 갱신
    
    if step % 30000 == 0:
        print(f'W : {W}, b {b}')
        
```

![image-20210226171405465](md-images/image-20210226171405465.png)



###### python 예측

```python
print(f'python : {predict(62)}')
```

![image-20210226171535522](md-images/image-20210226171535522.png)



###### python 그래프

```python
fig = plt.figure()
fig_python = fig.add_subplot(1,2,1)

fig_python.set_title('python Graph')

fig_python.scatter(x_data, t_data)
fig_python.plot(x_data, x_data*W.ravel() + b, color='r')

fig.tight_layout()
plt.show()
```

![image-20210226171627141](md-images/image-20210226171627141.png)





##### Multiple Linear Regression

- 독립변수가 여러개

![image-20210226141048735](md-images/image-20210226141048735.png)



##### tensorflow로 구현

```python
## Multiple Linear Regression
# 온도, 태양광세기, 바람세기를 이용하여 Ozone량 예측

# %reset

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler

## Raw Data Loading
df = pd.read_csv('./ozone.csv')

training_data = df[['Temp','Wind','Solar.R','Ozone']] # 153 rows × 4 columns


## 결측치 제거
training_data = training_data.dropna(how='any') # 111 rows × 4 columns



## 이상치 처리
zscore_threshold = 1.8

for col in training_data.columns:
    tmp = ~(np.abs(stats.zscore(training_data[col])) > zscore_threshold)
    training_data = training_data.loc[tmp]
    
# display(training_data) (86, 4)


## 정규화 Normalization
scaler_x = MinMaxScaler()  # 객체 생성
scaler_t = MinMaxScaler()  # 객체 생성

scaler_x.fit(training_data[['Temp','Wind','Solar.R']].values)
scaler_t.fit(training_data['Ozone'].values.reshape(-1,1))

# training data set 정의
training_data_x = scaler_x.transform(training_data[['Temp','Wind','Solar.R']].values)
training_data_t = scaler_t.transform(training_data['Ozone'].values.reshape(-1,1))


## tensorflow 코드


## placeholder
X = tf.placeholder(shape=[None,3], dtype=tf.float32)
T = tf.placeholder(shape=[None,1], dtype=tf.float32)


## Weight & bias
W = tf.Variable(tf.random.normal([3,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')


## Hypothesis
H = tf.matmul(X, W) + b


## loss function
loss = tf.reduce_mean(tf.square(H-T))


## train
train = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss)


## session, 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())


## 학습
for step in range(300000):
    _, W_val, b_val, loss_val = sess.run([train, W, b, loss],
                                         feed_dict={X: training_data_x,
                                                    T: training_data_t})
    
    if step % 30000 == 0:
        print(f'W: {W_val}, b: {b_val}, loss: {loss_val}')
        
```

![image-20210226173144613](md-images/image-20210226173144613.png)



##### sklearn으로 구현

```python
# 예측값을 한번 구해보아요
# tensorflow를 이용해서 만든 모델로 예측값을 구하고
# sklearn으로 구현한 모델을 이용해서 예측값을 구해서
# 값을 비교해보아요
# 예측할 값은 => (온도, 바람, 태양광) => ([80,10,150])

# sklearn을 이용해서 구현
# sklearn은 머신러닝을 모르는 사람도 데이터를 쉽게 학습해서
# 예측값을 알아낼 수 있도록 모듈화 시켜서 우리에게 제공
# 결측지 정도만 제거하고 이상치 정규화 과정 없이 sklearn 실행

## Training Data Set
training_data_x = scaler_x.transform(training_data[['Temp','Wind','Solar.R']].values)
training_data_t = scaler_t.transform(training_data['Ozone'].values.reshape(-1,1))

model = linear_model.LinearRegression()
model.fit(training_data[['Temp','Wind','Solar.R']].values,
         training_data['Ozone'].values.reshape(-1,1))
# model.fit(training_data_x, training_data_t)

print(f'w: {model.coef_}, b: {model.intercept_}')

# prediction

# sklearn을 이용
sklearn_result = model.predict([[80.0,10.0,150.0]])
print(f'sklearn : {sklearn_result}')


# tf에는 입력값도 정규화 해줘야함
predict_data = np.array([[80.0,10.0,150.0]])
scaled_predict_data = scaler_x.transform(predict_data)

# tensorflow를 이용
tensorflow_result = sess.run(H, feed_dict={X:scaled_predict_data})

tensorflow_result = scaler_t.inverse_transform(tensorflow_result)

print(f'tensorflow : {tensorflow_result}')
```

![image-20210226173247947](md-images/image-20210226173247947.png)