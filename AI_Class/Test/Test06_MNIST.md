# Test06_MNIST



###### 모델 만들고 학습

```python
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report

## Row data Loading
df = pd.read_csv('./data/digit-recognizer/train.csv')
# display(df.head(), df.shape)

# 데이터의 명세 파악
# 각 픽셀의 값은 0 ~ 255 값, 크면 클수록 어두운 색상


## 결측치와 이상치 처리 => 현재 데이터엔 결측치, 이상치 없음


## 이미지 확인
# img_data = df.drop('label', axis=1, inplace=False).values

# fig = plt.figure()
# fig_arr = []  # 10개의 subplot을 만들고 그  각각의 subplot을 list에 저장

# for n in range(10):
#     fig_arr.append(fig.add_subplot(2, 5, n+1))
#     fig_arr[n].imshow(img_data[n].reshape(28,28),
#                       cmap='Greys',
#                       interpolation='nearest')

# plt.tight_layout()
# plt.show()


## 데이터 분할
x_data_train, x_data_test, t_data_train, t_data_test =  \
train_test_split(df.drop('label', axis=1, inplace=False),
                 df['label'],
                 test_size=0.3,
                 random_state=0)


## 정규화
scaler = MinMaxScaler()
scaler.fit(x_data_train)

x_data_train_norm = scaler.transform(x_data_train)
x_data_test_norm = scaler.transform(x_data_test)


## Tensorflow 구현
sess = tf.Session()
t_data_train_onehot = sess.run(tf.one_hot(t_data_train, depth=10))
t_data_test_onehot = sess.run(tf.one_hot(t_data_test, depth=10))


## placeholder
X = tf.placeholder(shape=[None,784], dtype=tf.float32)
T = tf.placeholder(shape=[None,10], dtype=tf.float32)


## Weight & bias
W = tf.Variable(tf.random.normal([784,10]), name='weight')
b = tf.Variable(tf.random.normal([10]), name='bias')


## Hypothesis
logit = tf.matmul(X,W) + b
H = tf.nn.softmax(logit)


## loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit,
                                                                 labels=T))


## train
train = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)


## parameter
num_of_epoch = 1000
batch_size = 100


## 학습용 함수
def run_train(sess, train_x, train_t):
    print('### 학습 시작 ###')
    # 초기화
    sess.run(tf.global_variables_initializer())
    
    for step in range(num_of_epoch):
        total_batch = int(train_x.shape[0] / batch_size)
        
        for i in range(total_batch):
            batch_x = train_x[i*batch_size:(i+1)*batch_size]
            batch_t = train_t[i*batch_size:(i+1)*batch_size]
            _,loss_val = sess.run([train,loss],
                                   feed_dict={X:batch_x,
                                              T:batch_t})

        if step % 100 == 0:
            print(f'loss: {loss_val}')
    
    print('###학습종료')
            

## Accuracy 측정
predict = tf.argmax(H,1)
correct = tf.equal(predict, tf.argmax(T,1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

run_train(sess, x_data_train_norm, t_data_train_onehot)

# 우리 모델 성능평가 위해 K-Fold Cross Validation 수행
## K-Fold Cross Validation
cv = 5        # Fold의 수
results = []  # 각 Fold 당 학습과 성능평가가 진행되는데 
              # 이때 계산된 성능평가 값을 저장
kf = KFold(n_splits=cv, shuffle=True)

for training_idx, validation_idx in kf.split(x_data_train_norm):
    training_x = x_data_train_norm[training_idx]  # Fancy indexing
    training_t = t_data_train_onehot[training_idx]
    
    val_x = x_data_train_norm[validation_idx]
    val_t = t_data_train_onehot[validation_idx]
    
    run_train(sess,training_x, training_t)
    results.append(sess.run(accuracy, feed_dict={X:val_x, T:val_t}))
    
print(f'측정한 각각의 결과 값: {results}')
print(f'최종 K-Fold 교차검증을 사용한 Accuracy: {np.mean(results)}')
```

![image-20210310174852169](md-images/image-20210310174852169.png)



###### testing

```python
# testing
run_train(sess, x_data_train_norm, t_data_train_onehot)
final_accurancy = sess.run(accuracy, feed_dict={X:x_data_test_norm,
                                                T:t_data_test_onehot})
print(f'우리 모델의 최종 정확도는 : {final_accurancy}')
```

![image-20210310174932768](md-images/image-20210310174932768.png)



###### classification report

```python
# 만약 Precision, Recall, F1, Accuracy를 각각 구하고 싶으면
# 첫번째 인자로 정답이 들어가야함(onehot encoding이 안된 형태)
target_names=['num 0', 'num 1', 'num 2', 'num 3', 'num 4', 
              'num 5', 'num 6', 'num 7', 'num 8', 'num 9',]
print(
classification_report(t_data_test,
                      sess.run(predict, feed_dict={X:x_data_test_norm}),
                      target_names = target_names))
```

![image-20210310175015246](md-images/image-20210310175015246.png)



###### confusion matrix 출력

```python
from sklearn.metrics import confusion_matrix

print(
confusion_matrix(t_data_test,
                 sess.run(predict, feed_dict={X:x_data_test_norm})))
```

![image-20210310175051928](md-images/image-20210310175051928.png)



###### test 데이터로 테스트 후 제출

```python
import csv
test_df = pd.read_csv('./data/digit-recognizer/test.csv')

# display(test_df)
test_data = scaler.transform(test_df)

display(test_data)
sess.run(tf.global_variables_initializer())

predictions = tf.argmax(logit,1)
test_label = sess.run([predictions], feed_dict={
    X : test_data
})



with open('./data/digit-recognizer/submission.csv', mode='w', newline='\n') as csvfile:

    tensor_result = csv.writer(csvfile, delimiter=',')

    tensor_result.writerow(['ImageId', 'Label'])

    for line in range(len(test_label[0])):

        tensor_result.writerow([str(line+1), str(test_label[0][line])])



#print('test_label : ', test_label[0])

sess.close()

```

![image-20210315183508684](md-images/image-20210315183508684.png)



###### 실제 이미지로 예측

```python
import numpy as np
import cv2
from PIL import Image  # conda install Pillow
import matplotlib.pyplot as plt

# 이미지 로드
my_img = cv2.imread('./data/digit imege/5.jpg', cv2.IMREAD_COLOR) # (128, 128, 3)

# 흑백으로 변경 3차원 -> 2차원
im_grey = cv2.cvtColor(my_img, cv2.COLOR_BGRA2GRAY) # (128, 128)

# 새로운 이미지 저장
cv2.imwrite('./data/digit imege/5_grey.jpg', im_grey)

## 흑백 이미지 파일을 얻었어요
img = Image.open('./data/digit imege/5_grey.jpg')
img = img.resize((28,28))
img = np.array(img)
img = img.reshape((28 * 28,))
print(img.shape)
x_test = [img]
x_test = np.array(x_test)
x_test = x_test / 255

plt.imshow(x_test[0].reshape((28,28)), cmap='gray')
plt.show()

y_predict = sess.run(H, feed_dict={X: x_test})
print(y_predict) #[[0.20842288 0.41051054 0.38106653]]
print(y_predict.argmax(axis=1)) #[1]
print(y_predict.argmax(axis=1)[0]) #1
print(target_names[y_predict.argmax(axis=1)[0]]) #B
```

![image-20210310175129979](md-images/image-20210310175129979.png)

- 결과가 다르게 나옴;;