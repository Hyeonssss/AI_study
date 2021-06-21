# day32_machine learning_TIL





##### Machine Learning의 'hello world'라고 불리는 MNIST 예제 구현

- Tensorflow로 구현 
- MNIST -> Vision data set(이미지 data)-> 손으로 쓰여진 숫자 이미지

![image-20210309102733653](md-images/image-20210309102733653.png)

- 하나의 이미지 2차원 픽셀 데이터를 1차원 벡터로 만들고 label로 사용

![image-20210309103125435](md-images/image-20210309103125435.png)

- 이후 직접 적은 숫자를 업로드하여 예측



#### 수행평가 MNIST

![image-20210309142912681](md-images/image-20210309142912681.png)





##### confusion_Matrix

- sklearn이 제공하는 confusion_matrix()

###### binary classification의 경우

![image-20210310151443512](md-images/image-20210310151443512.png)



###### multinomial classification의 경우

