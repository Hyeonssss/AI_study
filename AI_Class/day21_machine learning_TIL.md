# day21_machine learning_TIL



### 기본 개념



##### AI

> Artificial Intelligence, 인공지능

- 인간이 가지는 특유의 학습 능력, 추론 능력을 컴퓨터로 구현하려는 가장 포괄적인 개념



##### Machine Learning

- AI를 구현하기 위한 하나의 방법
- 데이터의 특성과 패턴을 학습 -> 미지의 데이터에 대한 추정치를 계산하는 프로그래밍 기법



- Regression (회귀)

- SVM(Support Vector Machine)

- Decision Tree (의사결정 나무)

- Random Forest (앙상블 하기)

- Naive Bayes (나이브 베이지안)

- KNN 

- Neural Network (신경망) - 주목

  신경망을 이용해서 학습하는 구조와 알고리즘이 최근에 개선, 개발되었음 => Deep Learning(CNN, RNN, LSTM, GAN)

- Clustering (K-Means, DBSCAN)

- Reinforcement Learning (강화학습)



##### Deep Learning

- Neural Network, CNN, RNN, LSTM

- Machine Learning의 기법 중 일부



##### Data Mining

- 데이터들의 상관관계
- 새로운 feature를 알아내기
- 즉, 데이터로부터 상관관계를 찾아 의미있는 특징을 도출



##### Machine Learning

- 데이터를 학습해서 미지의 데이터에 대한 예측(prediction)
- Explicit program으로 해결할 수 없는 문제를 해결하기 위해 등장(1960)

- Explicit program
  - Rule based programing
  - 답이 정해져 있는, 규칙이 있는 프로그래밍
  - 정해져 있는대로 프로그래밍 하는 방식

- 규칙이 너무 많을 때 Explicit program에 한계

  경우의 수(조건)이 너무 많은 경우엔 Rule based programming을 할 수 없다 ex) 바둑 

- Machine Learning (Regression) 등장



#### Machine Learning

- 학습 방법에 따라 크게 4가지로 구분

  - 지도학습 (supervised Learning)

    우리가 해결하는 거의 대부분의 문제

    Label이 존재 (학습 데이터에 답이 들어있다)

    답을 기반으로 학습해서 예측한다

  - 비지도학습 (Unsupervised Learning)

    Label이 없다 

    Model (prediction, 공식) -> 예측

    분류 (clustering)를 나눠서 묶어주는 학습

    - clustering (분류) -> 비지도학습에서 사용 연관성으로 단순 묶기
    - classification (분류) -> 지도학습에서 사용, prediction할 때 어떤 걸로 할 지 알려주는거? 예측

  - 준지도학습 (semisuperviesd Learning)

    지도학습 + 비지도학습

  - 강화학습 (Reinforcement Learning)

    완전 별개의 방법으로 학습

    

- 지도학습, 비지도학습, 준지도학습은 비슷



##### 지도학습 (supervised Learning)

- Training Data Set (학습 데이터셋)

  입력값 : x

  정답(Label) : t

- 지도학습 (supervised Learning) 예시

  공부 시간에 따른 시험점수 예측 :

  - Regression (회귀)

    Label은 범위가 넓은 실수값으로 되어있음

    연속적인 숫자값을 얼마나????로 예측

    

  공부 시간에 따른 시험 합격 여부 예측 :

  - (True, False로 들어감)

  - 최종적으로 어떤 종류의 값이 도출될지를 예측 -> 어떤 것??

  - binary classification

    

  공부 시간에 따른 성적 Grade 예측

  - Label이 특정 문자로 들어감? A, B, C, D, F
  - Multinomial classification



- Label의 형태에 따라서 classification하는 것 : 지도학습





