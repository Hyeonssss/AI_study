# day46_ML_Decision Tree



#### Decision Tree (의사결정나무)

- 데이터를 분류해서 이들 사이에 존재하는 패턴 파악, 예측 가능한 규칙들의 조합을 만드는 알고리즘

- 어떤 사람이 청소년, 일반인, 대학생인지를 알아맞히는 스무고개 놀이

![image-20210402091858413](md-images/image-20210402091858413.png)

- root node, 중간 노드, 터미널(단말) 노드

  

- ##### 의사결정나무

  - 굉장히 직관적이고 친근한 방식

  - 장점:

    - 다른 모델에 비해 빠르고 간단

    - 데이터의 특성에 따라 다른 모델에 비해 더 좋은 성능을 내는 경우가 있음

      독립변수와 종속변수가 둘다 이산 데이터 일 경우(실수 데이터가 아닌 것, 연속적이지 않은 것, 정수 데이터)

      classification 문제에서 class 개수가 적을 때 적합, 데이터의 개수가 많을 때

![image-20210402100433496](md-images/image-20210402100433496.png)



##### 의사결정나무

- 영역의 순도(homogeneity)가 증가하는 방향

- 불순도(impurity)가 감소하는 방향

- 불확실성(Entropy)가 감소하는 방향

  => 학습이 진행 -> 영역을 분기한다는 의미

- 순도가 증가 / 불확실성이 것을 정보이론(Information theory) 쪽에서는 Information Gain(정보 획득)이라고 함

- 정보획득량: 어떤 사건이 얼마만큼의 정보를 줄 수 있을지를 수치화

  - 정보함수

    ![image-20210402102234168](md-images/image-20210402102234168.png)

    정보함수의 값이 크다 = 그 정보가 가치가 있다

    항상 일어나는 일이 아니라 특이한 순간에 일어나는 일

    ![image-20210402102634033](md-images/image-20210402102634033.png)

    

  - Entropy

    무질서도를 정량화한 값, 확률변수의 불확실성을 수치로 표현

    Entropy가 높으면 특징을 찾아내기 힘들고, 불확실성이 높아진다.

- A상태에서 B상태로 전이할 때

  불확실한 상황(entropy 높음) -> 불확실하지 않은 상황으로 전이될 때 정보량이 많음

- 정보획득량: 전체 Entropy - 분류 후 Entropy



- A 영역에 대한 Entropy (무질서도를 숫자로 표현)

  Pk: A 영역에 속하는 데이터 가운데 k 범주에 속하는 데이터의 비율

![image-20210402104010971](md-images/image-20210402104010971.png)

![image-20210402104618532](md-images/image-20210402104618532.png)

​	=> 0.954

![image-20210402112139904](md-images/image-20210402112139904.png)

- 분기 후 Entropy가 줄었음 -> 순도가 증가, 불확실성 감소, 정보획득 발생

  분할 전보다 분할 후가 더 좋다고 판단해 노드의 분기가 진행됨



##### 정리

- Decision Tree의 node 분기는 
  - 순도가 증가하는 방향
  - 불순도가 감소하는 방향(지니 계수 )
  - 불확실성이 최대로 감소하는 방향(Entropy가 감소하는 방향)

![image-20210402130325492](md-images/image-20210402130325492.png)

- 전체에선 0.94 wind 추가 시 0.892

  정보획득량은 0.94 - 0.892 = 0.048

![image-20210402130910916](md-images/image-20210402130910916.png)

- Decision Tree 가장 큰 단점:

1. Greedy 알고리즘을 이용하기 때문에 최적의 tree를 구성하지 못할 수 있음
2. overfitting이 심하다 -> 가지치기

```python
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import warnings
from mlxtend.plotting import plot_decision_regions

warnings.filterwarnings(action='ignore')

df = pd.read_csv('./dataset/bmi/bmi.csv', skiprows=3)

x_data = df[['height','weight']].values
t_data = df['label'].values

num_of_data = 30

# label 0 => red
x_data_red = x_data[t_data==0][:num_of_data]
t_data_red = t_data[t_data==0][:num_of_data]

# label 1 => blue
x_data_blue = x_data[t_data==1][:num_of_data]
t_data_blue = t_data[t_data==1][:num_of_data]

# label 2 => green
x_data_green = x_data[t_data==2][:num_of_data]
t_data_green = t_data[t_data==2][:num_of_data]

plt.scatter(x_data_red[:,0],x_data_red[:,1], color='r')
plt.scatter(x_data_blue[:,0],x_data_blue[:,1], color='b')
plt.scatter(x_data_green[:,0],x_data_green[:,1], color='g')

plt.show()


x_data_sample = np.concatenate((x_data_red,x_data_blue,x_data_green), axis=0)
t_data_sample = np.concatenate((t_data_red,t_data_blue,t_data_green), axis=0)

model = DecisionTreeClassifier()
model.fit(x_data_sample, t_data_sample)

plot_decision_regions(X=x_data_sample,
                      y=t_data_sample,
                      clf=model,
                      legend=3)

plt.show

# 전체 BMI 데이터를 이용해서 Decision Tree로 모델을 구축하고
# accuracy를 측정
# 입력 데이터가 이산형이 아니기 때문에 SVM보다 결과가 안 좋음

from sklearn.metrics import classification_report
# print(classification_report())
```

![image-20210402140604222](md-images/image-20210402140604222.png)