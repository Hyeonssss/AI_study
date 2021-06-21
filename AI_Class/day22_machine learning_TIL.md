# day22_machine learning_TIL



 ### 리뷰



##### AI

- 가장 포괄적인 개념



##### Machine Learning

- AI를 구현하기 위한 하나의 방법, 데이터를 기반
- 데이터의 특성과 패턴 파악 => 미지의 데이터에 대한 prediction



##### Deep Learning

- 머신 러닝 기법 중 Neural Network (신경망)을 이용해서 학습 및 추론하는 다양한 알고리즘 및 기법



##### 등장 배경

- 기존의 Data Mining과 다른점:

  Explicit program으로 해결할 수 없는 문제를 해결하기 위해 Machine Learning 등장



##### Machine Learning의 Type

###### 지도학습 (supervised Learning)

- Training Data Set

  입력값 : (x)

  label(1개) : (y) 

  

###### 비지도학습 (unsupervised Learning)

- label이 존재하지 않음
- 입력값만 존재
- 입력값을 이용해서 데이터를 유사한 것들 끼리 clustering



###### 준지도학습 (semisupervised Learning)

- 지도학습 + 비지도학습



###### 강화학습 (Reinforcement Learning)

- 위 방법과는 아예 다른 방식





### Today



##### 수치미분

> Numerical Differentiation

- 프로그램적으로 계산을 통해서 미분을 수행

- 오차가 발생 가능(약간의 오류가 발생)



##### 해석미분

- 미분 이론에 입각해서 펜과 종이를 이용해 논리적으로 문제를 해결 (고등학교 미분)



##### 미분의 정의

> 미분함수 (도함수) (derivative)

- 어떤 함수의 정의역 속 각 점에서 

  독립변수의 변화량과 함수값의 변화량에 대한 비율의 극한으로 구성된 집합

  으로 치역이 구성되는 함수



> 미분법 (differentiation)

- 미분을 하는 작업, 도함수를 구하는 작업 그 자체



> 미분

- 함수에 대한 특정 순간의 변화량
- x의 변화가 f(x)를 얼마나 변화 시키는지를 나타냄

![image-20210222101341500](md-images/image-20210222101341500.png)



##### 수치미분

- 전향차분



- 중앙차분

  일반적으로 이용 (정확도가 가장 높음)

  ![image-20210222101900766](md-images/image-20210222101900766.png)



- 후향차분



##### 기본 미분 공식

1. f(x) = constant (상수) ex) f(x) = 3

   f'(x) => 0

2. f(x) = ax^n

   f'(x) = n*ax^n-1

3. f(x) = e^x

   f'(x) = e^x

4. f(x) = e^-x

   f(x) = -e^-x

5. f(x) = ln x

   f'(x) = 1/x

![image-20210222102903879](md-images/image-20210222102903879.png)

- 예제:

  f(x) = 3x^2 + e^x + 1/x

  f'(x) = 6x + e^x - 1/x^2



##### 편미분

> partial derivate

- 편미분을 입력변수 (독립변수)가 2개 이상인 다변수 함수에서 미분하고자 하는 변수를 제외한 나머지 변수들을 상수 처리해서 미분 진행

- f(x,y)를 x에 대해서 partial derivate(편미분) 하는 경우

  af(x,y) / ax

- 예제:

  f(x,y) = 2x + 3xy + y^3

  af(x,y) / ax = 2 + 3y

  af(x,y) / ay =       3y + 3y^2



##### 연쇄법칙

> chain Rule

- 합성함수 (composite function): 여러 함수로 구성된 함수

  합성함수를 미분 -> chain rule -> 합성함수를 구성하는 각 함수를 각각 미분해서 그 결과의 곱으로 계산



##### Numerical Differentiation

- 일변수 함수에 대한 수치미분 코드로 구현

```python
# # 입력으로 들어오는 x에서 아주 미세하게 변화할때
# 함수 f가 얼마나 변화하는 지에 대해 수치적 계산

# 아래의 함수는 인자를 2개 받음
# 한개는 미분하려는 함수, 특정 점에서 미분값을 구하기 위한 x값

def numerical_derivative(f, x):
    # f: 미분하려는 함수
    # x: 미분값을 알고자하는 입력값
    # delta_x => 극한에 해당하는 값으로 아주 작은 값을 이용
    #            1e-8 이하로 지정하면 소수점 연산 오류가 발생
    #            일반적으로 1e-5 정도로 설정하면 됨
    delta_x = 1e-5
    
    return ( f(x+delta_x) - f(x-delta_x) ) / (2 * delta_x)

# 미분하려는 함수가 있어야 해요 (단변수 함수)
def my_func(x):
    
    return x ** 2    # f(x) = x^2

result = numerical_derivative(my_func, 3)

print(f'미분한 결과값은 : {result}')
```

- 미분한 결과값은 : 6.000000000039306

  

```python
## 일반적으로 다변수 함수의 수치미분 코드를 이용합니다.

# 입력변수가 2개 이상인 다변수 함수의 경우
# 입력변수는 서로 독립이기 때문에 수치미분 역시 변수의 개수만큼 개별적으로
# 진행해야해요

import numpy as np

def numerical_derivative(f, x):
    # f: 미분하려고 하는 다변수 함수
    # x: 모든 값을 포함하는 numpy array  ex) f'(1.0, 2.0) = (8.0, 15.0)
    delta_x = 1e-4
    derivative_x = np.zeros_like(x)     # [[0 0]
                                        #  [0 0]]
    it = np.nditer(x, flags=['multi_index'])
    
    while not it.finished:
        
        idx = it.multi_index # 현재의 iterator의 index를 추출 => tuple형태 
        print(f'현재의 idx: {idx}')
        
        tmp = x[idx] # 현재 index의 값을 잠시 보존,
                     # delta_x를 이용한 값으로 ndarray를 수정한 후 편미분
                     # 함수값을 계산한 후 원상복구를 해줘야 다음 독립변수에
                     # 대한 편미분을 정상적으로 수행 가능
        print(f'현재 temp: {tmp}')
        x[idx] = tmp + delta_x
        fx_plus_delta = f(x)    # f([1.00001, 2.0]) => f(x + delta_x)
        
        x[idx] = tmp - delta_x
        fx_minus_delta = f(x)    # f([0.99999, 2.0]) => f(x - delta_x)
        
        derivative_x[idx] = (fx_plus_delta - fx_minus_delta) / (2 * delta_x)
        
        x[idx] = tmp
        
        it.iternext()
    
    return derivative_x

def my_func(input_data):
    
    x = input_data[0]
    y = input_data[1]
    return 2*x + 3*x*y + np.power(y, 3)    # f(x) = 2x + 3xy + y^3


param = np.array([1.0, 2.0])
result = numerical_derivative(my_func, param)
print(f'미분한 결과는 : {result}')

```

```
현재의 idx: (0,)
현재 temp: 1.0
현재의 idx: (1,)
현재 temp: 2.0
미분한 결과는 : [ 8.         15.00000001]
```



##### Regression Model

- 어떠한 데이터에 대해 그 값에 영향을 주는 조건을 고려해 데이터의 평균을 구하기 위한 함수
- 평균 => 대표성
- 어떠한 데이터에 대해 그 값에 영향을 주는 조건을 고려해 그 데이터를 가장 잘 표현하는 값을 뽑아내는 함수



- 독립변수가 1개인 함수를 가정하면 

  y = B0 + B1x  =>  y = ax + b (직선)

  베타 0 : 기타 영향을 주는 요인

  베타 1 : x에 영향을 주는 요인

  => 결론적으로 독립변수가 한개이면 직선

- 우리가 구하려는 Regression Model은 주어진 데이터를 가장 잘 표현하는 '직선'을 찾는 것 (독립변수가 1개인 경우)

  'classical linear Regression Model'



##### Regression (회귀)

> Regression toward mean : 평균으로 되돌아간다

- 프란시스 골턴이 지음

  우생학 -> 인종을 개선하는 과학

  훌륭한 사람은 환경 x, 유전자 o



##### Regression Model

- 어떤 연속형 데이터 y와 이 y의 원인으로 생각되는 x 간의 관계를 추정하기 위해서 만든 y = f(x) + e 관계식

  y = h(x1, x2, x3, ..., xk, B1, B2, B3, ..., Bk) + e

  h : 조건에 따른 평균을 구하는 함수(회귀모델)

- Model을 왜 만들죠?

  우리가 해결해야하는 현실은 너무 복잡 -> 현실의 문제를 '단순화' (가정)

  

##### Classical Linear Regression Model (고전적 선형 회귀 모델)

- 오차항은 평균이 0, 정규분포
- 독립변수와 종속변수는 선형관계
- 데이터의 아웃라이어가 없어야 함
- 독립변수와 오차항은 독립

![image-20210222153546282](md-images/image-20210222153546282.png)



- 우리는 단변량 선형 회귀모델을 사용

  단변량 : 종속변수가 1개

  다변량 : 종속변수가 다수

- 단순 선형 회귀모델 : 독립변수가 1개

  다중 선형 회귀모델 : 독립변수가 다수