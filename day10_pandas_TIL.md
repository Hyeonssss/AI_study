# day10_pandas_TIL

## pandas

- 목차

  Pandas

  Series

  DataFrame 

  - 생성: CSV, list, dict, Open API, Database

            - indexing(열과 행의 indexing 구분)
              
              => indexing, slicing, fancy indexing, boolean indexing
            
            => loc[]
            
            - function(수학적 집계함수, 정렬함수, 유틸리티 함수)
                      - merg
     - merge(DataFrame 결합)



### DataFrame 함수

#### 집계함수

1. 기댓값:

   어떤 확률을 가진 사건을 무한히 반복했을 때 얻을 수 있는 값의 평균으로 기대할 수 있는 값

   

2. 편차:

   확률변수 X와 평균값의 차이, 데이터의 흩어짐 정도를 나타낼 수 있음

   편차의 합계는 0이기 때문에 데이터의 흩어짐 정도를 수치화하기 힘듦

   

3. 분산(var()):

   편차의 제곱의 평균, 원래 데이터에 제곱을 했기 때문에 값이 불려졌을 가능성

   

4. 표준편차(std()):

   분산의 제곱근

   

```python
import numpy as np
import pandas as pd

arr = np.array([4,6,1,3,8,8], dtype=np.int32)

print(arr)
print(arr.sum())  # 합
print(arr.mean()) # 평균
print(arr.var())  # 분산
print(arr.std())  # 표준편차
```



#### 공분산

- 공분산(covariance):

  두 개의 확률변수의 관계를 보여주는 값

  두 확률변수 편차의 곱에 대한 평균으로 계산

  

- 표현식: np.cov()

```python
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
from datetime import datetime

start = datetime(2019,1,1)  # 2019-01-01 날짜 객체 생성
end = datetime(2019,12,31)  # 2019-12-31 날짜 객체 생성

# YAHOO에서 제공하는 KOSPI 지수
df_KOSPI = pdr.DataReader('^KS11','yahoo',start,end)
# display(df_KOSPI['Close'].values)

# YAHOO에서 제공하는 삼성전자 지수
df_SE = pdr.DataReader('005930.KS','yahoo',start,end)
# display(df_SE['Close'].values)

# numpy가 제공하는 함수를 이용해서 공분산 계산
# np.cov: 공분산
print(np.cov(df_KOSPI['Close'].values, df_SE['Close'].values))
'''
[[6.28958682e+03 9.46863621e+04]
 [9.46863621e+04 1.41592089e+07]]
'''
# 0행 0열: KOSPI의 공분산( KOSPI & KOSPI )
# 0행 1열: KOSPI와 삼성전자의 공분산
# 1행 0열: 삼성전자와 KOSPI의 공분산 ( 0행 1열 = 1행 0열 )
# 1행 1열: 삼성전자의 공분산 ( 삼성전자 & 삼성전자 )
```



#### 상관관계

- 상관관계(correlation):

  두 대상이 서로 연관성이 있다고 추측되는 관계

  

- 상관계수(correlation cofficient):

  -1과 1 사이의 실수(주로 피어슨 상관계수 사용)

  하나의 변수가 변할 때 다른 변수가 변화하는 정도

  

  양수값: 정적상관관계

  음수값: 부적상관관계

  

  0에 가까울수록 관련성이 적다

  1에 가까울수록 관련성이 높다

  

- 표현식: corr()



- 주의점:

  상관관계는 인간관계를 설명할 수 없다

  인과관계 설명 시엔 회귀분석을 사용

```python
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
from datetime import datetime

start = datetime(2018,1,1)  # 2018-01-01 날짜 객체 생성
end = datetime(2018,12,31)  # 2018-12-31 날짜 객체 생성

df_KOSPI = pdr.DataReader('^KS11','yahoo',start,end)     # KOSPI
df_SE = pdr.DataReader('005930.KS','yahoo',start,end)    # 삼성전자
df_PUSAN = pdr.DataReader('011390.KS','yahoo',start,end) # 부산산업(남북경협)
df_LIG = pdr.DataReader('079550.KS','yahoo',start,end) # LIG넥스원(방위)

my_dict = {
    'KOSPI' : df_KOSPI['Close'],
    '삼성전자' : df_SE['Close'],
    '부산산업' : df_PUSAN['Close'],
    'LIG넥스원' : df_LIG['Close']
}

df = pd.DataFrame(my_dict)
display(df)

display(df.corr()) # DataFrame이 가지고 있는 상관계수를 구하는 함수 이용
```



#### 분석용 함수

- DataFrame 생성

```python
import numpy as np
import pandas as pd

data = [[2, np.nan],
        [7, -3],
        [np.nan, np.nan],
        [1, -2]]

df = pd.DataFrame(data,
                  columns=['one', 'two'],
                  index=['a', 'b', 'c', 'd'])

display(df)
```

|      |  one |  two |
| ---: | ---: | ---: |
|    a |  2.0 |  NaN |
|    b |  7.0 | -3.0 |
|    c |  NaN |  NaN |
|    d |  1.0 | -2.0 |



##### sum()

- axis를 생략하면 기본값인 axis=0 적용

- skipna = True(기본값), np.nan값을 연산에서 제거

- Series로 리턴

- DataFrame은 column별로 sum

  그러나, ndarray는 모든 값 sum



- fillna(value = x):

  nan 값을 x로 치환

```python
display(df.sum())  # axis를 생략하면 기본이 axis=0
                   # skipna = True(기본값), np.nan 연산에서 제거
                   # Series로 리턴
                   # DataFrame은 column 별로 sum
                   # Ndarray는 모든 값 sum

display(df.sum(axis=1))

print(df['two'].sum()) # 특정 column의 합 추출
print(df.loc['b'].sum())
print(df.mean(axis=0, skipna=False)) # nan이 연산에 들어가면 답은 nan
print(df.mean(axis=0, skipna=True))

df['two'] = df['two'].fillna(value=df['two'].mean())

# DataFrame 전체 nan 값을 value=df['two'].mean()으로 변경
df = df.fillna(value=df['two'].mean())

display(df)
```



### sort

- numpy & DataFrame

  numpy: row, (record라고 부르지 않음)

  DataFrame: record

```python
# numpy & DataFrame

arr = np.array([[1,2,3],[4,5,6]])
print(arr) # row, ( record라고 부르지 않음 )
'''
[[1 2 3]
 [4 5 6]]
'''

df = pd.DataFrame(arr)
display(df) # record
'''
	0	1	2
0	1	2	3
1	4	5	6
'''
```



#### date_range

> pandas의 날짜함수

- 표현식: pd.date_range('20200101', periods=6)

  periods: 몇일 동안인지 숫자 입력

- random.shuffle():

  원본 데이터를 변경

  index는 원본이 변경되지 않음, shuffle 사용 불가

- random.permutation():

  원본 변경 x, 순서가 바뀐 복사본 리턴

```python
# sort

import numpy as np
import pandas as pd

np.random.seed(1)
df = pd.DataFrame(np.random.randint(0,9,(6,4)))
# display(df)

df.columns = ['A', 'B', 'C', 'D']
# periods: 몇일 동안인지 숫자 
df.index = pd.date_range('20200101', periods=6)
# display(df)

arr = np.array([1,2,3,4])
np.random.shuffle(arr)
print(arr)
np.random.shuffle(df.index) # shuffle()의 특징: 원본 데이터를 변경
                            # index는 원본이 변경되면 안됨, shuffle 사용 불가
display(df)

new_index = np.random.permutation(df.index) # permutation(): 원본 변경 X
                                            # 순서가 바뀐 복사본 리턴
print(new_index)
df2 = df.reindex(index=new_index, columns=['B','A','D','C']) # 원본 변경 X
display(df)
display(df2)
```



#### sort

- 정렬은 기본적으로 axis를 기준으로 정렬

  numpy는 행이 같이 움직이지 않고 따로 움직임

  DataFrame은 행의 값들이 함께 정렬됨

- sort_index(axis=0, ascending=True):

  index를 기준으로 정렬하는 함수

  ascending=True: index를 오름차순 정렬

  ascending=False: index를 내림차순 정렬

- sort_values(by=['B', 'A']):

  특정 column의 값을 기준으로 행을 정렬하는 함수

  'B' 컬럼을 기준으로 정렬한 후 동일한 값이 있을 시 'A' 컬럼으로 추가 정렬

```python
# index를 기준으로 정렬
display(df2.sort_index(axis=0, ascending=True)) # index 정렬
'''
			B	A	D	C
2020-01-01	8	5	0	5
2020-01-02	1	0	6	7
2020-01-03	4	2	2	5
2020-01-04	2	4	7	4
2020-01-05	1	7	0	7
2020-01-06	7	6	1	6
'''

# 특정 column의 값을 기준으로 행을 정렬
display(df2.sort_values(by=['B', 'A']))
'''
			B	A	D	C
2020-01-02	1	0	6	7
2020-01-05	1	7	0	7
2020-01-04	2	4	7	4
2020-01-03	4	2	2	5
2020-01-06	7	6	1	6
2020-01-01	8	5	0	5
'''
```


### 다양한 함수

- unique():

  중복 제거, np.array로 출력됨

- value_counts():

  각 요소들의 개수 출력, Series로 출력됨

- isin():

  각 row들에 isin 속 값이 들어 있는지 bool값으로 출력, Series로 출력됨

  조건을 검색할 때 사용

```python
import numpy as np
import pandas as pd

np.random.seed(1)
df = pd.DataFrame(np.random.randint(0,10,(6,4)))
df.columns = ['A', 'B', 'C', 'D']
df.index = pd.date_range('20200101',periods=6)
display(df)

df['E'] = ['AA','BB','CC','CC','AA','CC']
display(df)

# unique(): 중복 제거, np.array로 출력됨
print(df['E'].unique()) # ['AA' 'BB' 'CC']


# value_counts(): 각 요소들의 개수 출력, Series로 출력됨
print(df['E'].value_counts())
'''
CC    3
AA    2
BB    1
Name: E, dtype: int64
'''


# isin(): 각 row들에 isin 속 값이 들어 있는지 bool값으로 출력, Series로 출력
print(df['E'].isin(['AA', 'BB'])) # 조건을 검색할 때 사용
'''
2020-01-01     True
2020-01-02     True
2020-01-03    False
2020-01-04    False
2020-01-05     True
2020-01-06    False
Freq: D, Name: E, dtype: bool
'''
```



### merge

> Database의 inner join과 비슷

- 표현식:

  pd.merge(df1, df2, on='학번', how='inner')

  - on='학번': 학번 컬럼을 기준으로 결합하라

  - how='inner': inner join, 겹치는 것만 결합하고, 겹치지 않는 것은 제외
  - how='outer': full outer join, 겹치지 않는 값은 nan으로 출력
  - how='left': left outer join
  - how='right': right outer join

```python
# DataFrame merge
# Database의 inner join과 비슷

import numpy as np
import pandas as pd

data1 = {
    '학번' : [1, 2, 3, 4],
    '이름' : ['홍길동', '신사임당', '아이유', ' 김연아'],
    '학년' : [1, 4, 1, 3]
}

data2 = {
    '학번' : [1, 2, 4, 5],
    '학과' : ['컴퓨터', '철학', '심리', '영어영문'],
    '학점' : [3.5, 2.7, 4.0, 4.3]
}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

display(df1)
'''
	학번	이름	학년
0	1	홍길동		1
1	2	신사임당	4
2	3	아이유		1
3	4	김연아		3
'''

display(df2)
'''
	학번	학과	학점
0	1	컴퓨터		3.5
1	2	철학		2.7
2	4	심리		4.0
3	5	영어영문	4.3
'''


# on='학번': 학번 컬럼을 기준으로 결합하라

# how='inner': 겹치는 것만 결합하라, 겹치지 않는 것은 제외
display(pd.merge(df1, df2, on='학번', how='inner'))
'''
	학번	이름	학년	학과	학점
0	1	홍길동		1	컴퓨터	3.5
1	2	신사임당	4	철학	2.7
2	4	김연아		3	심리	4.0
'''

# how='outer': full outer join하라, 겹치지 않는 값은 nan으로 출력
display(pd.merge(df1, df2, on='학번', how='outer'))
'''

	학번	이름	학년		학과		학점
0	1	홍길동		1.0		컴퓨터		3.5
1	2	신사임당	4.0		철학		2.7
2	3	아이유		1.0		NaN		NaN
3	4	김연아		3.0		심리		4.0
4	5	NaN		NaN		영어영문	4.3
'''

# how='left': left outer join
display(pd.merge(df1, df2, on='학번', how='left'))
'''
	학번	이름	학년		학과		학점
0	1	홍길동		1	컴퓨터		3.5
1	2	신사임당	4	철학		2.7
2	3	아이유		1	NaN		NaN
3	4	김연아		3	심리		4.0
'''

# how='left': right outer join
display(pd.merge(df1, df2, on='학번', how='right'))
'''
	학번	이름		학년		학과		학점
0	1	홍길동		1.0		컴퓨터		3.5
1	2	신사임당	4.0		철학		2.7
2	4	김연아		3.0		심리		4.0
3	5	NaN		NaN		영어영문	4.3
'''
```



- 데이터는 같으나 column명이 다를 경우 merge하는 법

  - 표현식: 

    pd.merge(df1, df2, left_on='학번', right_on='학생번호', how='inner')

```python
import numpy as np
import pandas as pd

data1 = {
    '학번' : [1, 2, 3, 4],
    '이름' : ['홍길동', '신사임당', '아이유', ' 김연아'],
    '학년' : [1, 4, 1, 3]
}

data2 = {
    '학생번호' : [1, 2, 4, 5],
    '학과' : ['컴퓨터', '철학', '심리', '영어영문'],
    '학점' : [3.5, 2.7, 4.0, 4.3]
}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# 데이터는 같으나 column명이 다를 경우 merge 하는 법
display(pd.merge(df1, df2, left_on='학번', right_on='학생번호', how='inner'))
'''
	학번	이름	학년	학생번호	학과	학점
0	1	홍길동		1	1	컴퓨터	3.5
1	2	신사임당	4	2	철학	2.7
2	4	김연아		3	4	심리	4.0
'''
```



- 학번이 index로 사용된 경우

  - 생성 시 학번을 index로 설정

    df2 = pd.DataFrame(data2,
                       				  index=[1, 2, 4, 5]) 

  - 표현식:

    pd.merge(df1, df2,
                      left_on='학번',
                      right_index=True,
                      how='inner')

    - right_index=True, index 값으로 merge하겠다
    - index로 merge할 경우 index값을 left DataFrame의 인덱스를 가져옴

```python
import numpy as np
import pandas as pd

data1 = {
    '학번' : [1, 2, 3, 4],
    '이름' : ['홍길동', '신사임당', '아이유', ' 김연아'],
    '학년' : [1, 4, 1, 3]
}

data2 = {
    '학과' : ['컴퓨터', '철학', '심리', '영어영문'],
    '학점' : [3.5, 2.7, 4.0, 4.3]
}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2,
                   index=[1, 2, 4, 5]) # 학번이 index로 사용된 경우

# right_index=True, index 값으로 merge하겠다
# index로 merge할 경우 index값을 left DataFrame의 인덱스를 가져옴
result = pd.merge(df1, df2,
                  left_on='학번',
                  right_index=True,
                  how='inner')

display(result)
# print(result.loc[2]) # Error
print(result.loc[3])  # OK
print(result.iloc[2]) # OK
```



- 둘 다 학번을 index로 사용할 경우

  - 표현식:

    pd.merge(df1, df2,
                      left_index=True,
                      right_index=True,
                      how='inner')

```python
# 둘 다 학번을 index로 사용할 경우

import numpy as np
import pandas as pd

data1 = {
    '이름' : ['홍길동', '신사임당', '아이유', ' 김연아'],
    '학년' : [1, 4, 1, 3]
}

data2 = {
    '학과' : ['컴퓨터', '철학', '심리', '영어영문'],
    '학점' : [3.5, 2.7, 4.0, 4.3]
}

df1 = pd.DataFrame(data1,
                   index=[1, 2, 3, 4])
df2 = pd.DataFrame(data2,
                   index=[1, 2, 4, 5]) # 학번이 index로 사용된 경우

result = pd.merge(df1, df2,
                  left_index=True,
                  right_index=True,
                  how='inner')

display(result)
'''
	이름		학년	학과	학점
1	홍길동		1	컴퓨터	3.5
2	신사임당	4	철학	2.7
4	김연아		3	심리	4.0
'''
```



### concatenation

> DataFrame의 연결

- concat():

  단순 결합, 데이터가 없으면 nan으로 붙음

- 표현식:

  pd.concat([df1, df2],
                     axis=1,
                     sort=True)

  - sort = True: index 정렬

```python
# DataFrame의 연결(concatenation)

import numpy as np
import pandas as pd

df1 = pd.DataFrame(np.arange(6).reshape(3,2),
                   index=['a','b','d'],
                   columns=['one','two'])

df2 = pd.DataFrame(np.arange(4).reshape(2,2),
                   index=['a','c'],
                   columns=['three','four'])

display(df1)
'''
	one	two
a	0	1
b	2	3
d	4	5
'''

display(df2)
'''
	three	four
a	0		1
c	2		3
'''

# concat(): 단순 결합, 데이터가 없으면 nan으로 붙음
result = pd.concat([df1, df2],
                   axis=1,
                   sort=True) # index 정렬

display(result)
'''
	one	two	three	four
a	0.0	1.0	0.0		1.0
b	2.0	3.0	NaN		NaN
c	NaN	NaN	2.0		3.0
d	4.0	5.0	NaN		NaN
'''
```



- 행방향으로 결합할 경우:

  동일 index가 있더라도 새롭게 생김

- 표현식:

  pd.concat([df1, df2],
                     axis=0,
                     ignore_index=True) 

  - ignore_index = True: 현재 index를 무시하고 새로운 숫자 index 삽입

```python
# DataFrame의 연결(concatenation)

import numpy as np
import pandas as pd

df1 = pd.DataFrame(np.arange(6).reshape(3,2),
                   index=['a','b','d'],
                   columns=['one','two'])

df2 = pd.DataFrame(np.arange(4).reshape(2,2),
                   index=['a','c'],
                   columns=['three','four'])

display(df1)
'''
	one	two
a	0	1
b	2	3
d	4	
'''

display(df2)
'''
	three	four
a	0		1
c	2		3
'''

# concat(): 단순 결합, 데이터가 없으면 nan으로 붙음
# 행방향으로 결합할 경우 동일 index가 있더라도 새롭게 생김
result = pd.concat([df1, df2],
                   axis=0,
                   ignore_index=True) # index를 무시하고 숫자 인덱스 삽입

display(result)
'''
	one	two	three	four
0	0.0	1.0	NaN		NaN
1	2.0	3.0	NaN		NaN
2	4.0	5.0	NaN		NaN
3	NaN	NaN	0.0		1.0
4	NaN	NaN	2.0		3.0
'''
```

