# day11_pandas_TIL

## pandas

### DataFrame 조작

- 예제 데이터 생성

```python
# DataFrame 조작

import numpy as np
import pandas as pd

# random 값을 도출해서 DataFrame 생성 => np.random.randint()
# 6행 4열짜리 DataFrame 생성

np.random.seed(1)

df = pd.DataFrame(np.random.randint(0,10,(6,4)))

# df.index = pd.date_range('20200101', periods=6)
df.index = pd.date_range('20200101', '20200106')
df.columns = ['A','B','C','D']

df['E'] = [7, np.nan, 4, np.nan, 2, np.nan]

display(df)
```

|            |    A |    B |    C |    D |    E |
| ---------: | ---: | ---: | ---: | ---: | ---: |
| 2020-01-01 |    5 |    8 |    9 |    5 |  7.0 |
| 2020-01-02 |    0 |    0 |    1 |    7 |  NaN |
| 2020-01-03 |    6 |    9 |    2 |    4 |  4.0 |
| 2020-01-04 |    5 |    2 |    4 |    2 |  NaN |
| 2020-01-05 |    4 |    7 |    7 |    9 |  2.0 |
| 2020-01-06 |    1 |    7 |    0 |    6 |  NaN |



#### nan

- nan은 데이터 분석이나 머신러닝, 딥러닝 전에 반드시 처리해야 하는 값

  

##### dropna()

- 결측치(nan)을 찾아서 drop하라

- 표현식:

  df.dropna(how='any')

  ​	how = 'any': nan이 하나라도 해당 행에 들어있으면 행 전체 제거

  ​	how = 'all': 모든 column의 값이 nan인 경우 삭제

  ​	inplace = False로 default 설정

```python
new_df = df.dropna(how='any')      # 복사본 생성
df.dropna(how='any', inplace=True) # 원본 변경

display(df)
display(new_df)
```

|            |    A |    B |    C |    D |    E |
| ---------: | ---: | ---: | ---: | ---: | ---: |
| 2020-01-01 |    5 |    8 |    9 |    5 |  7.0 |
| 2020-01-03 |    6 |    9 |    2 |    4 |  4.0 |
| 2020-01-05 |    4 |    7 |    7 |    9 |  2.0 |



##### fillna()

- nan의 값을 x로 채우는 함수

- 표현식:

  df.fillna(value=x):

```python
# fillna(value = x): nan의 값을 x로 채우는 함수
new_df = df.fillna(value=0) # 복사본 생성

display(df)
display(new_df)
```

|            |    A |    B |    C |    D |    E |
| ---------: | ---: | ---: | ---: | ---: | ---: |
| 2020-01-01 |    5 |    8 |    9 |    5 |  7.0 |
| 2020-01-02 |    0 |    0 |    1 |    7 |  0.0 |
| 2020-01-03 |    6 |    9 |    2 |    4 |  4.0 |
| 2020-01-04 |    5 |    2 |    4 |    2 |  0.0 |
| 2020-01-05 |    4 |    7 |    7 |    9 |  2.0 |
| 2020-01-06 |    1 |    7 |    0 |    6 |  0.0 |



##### isnull()

- nan인지 아닌지 bool값으로 mask를 생성하는 함수

- 표현식:

  df.isnull()

```python
# isnull(): nan인지 아닌지 bool값으로 mask를 생성하는 함수
new_df = df.isnull()

display(new_df)

# 'E' column의 값이 nan인 행들을 찾아 해당 행의 모든 column을 출력
display( df.loc[df['E'].isnull(),:] )
display( df.loc[df.isnull()['E'],:] )
```

|            |    A |    B |    C |    D |    E |
| ---------: | ---: | ---: | ---: | ---: | ---: |
| 2020-01-02 |    0 |    0 |    1 |    7 |  NaN |
| 2020-01-04 |    5 |    2 |    4 |    2 |  NaN |
| 2020-01-06 |    1 |    7 |    0 |    6 |  NaN |



#### 중복행 제어

- 예제 데이터 생성

```python
# 중복행 제어

import numpy as np
import pandas as pd

my_dict = {
    'k1' : ['one']*3 + ['two']*4,
    'k2' : [1,1,2,3,3,4,4]
}

df = pd.DataFrame(my_dict)
display(df)
```

|      |   k1 |   k2 |
| ---: | ---: | ---: |
|    0 |  one |    1 |
|    1 |  one |    1 |
|    2 |  one |    2 |
|    3 |  two |    3 |
|    4 |  two |    3 |
|    5 |  two |    4 |
|    6 |  two |    4 |



##### duplicated()

- 각각의 행 전체가 중복되어 있는지 bool값으로 mask 생성

- 표현식:

  df.duplicated():

```python
# duplicated(): 각각의 행 전체가 중복 되어 있는지 bool값으로 mask 생성
print(df.duplicated())
'''
0    False
1     True
2    False
3    False
4     True
5    False
6     True
dtype: bool
'''

# 중복된 행만 뽑아서 추출
display( df.loc[df.duplicated(),:] )
'''
	k1	k2
1	one	1
4	two	3
6	two	4
'''
```



##### drop_duplicates()

- 중복된 행을 제거한 후 추출

- 표현식:

  df.drop_duplicates():

  ​	df.drop_duplicates(['k1']) 등으로 특정 column을 기준으로 중복 제거 가능	

```python
# drop_duplicates(): 중복된 행을 제거한 후 추출
display( df.drop_duplicates() )
'''
	k1	k2
0	one	1
2	one	2
3	two	3
5	two	4
'''

df['k3'] = np.arange(7)
display(df)
'''
	k1	k2	k3
0	one	1	0
1	one	1	1
2	one	2	2
3	two	3	3
4	two	3	4
5	two	4	5
6	two	4	6
'''

# 특정 column을 기준으로 중복을 제거할 수 있음
display( df.drop_duplicates(['k1']) ) 
'''
	k1	k2	k3
0	one	1	0
3	two	3	3
'''

display( df.drop_duplicates(['k1','k2']) ) 
'''
	k1	k2	k3
0	one	1	0
2	one	2	2
3	two	3	3
5	two	4	5
'''
```



#### 대치

- 예제 데이터 생성

```python
import numpy as np
import pandas as pd

np.random.seed(1)
df = pd.DataFrame(np.random.randint(0,10,(6,4)))
df.index = pd.date_range('20200101', '20200106')
df.columns = ['A','B','C','D']
df['E'] = [7, np.nan, 4, np.nan, 2, np.nan]

display(df)
```

|          A |    B |    C |    D |    E |      |
| ---------: | ---: | ---: | ---: | ---: | ---- |
| 2020-01-01 |    5 |    8 |    9 |    5 | 7.0  |
| 2020-01-02 |    0 |    0 |    1 |    7 | NaN  |
| 2020-01-03 |    6 |    9 |    2 |    4 | 4.0  |
| 2020-01-04 |    5 |    2 |    4 |    2 | NaN  |
| 2020-01-05 |    4 |    7 |    7 |    9 | 2.0  |
| 2020-01-06 |    1 |    7 |    0 |    6 | NaN  |



##### replace()

- 원하는 값을 특정값으로 대체

- 표현식:

  df.replace(5, -100):

  ​	5 값을 -100 값으로 변경

```python
# df.replace(): 원하는 값을 특정값으로 대체
display( df.replace(5, -100) )
```

|            |    A |    B |    C |    D |    E |
| ---------: | ---: | ---: | ---: | ---: | ---: |
| 2020-01-01 | -100 |    8 |    9 | -100 |  7.0 |
| 2020-01-02 |    0 |    0 |    1 |    7 |  NaN |
| 2020-01-03 |    6 |    9 |    2 |    4 |  4.0 |
| 2020-01-04 | -100 |    2 |    4 |    2 |  NaN |
| 2020-01-05 |    4 |    7 |    7 |    9 |  2.0 |
| 2020-01-06 |    1 |    7 |    0 |    6 |  NaN |



### Grouping

- 예제 데이터 생성

```python
# Series, DataFrame의 Grouping

import numpy as np
import pandas as pd

my_dict = {
    '학과' : ['컴퓨터', '경영학과', '컴퓨터', '경영학과', '컴퓨터'],
    '학년' : [1, 2, 3, 2, 3],
    '이름' : ['홍길동', '신사임당', '김연아', '아이유', '강감찬'],
    '학점' : [1.5, 4.4, 3.7, 4.5, 4.2]
}

df = pd.DataFrame(my_dict)
display(df)
```

|      |     학과 | 학년 |     이름 | 학점 |
| ---: | -------: | ---: | -------: | ---: |
|    0 |   컴퓨터 |    1 |   홍길동 |  1.5 |
|    1 | 경영학과 |    2 | 신사임당 |  4.4 |
|    2 |   컴퓨터 |    3 |   김연아 |  3.7 |
|    3 | 경영학과 |    2 |   아이유 |  4.5 |
|    4 |   컴퓨터 |    3 |   강감찬 |  4.2 |



#### groupby()

- 특정 컬럼을 기준으로 그룹화

- 표현식:

  df[['학점', '이름']].groupby(df['학과'])

  ​	학과를 기준으로 학점과 이름 grouping

```python
# 학과를 기준으로 grouping
# groupby(): 특정 컬럼을 기준으로 묶기
score = df[['학점', '이름']].groupby(df['학과'])
```



##### get_group()

- 그룹 안의 데이터를 확인

- 표현식:

  score.get_group('컴퓨터')

```python
# 그룹 안에 데이터를 확인하고 싶은 경우 get_group() 함수 사용
print( score.get_group('컴퓨터') 
'''
    학점   이름
0  1.5  홍길동
2  3.7  김연아
4  4.2  강감찬
'''
```



##### size()

- 각 그룹 안에 몇 개의 데이터가 들어있는 지 Series로 리턴

- 표현식:

  score.size()

```python
# 각 그룹 안에 몇개의 데이터가 들어가 있나요?
print( score.size() ) # Series로 리턴됨
'''
학과
경영학과    2
컴퓨터     3
dtype: int64
'''
```



##### 집계함수

- mean(): 평균 추출

```python
# 집계함수
print( score.mean() )
'''
            학점
학과            
경영학과  4.450000
컴퓨터   3.133333
'''
```



##### 그룹 기준

- 그룹 기준이 두개일 경우

- unstack():

  최하위 인덱스를 column으로 변경

```python
score = df['학점'].groupby([df['학과'], df['학년']])
print(score)

print( score.size() )  # 결과가 Series로 리턴(멀티인덱스)
print( score.mean() )

# unstack(): 최하위 인덱스를 column으로 변경
display( score.mean().unstack() )
'''
학년	1	2	3
학과			
경영학과	NaN	4.45	NaN
컴퓨터	1.5	NaN	3.95
'''
```



- 데이터 프레임 전체를 그룹핑할 경우

```python
score = df.groupby(df['학과'])
print(score)
print(score.get_group('경영학과'))
print(score.size())
print(score.mean())
```



#### 간단한 퀴즈

1. 학과별 평균 학점
2. 학과별 학생 수

```python
# 학과별 평균 학점
df.groupby(df['학과'])['학점'].mean()

# 2. 학과별 몇 명?
df.groupby(df['학과'])['이름'].count()
```



#### 반복문으로 출력

```python
for dept, group in df.groupby(df['학과']):
    print(dept)
    display(group)
```

```
경영학과
```

|      |     학과 | 학년 |     이름 | 학점 |
| ---: | -------: | ---: | -------: | ---: |
|    1 | 경영학과 |    2 | 신사임당 |  4.4 |
|    3 | 경영학과 |    2 |   아이유 |  4.5 |

```
컴퓨터
```

|      |   학과 | 학년 |   이름 | 학점 |
| ---: | -----: | ---: | -----: | ---: |
|    0 | 컴퓨터 |    1 | 홍길동 |  1.5 |
|    2 | 컴퓨터 |    3 | 김연아 |  3.7 |
|    4 | 컴퓨터 |    3 | 강감찬 |  4.2 |