# day7_Python_TIL

## pandas

### pandas 개념

- pandas는 2개의 자료구조 사용

- numpy의 ndarray를 기본으로 그 위에 Series와 DataFrame이라는 자료구조를 정의해 사용
- Series는 DataFrame의 기본 구성요소

1. Series: 동일한 데이터 타입의 복수개의 성분으로 구성
2. DataFrame: Table 형식으로 데이터를 저장하는 자료구조



### Series

> ndarray + alpa인 자료구조

- 1차원 자료구조이며 같은 데이터 타입을 가짐
- ndarray 안에 다양한 데이터 타입이 있을 겨우엔 dtype=np.object로 생성
- s = pd.Series([1,2,3,4,5], dtype=np.float64) 와 같이 생성
- s.values: 시리즈의 값만 가져오기
- s.index: 시리즈의 인덱스만 가져오기

```python
import numpy as np
import pandas as pd

# ndarray(dtype: np.float64)
arr = np.array([1,2,3,4,5], dtype=np.float64)
print(arr)

# str말고 object 객체형태로 생성
# 객체 형태로 생성하면 object 형태로 들어가서 보기엔 정수 실수 논리형
# 타입 같이 보이나 사실은 객체 타입이다.
arr = np.array([1,3.14,True,'Hello'], dtype=np.object)
print(arr)
print(arr.dtype)

# Series
s = pd.Series([1,2,3,4,5],dtype=np.float64)
print(s)  # 1차원 자료구조, 인덱스와 데이터, 데이터타입 출력
'''
0    1.0
1    2.0
2    3.0
3    4.0
4    5.0
dtype: float64
'''

# Series의 값을 가져오면 ndarray로 표현
print(f"시리즈의 값만 가져오기 : {s.values}")    # [1. 2. 3. 4. 5.]
                                              # 1차원 vector 형태
print(f"시리즈의 인덱스만 가져오기 : {s.index}")  # rangeindex 형태 
# RangeIndex(start=0, stop=5, step=1)
```



#### Series의 index

- index 값에 문자열을 넣을 수 있다

  이를 인덱스값으로 사용하여 출력도 가능하며, 인덱스를 변경해도 숫자 인덱스는 사용 가능하다.

```python
import numpy as np
import pandas as pd


s = pd.Series([1,5,8,10],
              dtype=np.int32,
              index=['a','b','c','d']) # index에 문자열을 넣을 수 있다

print(s)
'''
a     1
b     5
c     8
d    10
dtype: int32
'''

# 5라는 값을 출력하려면?
print(s['b'])   # 5
result = s[1]   # 5, 인덱스를 변경해도 기본적으로 숫자 인덱스 사용 가능
print(result)
print(type(result)) # <class 'numpy.int32'>
```



- 새로 설정한 문자열 인덱스가 동일한 경우:

  동일한 인덱스의 값을 Serise 데이터 타입으로 리턴한다.

  단일값 출력시엔 데이터 타입이 int로 리턴

- Serise와 단일 int값 사칙연산 시엔 broadcasting 적용 됨

```python
# 만약 index가 같으면?
s = pd.Series([1,5,8,10],
              dtype=np.int32,
              index=['a','b','a','d']) # index가 같아도 되나?
print(s)
'''
a     1
b     5
a     8
d    10
dtype: int32
'''
result = s['a'] # 'a' 인덱스의 요소들을 Series로 추출
print(result)
                '''
                a    1
                a    8
                dtype: int32
                '''
    
print(type(result)) # <class 'pandas.core.series.Series'>
					# 복수값 출력시 Serise 데이터 타입으로 리턴

result = s['b'] # 단일값이면 'b' 인덱스의 요소를 int로 추출
print(result)       # 5
print(type(result)) # <class 'numpy.int32'>


# Serise와 단일 int 값의 사칙연산 시엔 broadcasting 적용됨
print(s['a'] + s['b'])
# s['a']: Series 1 8
# s['b']: int 5
# broadcasting 적용되어 
'''
a     6
a    13
dtype: int32
'''
```



##### fancy indexing, boolean indexing, 집계함수

- 모두 ndarray와 동일하게 동작

```python
s = pd.Series([1,5,8,10],
              dtype=np.int32,
              index=['a','b','c','d'])

# fancy indexing, boolean indexing
print(s[[0,2]])     # fancy indexing 사용 가능
'''
a    1
c    8
dtype: int32
'''

print(s[['a','c']]) # fancy indexing 사용 가능
'''
a    1
c    8
dtype: int32
'''

print(s[s % 2 == 0]) # boolean indexing 사용 가능
'''
c     8
d    10
dtype: int32
'''

# 집계함수
print(s.sum())       # 24
```





#### Serise의 slicing

- slicing은 python 리스트나 ndarray와 동일하게 동작

- 별도로 설정한 문자열 index로 slicing할 경우엔 시작과 끝 모두 inclusive 처리

```python
s = pd.Series([1,5,8,10],
              dtype=np.int32,
              index=['a','b','c','d'])

# Slicing
print(s[0:3]) # slicing은 그대로 적용
'''
a    1
b    5
c    8
dtype: int32
'''

print(s['a':'d']) # 별도의 index로 구현할 땐 둘 다 inclusive
'''
a     1
b     5
c     8
d    10
dtype: int32
'''
```



#### Series의 날짜 연산

- 날짜 연산은 함수를 이용해서 일, 월, 연, 주 단위로 증감 가능

- from datetime import detetime, timedelta

  datetime: 날짜 형식을 받아오는 함수

  timedelta: 날짜를 일, 월, 연, 주 단위로 증감시키는 함수

- Serise의 사칙연산은 같은 index를 기반으로 수행

  같은 index끼리만 연산이 되고 나머지 값들은 NaN을 출력

  NaN: Not a Number

  python에서 None은 pandas에서는 NaN으로 표현

  

- 예제:

  A 공장의 2020-01-01부터 10일간 생산량을 Series로 저장

  생산량은 평균이 50이고 표준편차가 5인 정규분포에서 랜덤하게 생성(정수)

  형식) 2020-01-01 52

  ​          2020-01-02 49

  ​          2020-01-03 55

  B공장의 2020-01-05부터 10일간 생산량을 Series로 저장

  생산량은 평균이 70이고 표준편차가 8인 정규분포에서 랜덤하게 생성(정수)

  날짜별로 모든(A공장, B공장)의 생산량 합계를 구하시오.

```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(1) # 생성한 값들이 변경되지 않도록 seed 설정

start_day = datetime(2020,1,1) # 시작날짜 지정
print(start_day)               # 2020-01-01 00:00:00

                      # for문을 이용하여 A공장의 생산량을 랜덤으로 추출 
factory_A = pd.Series([int(x) for x in np.random.normal(50,5,(10,))],
                      dtype=np.int32,
                      # index 값에 날짜 삽입
                      # timedelta(days=x):
                      # 변수 x를 for문을 이용해 1일씩 증감하는 날짜 삽입
                      index=[start_day + timedelta(days=x)
                             for x in range(10)])
print(factory_A)
'''
2020-01-01    58
2020-01-02    46
2020-01-03    47
2020-01-04    44
2020-01-05    54
2020-01-06    38
2020-01-07    58
2020-01-08    46
2020-01-09    51
2020-01-10    48
dtype: int32
'''

start_day = datetime(2020,1,5)
factory_B = pd.Series([int(x) for x in np.random.normal(70,8,(10,))],
                      dtype=np.int32,
                      index=[start_day + timedelta(days=x)
                             for x in range(10)])

print(factory_B)
'''
2020-01-05    81
2020-01-06    53
2020-01-07    67
2020-01-08    66
2020-01-09    79
2020-01-10    61
2020-01-11    68
2020-01-12    62
2020-01-13    70
2020-01-14    74
dtype: int32
'''


# Series의 사칙연산은 같은 index를 기반으로 수행
# 같은 인덱스끼리만 연산이 되고 나머지 값들은 NaN을 출력
# NaN: Not a Number
print(factory_A + factory_B)
'''
2020-01-01      NaN
2020-01-02      NaN
2020-01-03      NaN
2020-01-04      NaN
2020-01-05    135.0
2020-01-06     91.0
2020-01-07    125.0
2020-01-08    112.0
2020-01-09    130.0
2020-01-10    109.0
2020-01-11      NaN
2020-01-12      NaN
2020-01-13      NaN
2020-01-14      NaN
dtype: float64
'''
```



#### Series 생성

1. list를 이용한 생성 방법

```python
# Series를 리스트를 이용해서 만듦
s = pd.Series([1,5,8,10],
              dtype=np.int32,
              index=['a','b','c','d'])

print(s)
'''
a     1
b     5
c     8
d    10
dtype: int32
'''
```



2. dict를 이용한 생성방법

```python
# Series를 dict를 이용해서 만듦
my_dict = {'서울':1000, '인천':2000, '수원':3000}

s = pd.Series(my_dict)
print(s)
'''
서울    1000
인천    2000
수원    3000
dtype: int64
'''
```



##### Serise에 이름 지정

- 기능은 있지만 주로 사용되진 않음

```python
# Serise에 이름 지정하기
s.name = '지역별 가격 데이터'
print(s)
'''
서울    1000
인천    2000
수원    3000
Name: 지역별 가격 데이터, dtype: int64
'''


# 인덱스는 인덱스 객체이고 list는 아님, 하지만 list처럼 사용 가능
print(s.index) # Index(['서울', '인천', '수원'], dtype='object')


s.index = ['Seoul', 'Inchon', 'Suwon']
s.index.name = 'Region'
print(s)
'''
Region
Seoul     1000
Inchon    2000
Suwon     3000
Name: 지역별 가격 데이터, dtype: int64
'''
```



### DataFrame

> 1. Table 형식으로 데이터를 저장하는 자료구조

- 출력 시엔 print()가 아닌 display() 사용

- DataFrame 생성

```python
# DataFrame 만들기
# dict를 이용해서 DataFrame 생성
import numpy as np
import pandas as pd

# 데이터를 가지고 있는 dict 정의
my_dict = { 'name' : ['홍길동', '신사임당', '김연아', '강감찬'],
            'year' : [2015, 2016, 2019, 2016], # size 맞춰서 작성해야함
            'point' : [3.5, 1.4, 2.0, 4.5] }

df = pd.DataFrame(my_dict)
display(df)  # DataFrame을 출력할 때는 print보다 display를 이용

'''
name	year	point
0	홍길동	2015	3.5
1	신사임당	2016	1.4
2	김연아	2019	2.0
3	강감찬	2016	4.5
'''

print(df.shape)   # (4, 3)
print(df.size)    # 12
print(df.ndim)    # 2, DataFrame은 행과 열로 이루어진 2차원 자료구조
print(df.index)   # RangeIndex(start=0, stop=4, step=1)
print(df.columns) # Index(['name', 'year', 'point'], dtype='object')
print(df.values)  # 2차원 ndarray                  
                  # [['홍길동' 2015 3.5]
                  #  ['신사임당' 2016 1.4]
                  #  ['김연아' 2019 2.0]
                  #  ['강감찬' 2016 4.5]]
                    

# 이름 지정 가능 자주 쓰이진 않음            
df.index.name = '학번'    
df.columns.name = '정보'
display(df)
'''
정보	name	year	point
학번			
0	홍길동	2015	3.5
1	신사임당	2016	1.4
2	김연아	2019	2.0
3	강감찬	2016	4.5
'''
```



#### 데이터 표현 방식

> CSV, XML, JSON

- 많은 량의 데이터는 일반적으로 파일, Database, Open API를 이용해서 얻음
- 일반적으로 사용되는 데이터 표현 방식 (3가지)

1. CSV(Comma Seperated Values):

   ','로 구분되는 데이터 표현 방식

   ex) 홍길동,20,서울,김길동,30,인천,최길동,50,제주, ...

   장점: 많은 데이터를 표현하기에 적합, 데이터 용량이 적다

   단점: 데이터 구성을 알기 어려움, 구조적 데이터 표현이 힘들다

   ​          사용이 힘들고, 데이터 처리를 위해 따로 프로그램을 만들어야 함

   ​          데이터가 변경됐을 때 프로그램도 같이 변경 => 유지보수 문제가 발생

   데이터의 크기가 무지막지하고 데이터의 형태가 잘 변하지 않는 경우는 CSV형태가 가장 알맞은 형태

   

2. XML(eXtended Markup Language): 

   ex) <person><name>홍길동</name><age>20</age><address>서울</address></person>

   ​    <person><name>김길동</name><age>30</age><address>인천</address></person>

   ​    <person><name>최길동</name><age>50</age><address>제주</address></person>

   장점: 데이터 구성을 알기 쉬움, 사용하기 편함, 프로그램적 유지보수가 쉬움

   단점: 부가적인 데이터가 많다. 상대적으로 용량이 크다.

   

3. JSON(JavaScript Object Notation): 

   현재 일반적인 데이터 표현 방식

   자바스크립트 객체표현방식을 이용해서 데이터를 표현하는 방식

   JSON은 데이터 표현방식이며, 특정 프로그래밍 언어와 연관성은 없음

   ex) { 'name' : '홍길동', 'age' : 20, 'address' : '서울' }
         { 'name' : '김길동', 'age' : 30, 'address' : '인천' }
         { 'name' : '최길동', 'age' : 50, 'address' : '제주' }

   장점: 데이터 구성을 알기 쉬움, 사용하기 편함, 프로그램적 유지보수가 쉬움, XML보다 용량이 적음

   단점: CSV에 비해서는 부가적인 데이터가 많음



### DataFrame 생성

1. dict를 이용해서 수동으로 만드는 방법
2. CSV를 이용해서 파일로부터 데이터를 읽어들여 DataFrame을 만드는 방법
3. 기존 Database로부터 데이터를 읽어서 DataFrame을 만드는 방법



##### CSV 파일로 생성

```python
# CSV파일을 이용해서 DataFrame 만들기

import numpy as np
import pandas as pd

# read_csv: csv를 읽어오는 pandas 함수
df = pd.read_csv('D:\python_ML/movies.csv')

display(df.head())  # 상위 5개의 행을 출력
print(df.shape)     # (9742, 3)
```

![image-20210113175023261](C:%5CUsers%5Chyeonssss%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210113175023261.png)



##### Database로 생성

- MySQL 이용방법

  Database 설정하는 부분부터 MySQL 이용, 데이터베이스 구축

  MySQL의 설치파일이 아닌 단독실행파일을 이용해 실행 시킬 것

  1. MySQL 5.6 버전 다운로드

  2. 바탕화면에 다운로드

  3. cmd 창에서 bin 폴더 실행

  4. mysqld 명령어 실행 => MySQL DBMS를 실행시키는 명령어

  5. 끝나지 않고 계속 진행되면 정상 작동 중인 것

  6. 데이터베이스를 정상적으로 종료하려면 새로운 도스창을 띄운다

     mysqladmin -u root shutdown => MySQL DBMS를 종료시키는 명령어

  7. 새로운 사용자의 ID와 PW 설정

     mysql console에 접속(관리자 권한으로)

     mysql -u root => mysql console에 진입하는 명령어

  8. mysql> 이라는 프롬프트가 보이면 접속 성공

  9. 정상적으로 접속했으면 새로운 사용자를 생성

     create user data identified by "data"; => ID와 PW설정

  10. 외부접속을 위해 명령어 실행

      create user data@localhost identified by "data";

  11. 데이터가 저장될 데이터베이스 생성

      우리가 사용하는 mysql은 DBMS(DataBase Management system)

      여러개의 데이터베이스를 관리하는 프로그램

      create database library;

  12. 사용자(data)에게 데이터베이스(library)의 사용 권한을 부여

      grant all privileges on library.* to data;

      grant all privileges on library.* to data@localhost;

  13. 권한 설정에 대한 refresh를 시켜줘요!

      flush privileges;

  14. 설정이 끝나면 console에서 나온다

      exit; 명령어 실행

      도스창에서 제공된 script file을 이용해서 데이터를 적재

      mysql -u data -p library < _BookTableDump.sql

      password: data 입력

