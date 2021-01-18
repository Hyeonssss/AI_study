# day9_pandas_TIL

## pandas

### Open API로 DataFrame 생성

- request = URL을 이용해서 호출하는 행위
- response = request의 결과를 웹서버에서 우리에게 전달하는 행위
- urllib.request.urlopen(open_api_url) => response 객체
- json.loads (page_obj.read()) => json을 얻어오는 함수

```python
import numpy as np
import pandas as pd
import json
import urllib

# Open API URL: 
open_api = 'http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json'
query_string = '?key=682ca358e95538a21f1702baa2ddb218&targetDt=20210114'
open_api_url = open_api + query_string


# request = URL을 이용해서 호출하는 행위
# response = request의 결과를 웹서버에서 우리에게 전달하는 행위
# 
page_obj = urllib.request.urlopen(open_api_url)

# print(type(page_obj))  # response 객체
# json_page = json.loads(page_obj.read()) # 결과 객체 안에 들어있는 json을 얻어올 수 있음
# print(json_page)
# print(type(json_page))

# 결과 json을 python의 dict로 변환
# 원하는 내용을 뽑아서 DataFrame으로 생성
# my_dict = { 'rank' : ['홍길동','김길동'],
#             '영화제목' : [20, 30],
#             '당일매출액' : ['서울','부산']}
```



- json 파일에서 원하는 컬럼만 뽑아오기

  각각의 리스트를 만들고 반복문을 통해 리스트에 삽입

  전체를 포함하는 딕셔너리에 삽입하여 DataFrame으로 변환

```python
my_dict    = dict()
rank_list  = list()
title_list = list()
sales_list = list()

for tmp_dict in json_page['boxOfficeResult']['dailyBoxOfficeList']:
    rank_list.append(tmp_dict['rank'])
    title_list.append(tmp_dict['movieNm'])
    sales_list.append(tmp_dict['salesAmt'])

my_dict['순위'] = rank_list
my_dict['제목'] = title_list
my_dict['매출액'] = sales_list
    
df = pd.DataFrame(my_dict)
display(df)
```

![09_1](D:%5CAI_study%5Cmd-images%5C09_1.JPG)



### DataFrame 생성 마무리

1. dict으로 직접 데이터를 입력해 만드는 방법
2. CSV파일로 만드는 방법
3. Database로 만드는 방법
4. Open API로 만드는 방법



### DataFrame 사용

- warnings: 경고문구를 무시할 수 있음

```python
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings(action='ignore') # warning off
# warnings.filterwarnings(action='default') # warning on

data = { '이름' : ['홍길동','신사임당','강감찬','아이유','김연아'],
         '학과' : ['컴퓨터', '철학', '수학', '영어영문', '통계'],
         '학년' : [1, 2, 3, 4, 5],
         '학점' : [1.3, 3.5, 2.7, 4.3, 4.5]}

df = pd.DataFrame(data,
                  columns=['학과', '이름', '학년', '학점', '주소'],
                # columns=['학과', '이름', '학년', '평점'])
                # 이름에 오타가 날 경우 결측치(NaN)가 출력됨
                  index=['one', 'two', 'three', 'four', 'five'])
                # Series와 동일하게 문자열 인덱스 생성 가능
                # 숫자 인덱스는 내부적으로 존재
        
display(df)
```



- 예제 데이터

```python
import numpy as np
import pandas as pd

data = { '이름' : ['홍길동','신사임당','강감찬','아이유','김연아'],
         '학과' : ['컴퓨터', '철학', '수학', '영어영문', '통계'],
         '학년' : [1, 2, 3, 4, 5],
         '학점' : [1.3, 3.5, 2.7, 4.3, 4.5]}

df = pd.DataFrame(data,
                  columns=['학과', '이름', '학년', '학점'],
                  index=['one', 'two', 'three', 'four', 'five'])
        
```

![09_2](D:%5CAI_study%5Cmd-images%5C09_2.JPG)


#### column 제어

##### indexing

>  예제 데이터 활용

- 단일 column 출력시 Series로 출력

```python
print(df['이름'])   # Series로 출력
                    # one       홍길동
                    # two      신사임당
                    # three     강감찬
                    # four      아이유
                    # five      김연아
                    # Name: 이름, dtype: object

print(type(df['이름'])) # <class 'pandas.core.series.Series'>
```



- view가 생성되는 것이기 때문에 수정 시엔 원본도 함께 변경

```python
## 원본데이터의 view 생성
stu_name = df['이름']          # 원본데이터의 view가 생성
stu_name['three'] = '을지문덕' # view이기 때문에 수정시 원본도 수정
print(stu_name)
display(df)

## 별도의 객체 생성하기
stu_name = df['이름'].copy()   # 별도의 객체 생성
stu_name['three'] = '을지문덕' # 데이터를 변경해도 원본에는 이상 없음
print(stu_name)
display(df)
```



##### fancy indexing

> 예제 데이터 활용

- 두 개 이상의 column을 가져오기 위해선 fancy indexing 사용

```python
# column을 두 개 이상 가져오려면 fancy indexing
df[['이름','학점']]  # DataFrame 형태의 view 생성
                     # Series는 1차원이기 때문에 column 두개 표현 못함

df[['이름']]         # DataFrame으로 출력
                     # fancy indexing은 slicing 기반으로 원본 형태와 동일
```



##### column 수정, 삽입

```python
df['등급'] = 'A' # broadcasting 적용돼서 수행

# list만 줘도 내부적으로 np.array 변경되어 적용
df['등급'] = ['A','C','B','A','C'] 
display(df)

df['등급'] = np.array(['A','C','B','A','C']) # 조금 더 정확한 표현
display(df)

# 사이즈 안 맞으면 에러, 결측값에 명확히 지정해줘야함
# np.nan: 결측값을 표현하는 상수
df['등급'] = ['A','C','B','A',np.nan] 
display(df)
display(df.to_numpy())
```



- column 추가하기

  - 행의 개수를 맞춰야 새로운 column을 생성할 수 있음

  - Series로 추가하려면 문자열 인덱스를 맞춰야 함

    그러나 index 기반이기 때문에 행의 개수는 맞추지 않아도 됨

    값이 없는 행은 NaN으로 출력됨 

  - NaN의 기본 데이터타입: float64 

```python
# DataFrame에 column 추가하기
df['등급'] = np.arange(1,10,2) # 5개의 값을 이용
df['나이'] = [15,20,12,30]     # 행의 개수가 맞지 않으면 오류

# DataFrame과 Series를 연산하려면 인덱스를 맞춰줘야 한다.
df['나이'] = pd.Series([15,20,25,30,35], # 문자열 인덱스도 맞춰야 함
                        index=['one', 'two', 'three', 'four', 'five'])

# index를 기반으로 처리되기 때문에 아래와 같은 경우도 허용
# Series는 개수를 맞추지 않아도 됨
# NaN은 결측값이지만 데이터 타입은 실수로 간주가 됨
df['나이'] = pd.Series([15,20,25], # 문자열 인덱스도 맞춰야 함
                        index=['one', 'three', 'four'])

 
display(df)
```



- 연산으로 새로운 컬럼 추가하기

```python
# 연산으로 새로운 컬럼 추가
df['장학생여부'] = df['학점'] > 4.0
df
'''
	학과	이름	학년	학점	장학생여부
one	컴퓨터	홍길동	1	1.3	False
two	철학	신사임당	2	3.5	False
three	수학	강감찬	3	2.7	False
four	영어영문	아이유	4	4.3	True
five	통계	김연아	5	4.5	True
'''
```



##### column 삭제

> 예제 데이터 활용

- drop: column(열)이나 row·record(행) 삭제 가능

  axis를 명시해야함 axis=0은 행, axis=1은 열

- inplace: True => 원본에서 삭제하는 경우

  ​			   False => 원본을 보존하고 삭제처리된 복사본을 생성하는 경우

  일반적으로 원본이 훼손되는 것을 꺼려함

  inplace의 default 값은 inplace=False

```python
# 학년 컬럼 전체를 삭제
new_df = df.drop('학년', axis=1, inplace=False)

display(new_df)
```



##### column indexing 정리

- column은 slicing이 불가
- 단일 index, fancy indexing은 가능

```python
# column indexing
print(df['이름'])            # OK, Series로 결과 리턴
print(df[['이름':'학년']])    # Error, column은 slicing이 안됨
display(df[['이름','학년']])  # OK, fancy indexing은 가능
# boolean indexing은 column과는 상관이 없음. row undexing할 때 사용
```



#### record 제어

> 예제 데이터 활용

##### 숫자 index로 row indexing

- 숫자 index로 단일 indexing 불가
- slicing을 활용하면 가능 => view로 생성
- fancy indexing은 불가

```python
# Row indexing (숫자 indexing을 이용한 row indexing)
print(df[1])      # 행에 대한 숫자 인덱스로 단일 indexing이 안됨
display(df[1:2])  # slicing은 가능 => view
                  # slicing의 결과는 대부분 뷰
display(df[1:])
display(df[[1,3]]) # row의 fancy indexing은 Error
```



##### inex로 row indexing

- 단일 indexing 불가: column 인덱싱으로 간주

- slicing 가능: 숫자 인덱스와 다르게 문자 인덱스는 앞과 뒤 모두 포함

  ​					  문자와 숫자 인덱스 혼용은 안됨

- fancy indexing은 불가

```python
# Row indexing(index를 이용해서)
print(df['two']) # Error, 이 표현식은 column 인덱싱 방법
                 # 행에 대한 index를 이용한 단일 row 추출은 안됨
display(df['two':'four'])  # OK, 일반 인덱스로도 slicing 가능
                           # 문자열 인덱스는 앞과 뒤 둘 다 포함
display(df['two':-1])      # 문자열과 숫자 인덱스 혼용은 안됨
display(df[['one','three']]) # Error, column에 대한 fancy indexing
```



#### loc[]

> 예제 데이터 활용

- row indexing의 별도 표기법
- 단일 row 추출가능
- 숫자 index는 사용 불가
- slicing 가능
- fancy indexing 가능

```python
# loc[]를 이용해서 row indexing
# loc[]를 이용하면 단일 row도 추출할 수 있음
# loc[]는 숫자 index 사용 불가
display(df.loc['two'])     # Series로 리턴

# 둘 다 inclusive 두개의 행이 나오기 때문에 DataFrame으로 리턴
display(df.loc['two':'three'])    # OK, 결과는 DataFrame
display(df.loc[['two','four']])   # OK, fancy indexing도 가능

```



##### iloc[]

> 예제 데이터 활용

- 숫자 index를 사용할 수 있는 iloc[]
- 단일 row 추출 가능
- slicing 가능
- fancy indexing 가능

```python
display(df.iloc[1])     # iloc를 이용하면 단일 row 추출
display(df.iloc[1:3])   # OK
display(df.iloc[[0,3]]) # OK, Fancy indexing
```



##### loc[]로 행과 열 제어

> 예제 데이터 활용

- 열이나 행이 하나일 경우 Series로 출력
- 행과 열 둘 다 slicing 가능

```python
display(df.loc['one' : 'three'])        # OK, row indexing
display(df.loc['one' : 'three', '이름']) #OK, row indexong => Series

# loc를 이용하면 행과 열 둘 다 slicing 가능
display(df.loc['one' : 'three', '이름' : '학년']) # OK
display(df['이름' : '학년'])  # Error, column은 slicing 불가
display(df.loc['one' : 'three', ['이름', '학점']]) # slicing과 fancy indexing 혼용도 가능
```



##### boolean indexing

> 예제 데이터 활용

- boolean indexing은 row에만 적용되고 column에는 적용되지 않음

  df.loc[row=이부분만 쓸수 있음, column=불가능]

```python
# 학점이 4.0을 초과하는 학생의 이름과 학점을 DataFrame으로 출력
df['학점'] > 4.0  # boolean mask
display(df.loc[df['학점'] > 4.0, ['이름','학점']])
```



##### row 추가

> 예제 데이터 활용

```python
# loc를 이용한 row 추가하기
# 존재하지 않는 row인 six를 추가
df.loc['six','이름':'학점'] = ['김길동', 3, 3.7]
display(df)
```



##### 삭제 

- column 삭제

```python
# column 삭제
df.drop('학년', axis=1, inplace=True)
display(df)
```



- row 삭제
  - axis를 명시하지 않으면 default값은 axis=0(행)
  - inplace=False가 기본, 원본은 변경하지 않고 복사본에 저장해야함

```python
# row 삭제
df.drop('two', axis=0, inplace=True)  
display(df)
```



- fancy indexing

```python
# fancy indexing
df.drop(['one', 'three'], axis=0, inplace=True)
display(df)
```



- slicing
  - slicing은 사용 불가

```python
# slicing
# slicing은 사용 불가
df.drop('one':'three', axis=0, inplace=True)
display(df) # Error
'''
 File "<ipython-input-179-12ec29352fac>", line 34
    df.drop('one':'three', axis=0, inplace=True)
                 ^
SyntaxError: invalid syntax
'''
```





#### 간단한 퀴즈

1. 이름이 박동훈인 사람을 찾아 이름과 학점을 DataFrame으로 출력
2. 학점이 (1.5,2.5)인 사람을 찾아 학과, 이름, 학점을 DataFrame으로 출력
3. 학점이 3.0을 초과하는 사람을 찾아 등급을 'A'로 설정하세요!

```python
import numpy as np
import pandas as pd

data = {'이름' : ['이지은', '박동훈', '홍길동', '강감찬', '오혜영'],
        '학과' : ['컴퓨터', '기계', '철학', '컴퓨터', '철학'],
        '학년' : [1, 2, 2, 4, 3],
        '학점' : [1.5, 2.0, 3.1, 1.1, 2.7],}

df = pd.DataFrame(data,
                  columns=['학과', '이름', '학점','학년', '등급'],
                  index = ['one', 'two', 'three' ,'four', 'five'])

display(df)
```

![09_3](D:%5CAI_study%5Cmd-images%5C09_3.JPG)



- 풀이

  - and 연산은 pandas에서 &로 표현

    각각의 조건식에 괄호() 필요

```python
# 1.
display(df.loc[df['이름'] == '박동훈', ['이름', '학점']])
'''
	이름	학점
two	박동훈	2.0
'''

# 2.
display(df.loc[(df['학점'] > 1.5) & (df['학점'] < 2.5), '학과':'학점'])
# and 연산은 pandas에서 &로 표현, 각각의 조건식에 () 필요    
'''
	학과	이름	학점
two	기계	박동훈	2.0
'''    
    
    
# 3.
df.loc[df['학점'] > 3.0,'등급'] = 'A'

display(df)
'''
	학과	이름	학점	학년	등급
one	컴퓨터	이지은	1.5	1	NaN
two	기계	박동훈	2.0	2	NaN
three	철학	홍길동	3.1	2	A
four	컴퓨터	강감찬	1.1	4	NaN
five	철학	오혜영	2.7	3	NaN
'''
```



