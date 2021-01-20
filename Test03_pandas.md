# Test03_pandas

## pandas 연습문제



### 영화 data

#### 데이터 로드

```python
# movie 수행평가
# movieId로 merge

# timestamp => 날짜를 숫자로 표현하는 표준
# 1970년 1월1일0시0분0초 => 0
# 1970년 1월1일0시0분1초 => 1
# 1970년 1월1일0시0분2초 => 2

import numpy as np
import pandas as pd

# D:\python_ML\ml-latest-small
movie_df = pd.read_csv('D:\python_ML\ml-latest-small/movies.csv')
rating_df = pd.read_csv('D:\python_ML\ml-latest-small/ratings.csv')

display(movie_df)
display(rating_df)
```

|      | movieId |                                     title | genres                                          |
| ---: | ------: | ----------------------------------------: | ----------------------------------------------- |
|    0 |       1 |                          Toy Story (1995) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
|    1 |       2 |                            Jumanji (1995) | Adventure\|Children\|Fantasy                    |
|    2 |       3 |                   Grumpier Old Men (1995) | Comedy\|Romance                                 |
|    3 |       4 |                  Waiting to Exhale (1995) | Comedy\|Drama\|Romance                          |
|    4 |       5 |        Father of the Bride Part II (1995) | Comedy                                          |
|  ... |     ... |                                       ... | ...                                             |
| 9737 |  193581 | Black Butler: Book of the Atlantic (2017) | Action\|Animation\|Comedy\|Fantasy              |
| 9738 |  193583 |              No Game No Life: Zero (2017) | Animation\|Comedy\|Fantasy                      |
| 9739 |  193585 |                              Flint (2017) | Drama                                           |
| 9740 |  193587 |       Bungo Stray Dogs: Dead Apple (2018) | Action\|Animation                               |
| 9741 |  193609 |       Andrew Dice Clay: Dice Rules (1991) | Comedy                                          |

9742 rows × 3 columns



|        | userId | movieId | rating |  timestamp |
| -----: | -----: | ------: | -----: | ---------: |
|      0 |      1 |       1 |    4.0 |  964982703 |
|      1 |      1 |       3 |    4.0 |  964981247 |
|      2 |      1 |       6 |    4.0 |  964982224 |
|      3 |      1 |      47 |    5.0 |  964983815 |
|      4 |      1 |      50 |    5.0 |  964982931 |
|    ... |    ... |     ... |    ... |        ... |
| 100831 |    610 |  166534 |    4.0 | 1493848402 |
| 100832 |    610 |  168248 |    5.0 | 1493850091 |
| 100833 |    610 |  168250 |    5.0 | 1494273047 |
| 100834 |    610 |  168252 |    5.0 | 1493846352 |
| 100835 |    610 |  170875 |    3.0 | 1493846415 |

100836 rows × 4 columns



#### 1.

- 사용자가 평가한 모든 영화의 전체 평균 평점

```python
# 사용자가 평가한 모든 영화이니 movieId를 기준으로 right merge 후 rating 평균 출력
print( pd.merge(movie_df, rating_df, on='movieId', how='right')['rating'].mean() )
```

- ```
  3.501556983616962
  ```



#### 2.

- 각 사용자별 평균 평점

```python
# userId별로 그룹화한 후 평균 rating 출력
print( rating_df.groupby(rating_df['userId'])['rating'].mean() )
```

- ```
  userId
  1      4.366379
  2      3.948276
  3      2.435897
  4      3.555556
  5      3.636364
           ...   
  606    3.657399
  607    3.786096
  608    3.134176
  609    3.270270
  610    3.688556
  Name: rating, Length: 610, dtype: float64
  ```



#### 3.

- 각 영화별 평균 평점

```python
# rating_df의 rating을 movieId 그룹화
result = rating_df['rating'].groupby(rating_df['movieId']).mean()

# result를 movie_df와 movieId 기준으로 merge
# 누락되는 영화가 없도록 left merge 이용
df = pd.merge(movie_df, result,
              left_on = 'movieId',
              right_index = True,
              how = 'left')

# df의 movieId, title, rating을 요소로 갖는 DataFrame(result) 생성
result = df[['movieId','title','rating']]

display(result)
```

- | movieId |  title |                                    rating |          |
  | ------: | -----: | ----------------------------------------: | -------- |
  |       0 |      1 |                          Toy Story (1995) | 3.920930 |
  |       1 |      2 |                            Jumanji (1995) | 3.431818 |
  |       2 |      3 |                   Grumpier Old Men (1995) | 3.259615 |
  |       3 |      4 |                  Waiting to Exhale (1995) | 2.357143 |
  |       4 |      5 |        Father of the Bride Part II (1995) | 3.071429 |
  |     ... |    ... |                                       ... | ...      |
  |    9737 | 193581 | Black Butler: Book of the Atlantic (2017) | 4.000000 |
  |    9738 | 193583 |              No Game No Life: Zero (2017) | 3.500000 |
  |    9739 | 193585 |                              Flint (2017) | 3.500000 |
  |    9740 | 193587 |       Bungo Stray Dogs: Dead Apple (2018) | 3.500000 |
  |    9741 | 193609 |       Andrew Dice Clay: Dice Rules (1991) | 4.000000 |

  9742 rows × 3 columns



#### 4.

- 평균 평점이 가장 높은 영화의 제목(동률이 있을 경우 모두 출력)

```python
# rating_df의 rating을 movieId 그룹화
result = rating_df['rating'].groupby(rating_df['movieId']).mean()

# result를 movie_df와 movieId 기준으로 merge
# 누락되는 영화가 없도록 left merge 이용
df = pd.merge(movie_df, result,
              left_on = 'movieId',
              right_index = True,
              how = 'left')

# df의 movieId, title, rating을 요소로 갖는 DataFrame(result) 생성
result = df[['movieId','title','rating']]

# loc[]를 이용하여 rating 값이 rating.max()값과 동일한 record만 출력
display( result.loc[result['rating'] == result['rating'].max(),:] )
```

- |      | movieId |                                             title | rating |
  | ---: | ------: | ------------------------------------------------: | ------ |
  |   48 |      53 |                                   Lamerica (1994) | 5.0    |
  |   87 |      99 |              Heidi Fleiss: Hollywood Madam (1995) | 5.0    |
  |  121 |     148 |                  Awfully Big Adventure, An (1995) | 5.0    |
  |  405 |     467 |                            Live Nude Girls (1995) | 5.0    |
  |  432 |     495 | In the Realm of the Senses (Ai no corrida) (1976) | 5.0    |
  |  ... |     ... |                                               ... | ...    |
  | 9611 |  176601 |                                      Black Mirror | 5.0    |
  | 9633 |  179133 |                             Loving Vincent (2017) | 5.0    |
  | 9634 |  179135 |                             Blue Planet II (2017) | 5.0    |
  | 9688 |  184245 |                           De platte jungle (1978) | 5.0    |
  | 9711 |  187717 |                  Won't You Be My Neighbor? (2018) | 5.0    |

  296 rows × 3 columns



#### 5.

- Comedy영화 중 가장 평점이 낮은 영화의 제목

```python
# rating_df의 rating을 movieId 그룹화
result = rating_df['rating'].groupby(rating_df['movieId']).mean()

# result를 movie_df와 movieId 기준으로 merge
# 평점이 없는 영화를 제외하기 위해 inner merge 사용
df = pd.merge(movie_df, result,
              left_on = 'movieId',
              right_index = True,
              how = 'inner') 

# df의 movieId, title, genres, rating을 요소로 갖는 DataFrame(result) 생성
result = df[['movieId', 'title', 'genres','rating']]

# .str.contains()를 사용하여 genres 안에 Comedy가 포함되는 record만 추출한
# DataFrame(comedy_df) 생성
comedy_df = result[result.genres.str.contains('Comedy')]

# loc[]를 이용하여 rating 값이 rating.min()값과 동일한 record만 출력
display( comedy_df.loc[comedy_df['rating'] == comedy_df['rating'].min(),:] )
```

- |      | movieId |                                             title |                                  genres | rating |
  | ---: | ------: | ------------------------------------------------: | --------------------------------------: | -----: |
  | 4439 |    6557 |                            Born to Be Wild (1995) |      Adventure\|Children\|Comedy\|Drama |    0.5 |
  | 4881 |    7312 |                           Follow Me, Boys! (1966) |                           Comedy\|Drama |    0.5 |
  | 5258 |    8632 |                             Secret Society (2002) |                                  Comedy |    0.5 |
  | 5409 |   25782 | Boudu Saved From Drowning (Boudu sauvé des eau... |                                  Comedy |    0.5 |
  | 5453 |   26095 |        Carabineers, The (Carabiniers, Les) (1963) |                      Comedy\|Drama\|War |    0.5 |
  | 5662 |   27595 |                Jesus Christ Vampire Hunter (2001) |         Action\|Comedy\|Horror\|Musical |    0.5 |
  | 5777 |   31422 |                          Are We There Yet? (2005) |                        Children\|Comedy |    0.5 |
  | 5795 |   31692 |                                 Uncle Nino (2003) |                                  Comedy |    0.5 |
  | 6160 |   44243 |                     Leprechaun 4: In Space (1997) |         Comedy\|Fantasy\|Horror\|Sci-Fi |    0.5 |
  | 6545 |   54768 |                             Daddy Day Camp (2007) |                        Children\|Comedy |    0.5 |
  | 6554 |   54934 |                      Brothers Solomon, The (2007) |                                  Comedy |    0.5 |
  | 6784 |   60363 |                          Zombie Strippers! (2008) |                          Comedy\|Horror |    0.5 |
  | 7201 |   72696 |                                   Old Dogs (2009) |                                  Comedy |    0.5 |
  | 7553 |   85334 |                      Hard Ticket to Hawaii (1987) |                          Action\|Comedy |    0.5 |
  | 7762 |   91414 |                           Arthur Christmas (2011) |      Animation\|Children\|Comedy\|Drama |    0.5 |
  | 7820 |   92681 |           Journey 2: The Mysterious Island (2012) | Action\|Adventure\|Comedy\|Sci-Fi\|IMAX |    0.5 |
  | 8248 |  104644 | Maria Bamford: The Special Special Special! (2... |                                  Comedy |    0.5 |
  | 8417 |  110773 |                         Haunted House 2, A (2014) |                          Comedy\|Horror |    0.5 |
  | 8676 |  122246 |                              Tooth Fairy 2 (2012) |                        Children\|Comedy |    0.5 |
  | 8893 |  134528 |                                      Aloha (2015) |                  Comedy\|Drama\|Romance |    0.5 |
  | 8908 |  135216 |              The Star Wars Holiday Special (1978) |     Adventure\|Children\|Comedy\|Sci-Fi |    0.5 |
  | 8984 |  138798 |                Joe Dirt 2: Beautiful Loser (2015) |                                  Comedy |    0.5 |
  | 9056 |  141994 |                           Saving Christmas (2014) |                        Children\|Comedy |    0.5 |
  | 9419 |  165645 |                                Bad Santa 2 (2016) |                                  Comedy |    0.5 |
  | 9590 |  175475 |                            The Emoji Movie (2017) |             Animation\|Children\|Comedy |    0.5 |



#### 6.

- 2015년도에 평가된 모든 Romance 영화의 평균 평점은?

```python
# movie_df와 rating_df를 movieId를 기준으로 merge
# user에 의해 평가된 영화만 구하기 위해 right merge 사용
df = pd.merge(movie_df, rating_df, on='movieId', how='right')

# to_datetime()을 이용하여 timestamp를 datetime으로 변환
df['date'] = pd.to_datetime(df['timestamp'], unit='s')

# .strftime()을 이용하여 datetime에서 년도만 추출
df['year'] = df['date'].dt.strftime('%Y')

# df의 movieId, genres, rating, year를 요소로 갖는 DataFrame(result) 생성
result = df[['movieId', 'genres','rating', 'year']]

# .str.contains()를 사용하여 genres 안에 Romance가 포함되는 record만 추출한
# DataFrame(romance_df) 생성
romance_df = result[result.genres.str.contains('Romance')]

# loc[]를 이용하여 year가 2015년도인 record들의 rating 평균을 출력
display( romance_df.loc[romance_df['year'] == '2015',:]['rating'].mean() )
```

- ```
  3.396375098502758
  ```



### mpg data

#### 데이터 로드

```python
import numpy as np
import pandas as pd

df = pd.read_csv('D:\python_ML/mpg.txt')

display(df)

# 주요컬럼 설명
# manufacturer : 제조회사
# displ : 배기량
# cyl : 실린더 개수
# drv : 구동 방식
# hwy : 고속도로 연비
# class : 자동차 종류
# model : 자동차 모델명
# year : 생산연도
# trans : 변속기 종류
# cty : 도시 연비
# fl : 연료 종류
```

- |      | manufacturer |  model | displ | year |  cyl |      trans |  drv |  cty |  hwy |   fl | class   |
  | ---: | -----------: | -----: | ----: | ---: | ---: | ---------: | ---: | ---: | ---: | ---: | ------- |
  |    0 |         audi |     a4 |   1.8 | 1999 |    4 |   auto(l5) |    f |   18 |   29 |    p | compact |
  |    1 |         audi |     a4 |   1.8 | 1999 |    4 | manual(m5) |    f |   21 |   29 |    p | compact |
  |    2 |         audi |     a4 |   2.0 | 2008 |    4 | manual(m6) |    f |   20 |   31 |    p | compact |
  |    3 |         audi |     a4 |   2.0 | 2008 |    4 |   auto(av) |    f |   21 |   30 |    p | compact |
  |    4 |         audi |     a4 |   2.8 | 1999 |    6 |   auto(l5) |    f |   16 |   26 |    p | compact |
  |  ... |          ... |    ... |   ... |  ... |  ... |        ... |  ... |  ... |  ... |  ... | ...     |
  |  229 |   volkswagen | passat |   2.0 | 2008 |    4 |   auto(s6) |    f |   19 |   28 |    p | midsize |
  |  230 |   volkswagen | passat |   2.0 | 2008 |    4 | manual(m6) |    f |   21 |   29 |    p | midsize |
  |  231 |   volkswagen | passat |   2.8 | 1999 |    6 |   auto(l5) |    f |   16 |   26 |    p | midsize |
  |  232 |   volkswagen | passat |   2.8 | 1999 |    6 | manual(m5) |    f |   18 |   26 |    p | midsize |
  |  233 |   volkswagen | passat |   3.6 | 2008 |    6 |   auto(s6) |    f |   17 |   26 |    p | midsize |

  234 rows × 11 columns



#### 1.

- displ(배기량)이 4 이하인 자동차와 5 이상인 자동차 중 어떤 자동차의 hwy(고속도로 연비)가 평균적으로 더 높은지 확인

```python
print( f"배기량 4이하인 자동차 : {df.loc[df['displ'] <= 4,:]['hwy'].mean()}" )
print( f"배기량 5이상인 자동차 : {df.loc[df['displ'] >= 5,:]['hwy'].mean()}" )
```

- ```
  배기량 4이하인 자동차 : 25.96319018404908
  배기량 5이상인 자동차 : 18.07894736842105
  ```



#### 2.

- "audi"와 "toyota" 중 어느 manufacturer(제조회사)의 cty(도시 연비)가 평균적으로 더 높은지 확인

```python
print( f"audi의 도시연비평균 : {df.loc[df['manufacturer'] == 'audi',:]['cty'].mean()}" )
print( f"audi의 도시연비평균 : {df.loc[df['manufacturer'] == 'toyota',:]['cty'].mean()}" )
```

- ```
  audi의 도시연비평균 : 17.61111111111111
  audi의 도시연비평균 : 18.529411764705884
  ```



#### 3.

- "chevrolet", "ford", "honda" 자동차의 고속도로 연비 평균

```python
print(f"hwy 평균 : {df.loc[df['manufacturer'].isin(['chevrolet', 'ford', 'honda']),:]['hwy'].mean()}")
```

- ```
  hwy 평균 : 22.50943396226415
  ```



#### 4.

- "audi"에서 생산한 자동차 중 hwy가 1~5위

```python
display( df.loc[df['manufacturer'] == 'audi',:].sort_values(by='hwy', ascending=False).head() )
```

- |      | manufacturer |      model | displ | year |  cyl |      trans |  drv |  cty |  hwy |   fl |   class |
  | ---: | -----------: | ---------: | ----: | ---: | ---: | ---------: | ---: | ---: | ---: | ---: | ------: |
  |    2 |         audi |         a4 |   2.0 | 2008 |    4 | manual(m6) |    f |   20 |   31 |    p | compact |
  |    3 |         audi |         a4 |   2.0 | 2008 |    4 |   auto(av) |    f |   21 |   30 |    p | compact |
  |    0 |         audi |         a4 |   1.8 | 1999 |    4 |   auto(l5) |    f |   18 |   29 |    p | compact |
  |    1 |         audi |         a4 |   1.8 | 1999 |    4 | manual(m5) |    f |   21 |   29 |    p | compact |
  |    9 |         audi | a4 quattro |   2.0 | 2008 |    4 | manual(m6) |    4 |   20 |   28 |    p | compact |



#### 5.

- 회사별로 "suv" 자동차의 평균 연비를 구한후 내림차순으로 정렬한 후 1~5위

```python
df['avg_fuel_efficiency'] = (df['cty'] + df['hwy']) / 2
suv_df = df.loc[df['class'] == 'suv',:]
result = suv_df[['manufacturer', 'avg_fuel_efficiency']]

display( result.sort_values(by='avg_fuel_efficiency', ascending=False).head() )
```

- |      | manufacturer | avg_fuel_efficiency |
  | ---: | -----------: | ------------------: |
  |  161 |       subaru |                23.5 |
  |  163 |       subaru |                23.0 |
  |  162 |       subaru |                22.0 |
  |  159 |       subaru |                21.5 |
  |  160 |       subaru |                21.0 |



#### 6.

- class별 cty 평균을 구하고 cty 평균이 높은 순

```python
result = df['cty'].groupby(df['class']).mean()

display( result.sort_values(ascending=False) )
```

- ```
  class
  subcompact    20.371429
  compact       20.127660
  midsize       18.756098
  minivan       15.818182
  2seater       15.400000
  suv           13.500000
  pickup        13.000000
  Name: cty, dtype: float64
  ```



#### 7.

- hwy(고속도로 연비) 평균이 가장 높은 회사 세 곳

```python
result = df['hwy'].groupby(df['manufacturer']).mean()

display( result.sort_values(ascending=False).head(3) )
```

- ```
  manufacturer
  honda         32.555556
  volkswagen    29.222222
  hyundai       26.857143
  Name: hwy, dtype: float64
  ```



#### 8.

- 각 회사별 "compact" 차종 수를 내림차순

```python
# compact 차종만 들어있는 compact_df 생성
compact_df = df.loc[df['class'] == 'compact',:]

# compact_df를 manufacturer로 그룹화한 뒤 개수를 출력
result = compact_df['class'].groupby(compact_df['manufacturer']).count()

display( result.sort_values(ascending=False) )
```

- ```
  manufacturer
  audi          15
  volkswagen    14
  toyota        12
  subaru         4
  nissan         2
  Name: class, dtype: int64
  ```



### 한국복지패널 data

#### 데이터 로드

```python
import numpy as np
import pandas as pd
import savReaderWriter

with savReaderWriter.SavReader('D:\python_ML\Koweps/Koweps_hpc10_2015_beta1.sav', ioUtf8 = True) as reader:
    df = pd.DataFrame(reader.all(), columns = [s for s in reader.header])

# print(df.shape)      # # (16664, 957)
# df['h10_g3'].head()  # 성별
# df['h10_g4'].head()  # 태어난 연도

df = df[['h10_g3', 'h10_g4', 'h10_g10', 'h10_g11', 'h10_eco9', 'p1002_8aq1', 'h10_reg7']]
display(df)

# 변수명	설명	내용
# h10_g3	성별	1.남         2.여
# h10_g4	태어난 연도	년
# h10_g10	혼인상태	"0.비해당(18세 미만)
#                        1.유배우         2.사별         3.이혼          4.별거          
#                        5.미혼(18세이상, 미혼모 포함)   6.기타(사망 등)"
# h10_g11	종교	1.있음                2.없음
# h10_eco9	직종	직종 코드표 참조
# p1002_8aq1	일한달의 월 평균 임금	만원
# h10_reg7	7개 권역별 지역구분	"1. 서울          2. 수도권(인천/경기)    3. 부산/경남/울산   4.대구/경북   
#                                5. 대전/충남   6. 강원/충북               7.광주/전남/전북/제주도"
```

- |       | h10_g3 | h10_g4 | h10_g10 | h10_g11 | h10_eco9 | p1002_8aq1 | h10_reg7 |
  | ----: | -----: | -----: | ------: | ------: | -------: | ---------: | -------: |
  |     0 |    2.0 | 1936.0 |     2.0 |     2.0 |      NaN |        NaN |      1.0 |
  |     1 |    2.0 | 1945.0 |     2.0 |     2.0 |      NaN |        NaN |      1.0 |
  |     2 |    1.0 | 1948.0 |     2.0 |     2.0 |    942.0 |      120.0 |      1.0 |
  |     3 |    1.0 | 1942.0 |     3.0 |     1.0 |    762.0 |      200.0 |      1.0 |
  |     4 |    2.0 | 1923.0 |     2.0 |     1.0 |      NaN |        NaN |      1.0 |
  |   ... |    ... |    ... |     ... |     ... |      ... |        ... |      ... |
  | 16659 |    2.0 | 1967.0 |     1.0 |     1.0 |      NaN |        NaN |      5.0 |
  | 16660 |    2.0 | 1992.0 |     5.0 |     1.0 |    314.0 |      302.5 |      5.0 |
  | 16661 |    1.0 | 1995.0 |     5.0 |     1.0 |      NaN |        NaN |      5.0 |
  | 16662 |    2.0 | 1998.0 |     0.0 |     1.0 |      NaN |        NaN |      5.0 |
  | 16663 |    1.0 | 2001.0 |     0.0 |     1.0 |      NaN |        NaN |      5.0 |

  16664 rows × 7 columns



#### 1.

- 성별에 따른 월급 차이

```python
man = df.loc[df['h10_g3'] == 1,:]['p1002_8aq1'].mean()
woman = df.loc[df['h10_g3'] == 2,:]['p1002_8aq1'].mean()

print( f"남성 평균 월급 : {man}" )
print( f"여성 평균 월급 : {woman}" )
print( f"남성이 여성보다 월급이 약 {int(man - woman)}만원 많다.")
```

- ```
  남성 평균 월급 : 312.29316476345844
  여성 평균 월급 : 162.19967002749772
  남성이 여성보다 월급이 약 150만원 많다.
  ```



#### 2.

- 나이와 월급의 관계

```python
# 나이를 구하기 위해 datetime 모듈 호출
from datetime import datetime
# 한국 나이를 구하기 위해 현재년도에 1을 더한 변수 생성
temp_korean_age = datetime.today().year + 1

# temp_korean_age에서 출생년도를 뺀 age 컬럼 추가
df['age'] = temp_korean_age - df['h10_g4']

# 임금 평균을 연령대별로 그룹화한 뒤 max값 추출
result = df['p1002_8aq1'].groupby(df['age']).mean()
result = result.sort_values(ascending=False).head(1)

# Series의 값을 깔끔하게 출력하기 위해 list로 생성
r_list = []
for i in result.index:
    r_list.append(i)
for i in result.values:
    r_list.append(i)

print(f"월급을 가장 많이 받는 나이는 {int(r_list[0])}살, 월급 : {r_list[1]}")
```

- ```
  월급을 가장 많이 받는 나이는 59살, 월급 : 318.67774774774773
  ```



#### 3.

- 연령대에 따른 월급 차이

```python
# 연령대별 임금 평균 추출
young = df.loc[df['age'] < 30,:]['p1002_8aq1'].mean()
middle = df.loc[(df['age'] >= 30) & (df['age'] < 60),:]['p1002_8aq1'].mean()
old = df.loc[df['age'] >= 60,:]['p1002_8aq1'].mean()

print(f"초년(young) : {young}\n중년(middle) : {middle}\n노년(old) : {old}")
```

- ```
  초년(young) : 125.06214285714285
  중년(middle) : 269.0703669154229
  노년(old) : 182.77886056971514
  ```



#### 4.

- 연령대 및 성별 월급 차이

```python
# 연령대별 성별 임금 평균 추출
young_m = df.loc[(df['h10_g3'] == 1)&(df['age'] < 30),:]['p1002_8aq1'].mean()
middle_m = df.loc[(df['h10_g3'] == 1)&(df['age'] >= 30)&(df['age'] < 60),:]['p1002_8aq1'].mean()
old_m = df.loc[(df['h10_g3'] == 1)&(df['age'] >= 60),:]['p1002_8aq1'].mean()

young_w = df.loc[(df['h10_g3'] == 2)&(df['age'] < 30),:]['p1002_8aq1'].mean()
middle_w = df.loc[(df['h10_g3'] == 2)&(df['age'] >= 30)&(df['age'] < 60),:]['p1002_8aq1'].mean()
old_w = df.loc[(df['h10_g3'] == 2)&(df['age'] >= 60),:]['p1002_8aq1'].mean()

print(f"초년 남성 : {young_m}\n초년 여성 : {young_w}\n중년 남성 : {middle_m}\n중년 여성 : {middle_w}\n노년 남성 : {old_m}\n노년 여성 : {old_w}")
```

- ```
  초년 남성 : 116.80851851851851
  초년 여성 : 128.9717543859649
  중년 남성 : 335.6036725412166
  중년 여성 : 188.74635552505148
  노년 남성 : 258.6518768768769
  노년 여성 : 107.13300898203593
  ```



#### 5.

- 직업별 월급 차이

```python
# 직업명이 들어있는 엑셀파일을 로드
work_xlsx = pd.read_excel('D:\python_ML\Koweps/Koweps_Codebook.xlsx', sheet_name='직종 코드')

# df를 직업별로 그룹화한 후 임금 데이터의 평균이 들어있는 work_df 생성
work_df = df['p1002_8aq1'].groupby(df['h10_eco9']).mean()

# work_df와 work_xlsx를 직업 코드로 merge
result = pd.merge(work_xlsx, work_df,
                  left_on = 'code_job',
                  right_on = 'h10_eco9',
                  how = 'inner')

# max 데이터 하나만 추출
display(result[['job','p1002_8aq1']].sort_values(by='p1002_8aq1', ascending=False).head(1))
```

- |      |                             job | p1002_8aq1 |
  | ---: | ------------------------------: | ---------: |
  |   23 | 금속 재료 공학 기술자 및 시험원 | 845.066667 |



#### 6.

- 성별 직업 빈도

```python
# 직업명이 들어있는 엑셀파일과 df를 직업코드로 merge한 work_df 생성
work_df = pd.merge(work_xlsx, df,
                  left_on = 'code_job',
                  right_on = 'h10_eco9',
                  how = 'inner')

# Series의 index값과 value값을 깔끔하게 추출하기 위한 list 생성
m_list=[]
w_list=[]

# 성별이 남성인 사람들을 직업명으로 그룹핑한 후 max값 하나만 Series로 추출 
man = work_df.loc[work_df['h10_g3'] == 1,:].groupby(work_df['job'])
man = man.size().sort_values(ascending=False).head(1)
for i in man.index:
    m_list.append(i)
for i in man.values:
    m_list.append(i)

# 성별이 여성인 사람들을 직업명으로 그룹핑한 후 max값 하나만 Series로 추출 
woman = work_df.loc[work_df['h10_g3'] == 2,:].groupby(work_df['job'])
woman = woman.size().sort_values(ascending=False).head(1)
for i in woman.index:
    w_list.append(i)
for i in woman.values:
    w_list.append(i)

print(f"남성 - 직업 : {m_list[0]}, 종사자 수 : {m_list[1]}\n여성 - 직업 : {w_list[0]}, 종사자 수 : {w_list[1]}")
```

- ```
  남성 - 직업 : 작물재배 종사자, 종사자 수 : 640
  여성 - 직업 : 작물재배 종사자, 종사자 수 : 680
  ```



#### 7.

- 종교 유무에 따른 이혼율

```python
# h10_g10 = 이혼 3
# h10_g11 = 종교 있음 1 종교 없음 2

# 종교 유무에 따른 인구수를 나타내는 religion 변수 생성
religion_y = df.loc[df['h10_g11'] == 1,:]['h10_g3'].count() 
religion_n = df.loc[df['h10_g11'] == 2,:]['h10_g3'].count() 

# 종교 유무에 따른 이혼 인구수를 나타내는 divorce_r 변수 생성
divorce_r_y = df.loc[(df['h10_g10'] == 3)&(df['h10_g11'] == 1),:]['h10_g3'].count() 
divorce_r_n = df.loc[(df['h10_g10'] == 3)&(df['h10_g11'] == 2),:]['h10_g3'].count() 

# (종교 유무 이혼 인구수 / 종교 유무 인구수)를 통해 종교 유무 이혼율 출력
print(f"종교 유 이혼율 : {divorce_r_y/religion_y}\n종교 무 이혼율 : {divorce_r_n/religion_n}")
```

- ```
  종교 유 이혼율 : 0.04076053187523301
  종교 무 이혼율 : 0.04456307299524196
  ```



#### 8.

- 지역별 연령대, 노년층이 많은 지역

```python
# h10_reg7 = 1. 서울      2. 수도권(인천/경기)  3. 부산/경남/울산      
#            4.대구/경북  5. 대전/충남          6. 강원/충북    
#            7.광주/전남/전북/제주도
# 노년층 = 60세 이상

# 지역명을 매핑하기 위한 DataFrame(reg_df) 생성
reg_dict = {
    1 : '서울',
    2 : '수도권(인천/경기)',
    3 : '부산/경남/울산',
    4 : '대구/경북',
    5 : '대전/충남',
    6 : '강원/충북',
    7 : '광주/전남/전북/제주도'
}
reg_df = pd.DataFrame(list(reg_dict.items()),
                      columns=['reg_num','name'])

# df와 지역명이 들어있는 reg_df 결합
reg_merge_df = pd.merge(reg_df, df,
                        left_on = 'reg_num',
                        right_on = 'h10_reg7',
                        how = 'inner')

# 60세 이상 인구를 지역명으로 그룹핑한 뒤 가장 많은 지역 1곳 추출
result = reg_merge_df.loc[df['age'] >= 60].groupby(reg_merge_df['name']).size().sort_values(ascending=False).head(1)

# Series의 index값과 value값을 깔끔하게 보이기 위해 list로 추출
r_list = []
for i in result.index:
    r_list.append(i)
for i in result.values:
    r_list.append(i)

print(f"노년층이 가장 많은 지역 : {r_list[0]}, 노년층 인구 수 : {r_list[1]}")
```

- ```
  노년층이 가장 많은 지역 : 광주/전남/전북/제주도, 노년층 인구 수 : 1640
  ```