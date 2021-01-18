# day6_Python_TIL

## Numpy

### numpy 개념

> numerical Python

1. vector와 matrix 연산에 강점

2. ndarray => n 차원의 배열

3. list와 유사, ndarray의 각 요소는 모두 같은 데이터 타입

4. 함수

   1. numpy.array([1,2,3,4])

   2. numpy.arange()

   3. random 기반의 생성함수

      zeros, ones, empty, full

   4. shape(), reshape() => view
   5. resize()



### indexing, slicing

- 기본 indexing, boolean indexing, fancy indexing



#### enumerate

> 반복문에서 데이터와 index의 값을 같이 가져옴

```python
import numpy as np

arr = np.arange(10,20,1)

# enumerate: 반복문에서 데이터와 index값을 같이 가져옴
for i,t in enumerate(arr):
    print(f"인덱스 : {i}, 값 : {t}")
```



#### slicing

> ndarray의 slicing도 python의 list와 동일한 형태로 사용 가능

```python
arr = np.arange(0, 5, 1)
print(arr)
print(arr[0:2])
print(arr[0:-1]) # 맨 마지막 요소만 제외하고 슬라이싱
print(arr[1:4:2]) # 2칸씩 이동하며 슬라이싱

# 2차원 형태의 ndarray를 slicing
arr = np.array([[1,2,3,4],
                [5,6,7,8],
                [9,10,11,12],
                [13,14,15,16]])
print(arr)
print(arr[1,2]) # => 7, 1행 2열, 일반적은 2차원의 indexcing
print(arr[1,:]) # [5 6 7 8]
print(arr[1:3]) # [[ 5  6  7  8]
                #  [ 9 10 11 12]]
print(arr[0])   # [1 2 3 4], 값이 하나일 땐 행을 지칭
```




#### boolean indexing

- boolean mask: 원본 ndarray와 shape이 같고 그 요소 값이 모두 boolean(True, False)로 구성된 ndarray
- boolean mask를 이용해서 indexing하는 방식을 boolean indexing

```python
np.random.seed(1)
arr = np.random.randint(1,20,(10,))
print(arr)

# boolean mask
# [ 6 12 13  9 10 12  6 16  1 17]
# [True, True, False, False, ... , False]
# 결국 우리가 구한 boolean mask => arr % 2 == 0
print(arr % 2 == 0)
print(arr[arr % 2 == 0]) # boolean indexing
print(arr[arr ** 2 == 36]) # [6 6]


for temp in arr:
    if temp % 2 == 0:
        print(temp, end = ' ') # 6 12 10 12 6 16 
```



#### fancy indexing

- ndarray에 index배열(list형식)을 전달하여 배열요소를 참조하는 방식
- 추출할 값들의 인덱스 리스트를 indexing에 전달

```python
arr = np.array([1,2,3,4,5])
print(arr)
print(arr[[1,3,4]]) # [2 4 5]
                    # 추출할 값들의 인덱스 리스트를 indexing에 전달

# Fancy indexing
import numpy as np

# resize는 잘못됐는지 파악할 수 없다. reshape 이용 권장
arr = np.arange(0,12,1).reshape(3,4).copy()
print(arr)
'''
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
 '''

print(arr[2,2])       # 10, indexing
print(arr[1:2,2])     # [6], slicing, 원본과 결과본이 같다
print(arr[1:2,1:2])   # [[5]]

print(arr[[0,2],2])   # [2 10] fancy indexing
print(arr[[0,2],2:3]) # [[ 2]
                      #  [10]]
```



- np.ix_():

  다차원에서 열과 행 모두 fancy indexing은 불가능

  np.ix_()를 이용하면 가능

```python
print(arr[np.ix_([0,2],[0,2])]) 
'''
[[ 0  2]
 [ 8 10]]
'''
```



### ndarray의 연산

- 4칙연산은 python의 list와 동일
- 그러나 shape이 다르면 연산이 수행되지 않는다.
- 행렬곱 연산
  - 앞쪽 행렬의 열과 뒤쪽 행렬의 행의 수가 일치해야 함
  - 행렬곱 (2,3)과 (3,2) ==> (2,2)로 결과가 나옴
  - 행렬곱 연산 함수: np.matmul(), dot()

```python
import numpy as np

arr1 = np.array([[1,2,3],
                 [4,5,6]]) # 2 x 3 ndarray
arr2 = np.arange(10,16,1).reshape(2,3).copy() # 2 x 3 ndarray
arr3 = np.arange(10,16,1).reshape(3,2).copy() # 3 x 2 ndarray

# ndarray 연산의 기본은 같은 위치에 있는 원소 간의 연산
# shape이 다르면 연산이 수행되지 않음
print(arr1 + arr2) # 4칙연산 모두 같아요
'''
[[11 13 15]
 [17 19 21]]
'''

# 앞쪽 행렬의 열과 뒤쪽 행렬의 행의 수가 일치해야함
# 행렬곱 2x3과 3x2 ==> 2x2로 결과가 나옴 
# 행렬곱 연산 함수 : np.matmul(), dot() 
print(np.matmul(arr1, arr3))
print(np.dot(arr1, arr3))
'''
[[ 76  82]
 [184 199]]
[[ 76  82]
 [184 199]]
'''
```



#### broadcasting

> 다른 shape을 가지는 ndarray의 4칙연산을 가능하게 하는 기능

- 열의 수가 다른 경우엔 broadcasting 불가
- 행렬곱에서는 broadcasting 적용 안됨, 4칙연산에서만 적용
- 요소가 하나인 scala의 경우엔 broadcasting 가능

```python
# 같은 shqpe을 가지는 ndarray끼리만 4칙연산이 가능한가요?
# 다른 shape을 가지는 ndarray의 4칙연산을 가능하게 하는 broadcasting 지원
import numpy as np

arr1 = np.array([[1,2,3],
                 [4,5,6]])  # (2,3) ndarray, 2d-matrix
arr2 = np.array([7,8,9])    # (3,)  ndarray, 1d-vector
# 열의 수가 다른 경우엔 broadcasting 불가
# 행렬곱에서는 broadcasting 적용 안됨 - 4칙연산에서만 적용

print(arr1)
print(arr2)
print(arr1 + arr2)
print(arr1 + 10)            # 요소가 하나인 scala의 경우엔 broadcasting 가능
```



### ndarray의 차원

- 0차원: 하나의 숫자 	 => scalar	(ex. 5)
- 1차원: 열로 구성          => vector   (ex. [1 3 5 7 9])
- 2차원: 행과 열로 구성 => matrix   (ex. [[1 2 3]

​																          [4 5 6]])

- 3차원: array라고 표현



#### 전치행렬

> transpose

- 원본행렬의 행은 열로, 열은 행으로 바꾼 matrix를 의미
- 이렇게 만든 전치행렬(transpose)는 view이다.
- 수학적으로 전치행렬을 표현할 때는 위첨자 T를 사용
- 1차원 vector에서는 전치행렬을 구할 수 없음

```python
import numpy as np

arr = np.array([[1,2,3],
                [4,5,6]])

print(arr.T) # 전치행렬을 구할때 T 속성을 이용
'''
[[1 4]
 [2 5]
 [3 6]]
'''

# 1차원 vector에 대해 전치행렬을 구하면 어떻게 되나요?
# 1차원 vector에 대해서는 전치행렬을 구할 수 없다.
```



#### iterator

> 반복자, 지시자, 포인터, 가리키는 용도

- 반복문은 일반적으로 for문과 while문 사용

  반복하는 횟수를 알고 있을 때는 for

  조건에 따라서 반복할 때는 while

  numpy ndarray를 반복처리할 때는 일반적으로 for

- iterator는 while문과 함께 이용해서 반복처리

  - 표현식:

  - it = np.nditer(arr, flags=[' '])

    while not it.finished:

    ​	idx = it.index

    ​	print(arr[idx], end=' ')

    ​	it.iternext

  - it.iternext: 가리키는 화살표를 한 칸 옮기는 작업

- 1차원:
  - flags=['c_index']: c언어의 기본 인덱스 방식을 이용하라

```python
import numpy as np

# 1차원 numpy array(ndarray)에 대한 요소를 출력
arr = np.array([1,2,3,4,5])

# for문을 이용한 반복처리
for i in arr:
    print(i, end=' ')
    
print()
    
# iterator: 지시자, 포인터, 가리키는 용도
# while문과 iterator를 이용해서 반복처리
arr = np.array([1,2,3,4,5])
# flags=['c_index']: 인덱스 값을 사용할 때 c언어의 인덱스 방식을 이용하라
# 다만 1차원에만 해당


it = np.nditer(arr, flags=['c_index']) 
while not it.finished:
    idx = it.index   # 0
    print(arr[idx], end=' ')
    it.iternext()    # 화살표를 옮기는 작업

# 동일한 결과
# 1 2 3 4 5
# 1 2 3 4 5
```



- 2차원:
  - for문은 차원이 늘어날수록 for문의 개수가 늘어나야함
  - iterator는 multi_index를 사용해 쉽게 출력

```python
# 2차원 matrix에 대해서 각 요소를 출력하고 싶다
arr = np.array([[1,2,3],
                [4,5,6]])

# for문은 차원이 늘어날수록 for문 개수가 늘어나야함
for tmp1 in range(arr.shape[0]):      # arr.shape=> (2,3)
    for tmp2 in range(arr.shape[1]):
        print(arr[tmp1,tmp2], end=' ')

        
print()


# iterator를 가지고 반복처리
arr = np.array([[1,2,3],
                [4,5,6]])
# 2차원 이상부터는 flags=['multi_index'] 사용
it = np.nditer(arr, flags=['multi_index'])

while not it.finished:
    
    idx = it.multi_index  # (0,0): fancy indexing의 형태
    it.iternext()
    print(arr[idx], end=' ')
 
# 동일한 결과
# 1 2 3 4 5 6 
# 1 2 3 4 5 6 
```



### numpy의 비교연산

> 비교연산도 사칙연산과 유사하게 동작

- np.array_equal(arr1, arr2):

  전체 내용이 동일한 지 확인할 때는 array_equal 함수 이용

```python
import numpy as np

np.random.seed(4)
arr1 = np.random.randint(0,10,(2,3))
arr2 = np.random.randint(0,10,(2,3))

print(arr1)
'''
[[7 5 1]
 [8 7 8]]
'''

print(arr2)
'''
[[2 9 7]
 [7 7 9]]
'''

print(arr1 == arr2)  # boolean mask
'''
[[False False False]
 [False  True False]]
'''

print(arr1 > arr2)   # boolean mask
'''
[[ True False False]
 [ True False False]]
'''

arr1 = np.arange(10)
arr2 = np.arange(10)
print(np.array_equal(arr1,arr2))  # 전체 내용이 같은지 equal 함수 이용
```



### 집계함수, 수학함수

- 집계함수: 합, 평균, 표준편차, 분산
- 수학함수: 최대값, 최소값, 제곱근, 제곱값, 로그값

```python
import numpy as np

arr = np.arange(1,7,1).reshape(2,3).copy()

print(arr)
'''
[[1 2 3]
 [4 5 6]]
'''

print(np.sum(arr))    # 21,   합계
print(arr.sum())      # 21
print(np.mean(arr))   # 3.5.  평균
print(arr.mean())     # 3.5
print(arr.max())      # 6,    최대값
print(arr.min())      # 1,    최소값
print(np.argmax(arr)) # 5,    최대값의 index 리턴
# argmax: tuple로 리턴하는 게 아닌 단순 순서
print(arr.argmin())   # 0,    최소값의 index 리턴
print(np.std(arr))    # 1.70, 표준편차
print(np.sqrt(arr))   # 제곱근: 각 요소들의 제곱근을 모두 리턴
'''
[[1.         1.41421356 1.73205081]
 [2.         2.23606798 2.44948974]]
'''
```



### axis

> 축, 집계함수나 수학함수 연산의 기준

- numpy의 모든 집계함수는 기본적으로 axis를 기준으로 계산
- axis를 지정하지 않으면 axis가 None으로 설정되고 대상 범위가 배열의 전체로 지정됨
- axis는 축을 의미하며, 숫자로 표현됨

- 행 방향: 세로, 열 방향: 가로
- 'axis='을 적지 않아도 default값이 axis이기 때문에 정상적으로 작동함

1. 1차원: 0 => 열 방향(가로)
2. 2차원: 0 => 행 방향(세로), 1 => 열 방향(가로)
3. 3차원:  0 => 면. 1 => 행 방향(세로), 2 => 열 방향(가로)

- axis = -1 은 항상 열 방향을 지칭



- 1차원:

```python
'''
# 1차원: 0 => 열 방향(가로)
# 2차원: 0 => 행 방향(세로), 1 => 열 방향(가로)
# 3차원: 0 => 면, 1 => 행 방향(세로), 2 => 열 방향(가로)
'''

arr = np.array([1,2,3,4,5])      # 1차원의 ndarray
# axis는 축을 의미, 축은 숫자로 표현됨
# 1차원에서 axis값은 0밖에 쓸 수 없다.
# 1차원: 0 => 열 방향(가로)
# 0이 의미하는 것은 열의 방향(가로)
# 열 방향: 가로, 행 방향: 세로
# 행들은 밑으로 떨어지고, 열들은 가로로 펼쳐짐

# 1차원 vector는 axis가 큰 의미가 없다
print(arr.sum(axis=0)) # 15
```



- 2차원:

```python
arr = np.array([[1,2,3],
                [4,5,6],
                [7,8,9],
                [10,11,12]])

print(arr.shape)    # (4, 3)
print(arr.sum())    # 78, 축이 명시되지 않으면 sum()의 대상은 전체 ndarray

# axis는 0과 1을 사용할 수 있음
# 2차원: 0 => 행 방향(세로), 1 => 열 방향(가로)
print(arr.sum(axis=0)) # [22 26 30]
print(arr.sum(axis=1)) # [ 6 15 24 33]
# 'axis=' 를 적지 않아도 default값이 axis이기 때문에 정상적으로 작동
print(arr.sum(0)) # [22 26 30]
print(arr.sum(1)) # [ 6 15 24 33]
```



- argmax() 함수

  최대값의 인덱스 자리를 리턴

```python
np.random.seed(1)
arr = np.random.randint(0,11,(4,3))
print(arr)
'''
[[5 8 9]
 [5 0 0]
 [1 7 6]
 [9 2 4]]
'''

print(arr.argmax(axis=0)) # [3 0 0], 행 방향(세로)
print(arr.argmax(axis=1)) # [2 0 1 0], 열 방향(가로)

```



- ndarray arr 안에 10보다 큰 수가 몇개 있는지 알아보려면?

  내 풀이:

```python
arr = np.array([[1,2,3,4],
                [5,6,7,8],
                [9,10,11,12],
                [13,14,15,16]])

count = 0
bool_arr = arr > 10

it = np.nditer(bool_arr, flags=['multi_index'])

while not it.finished:
    idx = it.multi_index
    it.iternext()
    if bool_arr[idx]:
        count += 1
        
print(count) # 6
```

​		모범 답안:

```python
arr = np.array([[1,2,3,4],
                [5,6,7,8],
                [9,10,11,12],
                [13,14,15,16]])

print((arr > 10).sum())  # 6     

# 내부적으로 broadcasting이 일어남
# 파이썬에서 false = 0, True = 1
# boolean 값을 단순 sum하면 True의 개수    
```



### 정렬

> 정렬은 상당히 많이 사용하는 기능

- np.sort(): 인자로 들어가는 원본 ndarray는 변화가 없고 정렬된 복사본을 리턴
- arr.sort(): 원본 배열을 정렬하고 리턴이 없음
- np.sort()[: : -1]: 3번째 인자에 -1을 넣으면 역순으로 정렬

```python
import numpy as np

np.random.seed(1)
arr = np.arange(10)
print(arr)                 # [0 1 2 3 4 5 6 7 8 9]

np.random.shuffle(arr)     # 정렬 함수를 사용하기 위해 ndarray를 섞음
print(arr)                 # [2 9 6 4 0 3 1 7 8 5]

print(np.sort(arr))        # [0 1 2 3 4 5 6 7 8 9], 정렬된 형태 출력

print(arr)                 # [2 9 6 4 0 3 1 7 8 5], 원본엔 변화가 없음

print(arr.sort())          # None, 리턴 값이 없어서 None 출력

print(arr)                 # [0 1 2 3 4 5 6 7 8 9], 원본이 변경되어 있음


# 특수한 슬라이싱을 이용하면 역순으로 정렬할 수 있다.
print(np.sort(arr)[::-1])  # [9 8 7 6 5 4 3 2 1 0]
                           # 3번째 인자에 -1을 넣어 역순으로 정렬


# 2차원 정렬은 pandas를 이용
```



### np.concatenate

> numpy array를 연결하는 np.concatenate()

- 합치고자 하는 ndarray의 shape이 동일해야 함
- reshape을 이용해서 shape을 맞춘 후 합침

```python
import numpy as np

arr = np.array([[1,2,3],
                [4,5,6]])     # (2,3)

# 행 방향 concat(세로)
new_row = np.array([7,8,9])   # (3,)
result = np.concatenate((arr, new_row.reshape(1,3)), axis=0)
# reshape을 이용해서 new_row를 2차원으로 변경
print(result)
'''
[[1 2 3]
 [4 5 6]
 [7 8 9]]
'''

# 열 방향 concat(가로)
new_col = np.array([7,8,9,10])
result = np.concatenate((arr, new_col.reshape(2,2)), axis=1)
print(result)
'''
[[ 1  2  3  7  8]
 [ 4  5  6  9 10]]
'''
```



### delete

> numpy의 delete()

- delete()도 axis를 기준으로 삭제가 진행
- 만약 axis를 명시하지 않으면 자동으로 1차 배열로 변환이 된 후 삭제

```python
arr = np.array([[1,2,3],
                [4,5,6]])     # (2,3)

result = np.delete(arr, 1)     # 1차배열로 변환한 후 삭제
                               # [1 2 3 4 5 6] => 이렇게 1차배열로 변환 후
                               # 1번째 인덱스 삭제
print(result)                  # [1 3 4 5 6]


result = np.delete(arr, 1, axis=0)  # 행
print(result)                       # [[1 2 3]]

result = np.delete(arr, 1, axis=1)  # 열
print(result)                       
'''
[[1 3]
 [4 6]]
'''

result = np.delete(arr, 1, axis=-1)  # 열, axis=-1 값은 항상 열 방향을 지칭
print(result)                       
'''
[[1 3]
 [4 6]]
'''

# numpy는 pandas의 기본 자료구조이며 머신러닝에서도 기본 자료구조

# pandas는 파일 관리
```

