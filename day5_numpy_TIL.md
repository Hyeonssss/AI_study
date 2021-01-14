# day5_Python_TIL

## python 기본

### error 처리

- try : 실행해보고
- except : 예외가 있다면 실행
- else : 없으면 실행
- finally : 무조건 실행

```python
def my_func(list_data):
    
    my_sum = 0
    try:
        my_sum = list_data[0] + list_data[1] + list_data[2]
        
    except Exception as err:
        print('실행시 문제가 발생했어요')
        my_sum = 0
        
    else:
        print('실행시 문제가 없어요')
        
    finally:
        print('무조건 실행돼요')
    
    return my_sum


my_list = [1, 2, 3]
print(my_func(my_list))

'''
=>
실행시 문제가 없어요
무조건 실행돼요
6
'''
```



### file 처리

```python
# 기본적인 file 처리에 대해 알아보자
# file 처리는 대부분 pandas 이용

my_file = open('mpg.txt', 'r')

# 파일 안에 있는 모든 내용을 출력할 것
# '' 공백 문자열은 False로 간주

while True:
    line = my_file.readline()
    print(line)
    if not line:
        break
        
my_file.close()  # 반드시 사용한 resouce는 해제
```



## numpy

### numpy 개념

> Numerical python

- vector, matrix 연산에 편리성 제공
- ndarray(n-dimention array) 제공



### ndarray

- import numpy as np : numpy module 호출

- 공백으로 데이터 구분
- 모든 원소가 같은 데이터 타입 가져야 함
- 다 차원으로 구현 가능
- dtype = np.float64 : 데이터 타입 지정 가능

```python
import numpy as np

# python의 리스트
a = [1, 2, 3, 4]  # literal(프로그램적 기호를 이용해서 표현)
# a = list()        # list라는 class를 이용해서 생성
print(type(a))    # <calss 'list'>
print(a)          # [1, 2, 3, 4], ','로 데이터 구분

# numpy의 ndarray
b = np.array([1, 2, 3, 4])
print(b)          # [1 2 3 4], ' ' 공백으로 데이터 구분
print(type(b))    # <class 'numpy.ndarray'>
print(b.dtype)    # int32
# ndarray는 모든 원소가 같은 데이터 타입을 가져야 한다
# list는 모든 원소가 같은 데이터 타입을 가지지 않아도 됨

list_a = [1, 3.14, True, 'Hello']
np_b = np.array([1, 3.14, True, 'Hello'])
print(np_b)       # 문자열로 모든 데이터 타입이 변경됨
print(np_b.dtype) # <U32

b = np.array([1, 2, 3, 4])

print(b[0])
print(type(b[0])) # <class 'numpy.int32'>

# 다차원 ndarray
my_list = [[1,2,3], [4,5,6]]  # 차원 개념이 없이 중첩된 형태
print(my_list)    # [[1, 2, 3], [4, 5, 6]]

                                        # 데이터 타입 지정하기
my_array = np.array([[1,2,3], [4,5,6]], dtype=np.float64)
print(my_array)   # [[1 2 3]
                  #  [4 5 6]]

print(my_array[1, 1])  # 5
```



#### shape

> ndarray의 대표적인 속성

- 차원과 요소의 개수를 tuple로 표현
- ndarray의 차원과 요소의 개수를 볼 수 있는 ndarray 속성
- ndim : 차원의 수를 나타내는 ndarray 속성

```python
import numpy as np
# 1차원
my_list = [1, 2, 3, 4]
arr = np.array(my_list)

print(arr.ndim)  # ndim이라는 속성은 차원의 수를 나타냄 => 1
print(arr.shape) # 차원과 요소의 개수를 tuple로 표현 => (4,)

# 2차원
my_list = [[1,2,3], [4,5,6]]
arr = np.array(my_list)
print(arr.ndim)  # 2
print(arr.shape) # (2, 3) => 2는 행의 개수(차원) 3은 열의 개수(요소)
                            # 열의 개수(요소)는 다 같아야 함

# 3차원
my_list = [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]
arr = np.array(my_list)
print(arr)
print(arr.ndim)  # 3
print(arr.shape) # (2, 2, 3) => 2면 2행 3열
```

​	

- 직접적으로 shape 변경이 가능하지만 권장되진 않음
- reshape() method 사용하여 변경

```python
import numpy as np

my_list = [1,2,3,4]
arr = np.array(my_list)
print(arr)
print(arr.shape) # => (4,)
arr.shape = (2, 1, 2)  

print(arr)
'''
=>
[[[1 2 ]]

 [[3 4]]]
 '''
```



#### size

- 차원에 상관 없이 모든 요소의 개수를 출력하는 ndarray 속성

```python
import numpy as np

print('---1차원---')
my_list = [1, 2, 3, 4]
arr = np.array(my_list)
print(arr)
print(f'shape : {arr.shape}') # (4,)
print(f'크기(len) : {len(arr)}') # 4
print(f'크기(size) : {arr.size}') # 4

print('---2차원---')
my_list = [[1,2,3], [4,5,6]]
arr = np.array(my_list)
print(arr)
print(f'shape : {arr.shape}') # (2, 3)
print(f'크기(len) : {len(arr)}') # 2
print(f'크기(size) : {arr.size}') # 6
```



#### astype

- ndarray의 data type을 변경

```python
# astype() ndarray의 data type을 변경
import numpy as np

arr = np.array([1.2, 2.3, 3.5, 4.1, 5.7])
print(arr) # => [1.2 2.3 3.5 4.1 5.7]

arr = arr.astype(np.int32)  # 소수점 이하는 버림(반올림 x)
print(arr) # => [1 2 3 4 5]

arr = arr.astype(np.float64)
print(arr) # => [1. 2. 3. 4. 5.]
```



### ndarray 생성

#### zeros, ones, empty, full

- 특정 shape을 지정해서 ndarray를 만들며 내용을 특정 값으로 채움
- 기본 데이터 타입은 np.float64
- 표현 : np.zeros((shape(tuple로 표현)))
  - zeros = 내용을 0으로
  - ones = 내용을 1로
  - empty = 내용을 랜덤한 값으로
  - full = 특정 값 지정

```python
# ndarray의 다양한 생성 함수(zeros, ones, empty, full)
import numpy as np

# zeros
arr = np.zeros((3, 4)) # 특정 형태의 ndarray를 만들어서 내용을 0으로 채움
print(arr)             # 기본 데이터 타입은 np.float64
'''=>
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
 '''

arr = np.zeros((3, 4), dtype=np.int32) 
print(arr)             

# ones
arr = np.ones((3, 4)) 
print(arr)             
'''=>
[[1. 1. 1. 1.]
 [1. 1. 1. 1.]
 [1. 1. 1. 1.]]
 '''

# empty
arr = np.empty((3, 4)) 
print(arr)            
'''=> 
[[6.23042070e-307 4.67296746e-307 1.69121096e-306 9.34613185e-307]
 [1.69121231e-306 7.56587585e-307 1.37961302e-306 1.05699242e-307]
 [1.95821439e-306 7.56593017e-307 9.34595530e-307 1.24610383e-306]]
 '''

arr = np.full((3, 4), 7, dtype=np.float64) 
print(arr)
'''=>
[[7. 7. 7. 7.]
 [7. 7. 7. 7.]
 [7. 7. 7. 7.]]
 '''
```



#### arange

- ndarray를 생성할 범위 지정
- range와 비슷

```python
import numpy as np

# python range()
arr1 = range(0, 10, 2)  # 의미만 가지고 있음
print(arr1)

# numpy arange()
arr = np.arange(0, 10, 1)  # 실제로 생성이 된다
print(arr.reshape((2,5)))
```



#### ndarray 내부에 list 삽입

```python
arr = np.array([1,2,3])
```



#### random 기반의 생성

> normal, rand, randn, randint, random

1. np.random.nomal()

   정규분포에서 실수 형태의 난수를 추출

   표현식 : np.random.normal(평균, 표준편차, shape)

2. np.random.rand()

   균등분포에서 0이상 1미만의 실수를 난수로 추출

   표현식 : np.random.rnad(d0, d1, d2, d3)

3. np.random.randn()

   표준정규분포에서 실수형태로 난수를 추출

   표현식 : np.random.randn(개수)

4. np.random.randint()

   균등분포에서 정수 표본 추출

   표현식 : np.random.randint(low, high, shape)

5. np.random.random()

   균등분포에서 0이상 1미만의 실수를 난수로 추출

   표현식 : np.random.random(shape)



- matplotlib.pyplot as plt
  - 간단한 그래프를 이용해서 데이터의 분포를 보는 모듈

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. np.random.normal() : 정규분포에서 실수 형태의 난수를 추출
my_mean = 50  # 평균
my_std = 2    # 표준편차
arr = np.random.normal(my_mean, my_std, (10000,))
print(arr)
plt.hist(arr, bins=100)  # bins : 영역
plt.show()

# 2. np.random.rand() : 0이상 1미만의 실수를 난수로 추출
#                       균등 분포로 난수를 추출
#     np.random.rand(d0, d1, d2, d3, ...)
arr = np.random.rand(10000)
print(arr)
plt.hist(arr, bins=100)  # bins : 영역
plt.show()

# 가장 많이 사용됨
# 3. np.random.randn() : 표준 정규분포에서 실수형태로 난수를 추출
arr = np.random.randn(10000)
print(arr)
plt.hist(arr, bins=100)  # bins : 영역
plt.show()

# 가장 많이 사용됨
# 4. np.random.randint(low, high, shape) : 균등분포로 정수표본 추출
arr = np.random.randint(10, 100, (10000,))
print(arr)
plt.hist(arr, bins=10)  # bins : 영역
plt.show()

# 5. np.random.random() : 0이상 1미만의 실수를 난수로 추출
#                         균등 분포로 난수를 추출
#    np.random.rand(d0, d1, d2, d3, ...)
#    np.random.random(10000,)  => shape이 들어감
arr = np.random.random((10000,))
print(arr)
plt.hist(arr, bins=10)  # bins : 영역
plt.show()
```



##### seed

- 실행할 때마다 같은 난수가 추출되도록 설정
- 난수의 재현

```python
np.random.seed(1)
arr = np.random.randint(0, 100, (10,))
print(arr)
```

##### shuffle

- ndarray의 순서를 섞고 싶을 때

```python
arr = np.arange(10)
print(arr) # => [0 1 2 3 4 5 6 7 8 9]               
# ndarray의 순서를 섞고 싶어요
np.random.shuffle(arr)
print(arr) # => [9 5 3 0 8 4 2 1 6 7]
```



### ndarray의 형태

#### reshape

- ndarray를 생성한 후 형태를 제어할 수 있음
- reshape() 함수는 새로운 ndarray를 만드는 것이 아닌 view를 생성
- 표현식 : arr.reshape(shape)

```python
import numpy as np

arr = np.arange(12) # 12개의 요소를 가지는 1차원의 ndaraay
print(arr) # => [ 0  1  2  3  4  5  6  7  8  9 10 11]

arr1 = arr.reshape(3, 4) # 3행 4열의 2차원의 ndarray로 변경
print(arr1) # => [[ 0  1  2  3]
            #     [ 4  5  6  7]
            #     [ 8  9 10 11]]

arr1[0, 2] = 200
print(arr1) # => [  0   1 200   3   4   5   6   7   8   9  10  11]
# reshqpe()은 view를 생성하기 때문에 형태는 그대로 1차원에 값만 바뀐 형태
```



- -1이 들어갈 경우 환경에 맞춰서 자동으로 형태 구성

```python
arr = np.arange(12) # 12개의 요소를 가지는 1차원의 ndaraay
print(arr) # => [ 0  1  2  3  4  5  6  7  8  9 10 11]

arr1 = arr.reshape(3, 4)
print(arr1) # => [[ 0  1  2  3]
            #     [ 4  5  6  7]
            #     [ 8  9 10 11]]

arr1 = arr.reshape(2, -1, 3)  # -1은 특별한 의미를 가짐
                              # 환경에 맞춰서 자동으로 구성
print(arr1) # => [[[ 0  1  2]
            #      [ 3  4  5]]
            #
            #     [[ 6  7  8]
            #      [ 9 10 11]]]
```



##### copy

- reshape()은 새로운 ndarray를 만드는 게 아니라, view를 생성

- copy는 내용을 복사해서 새로운 ndarray 생성

```python
arr = np.arange(12) # 12개의 요소를 가지는 1차원의 ndaraay
print(arr) # => [ 0  1  2  3  4  5  6  7  8  9 10 11]


arr1 = arr.reshape(3,4).copy()  
print(arr1) # => [[ 0  1  2  3]
            #     [ 4  5  6  7]
            #     [ 8  9 10 11]]

arr1[0,0] = 100
print(arr1) # => [[100   1   2   3]
            #     [  4   5   6   7]
            #     [  8   9  10  11]]
			# 값뿐 아니라 형태도 바뀌었음을 알 수 있음
```



#### ravel

- ndarray가 가지고 있는 모든 요소를 포함하는 1차원의 ndarray로 변경
- reshape과 동일하게 view를 생성

```python
arr = np.array([[1,2,3], [4,5,6]])
print(arr)

arr1 = arr.ravel() # View
print(arr1)
```



#### resize

- ndarray의 형태 변경하는 메소드
- 결과를 리턴하지 않고 원본을 바꿈

```python
arr = np.array([[1,2,3], [4,5,6]])
print(arr)   # [[1 2 3]
             # [4 5 6]]

arr1 = arr.resize(1, 6)  # 결과를 리턴하지 않고 원본을 바꿈
arr.resize(1, 6)
print(arr)   # [[1,2,3,4,5,6]]
print(arr1)  # None
```



- 복사본을 만들려면

```python
arr1 = np.resize(arr, (1, 6))  # 원본은 불변, 복사본이 만들어짐
print(arr1)  # [[1 2 3 4 5 6]]
print(arr)   # [[1 2 3]
             # [4 5 6]]
```



- reshape과 다르게 사이즈가 안 맞아도 변경 가능
- 모자란 데이터는 자동적으로 0이 삽입

```python
arr.resize(3, 4) # 원본이 2행 3열인데 3행 4열로 변경
                 # reshape은 안됨
                 # resize는 reshape과 다르게 사이즈가 안 맞아도 변경
                 # 모자란 데이터는 자동적으로 0이 삽입
print(arr)       # [[1 2 3 4]
                 # [5 6 0 0]
                 # [0 0 0 0]]
```

