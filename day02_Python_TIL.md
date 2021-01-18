# day2_Python_TIL

## python 기본

### python 데이터 타입

1. numeric
   - int, float, complex
2. sequence
   - list, tuple, range
3. text sequence(string)
   - str
4. Mapping
   - dict(dictionary)
5. Set
   - set
6. Bool
   - bool



#### range

 - range( [시작], 끝, [증가치] )

   [시작] default: 0

   [증가치] default: 1

   

#### string

> python 문자열

- (+) 연산은 concatnate 연산

```python
a = 'this is a'
b = ' same'
c = ' text'

print(a + b + c) 
# => this is a same text
```

- len(x)는 길이를 나타내는 함수(length)
- f.string 문자열 출력

```python
name = 김현석
school = 상명대학교
age = 27

print(f"안녕하세요 저는 {name}이고 나이는 {age}세입니다. 학교는 {school}입니다.")
# => 안녕하세요 저는 김현석이고 나이는 27세입니다. 학교는 상명대학교입니다.
```



#### Mapping(dict)

> dictionary

- 기호: {}
- key와 value의 쌍으로 표현

```python
ex_dict = {'name':'홍길동', 'age':27}
```

- 새로운 내용 추가

```python
ex_dict['adress'] = '서울'
```

- key, value 출력방법

```python
# key만 출력
print(ex_dict.keys())
# value만 출력
print(ex_dict.value())
```



#### Set

> set: 중복을 배제, 순서가 없는 자료형

- 기호: {}

```python
# 생성
ex_set = {} # 이는 dict로 간주
ex_set = set() # 이렇게 만들어야 함
```

- 집합 연산 사용 가능

```python
# 교집합
result = a & b # intersection
# 합집합
result = a | b # union
# 차집합
result = a - b # difference
```

- set 데이터 추가

```python
# 데이터 1개 추가할 때
ex_set.add(x) 
# 데이터 여러개 추가할 때
ex_set.update({1, 2, 3, 5})

# set은 순서가 없는 자료형이기 때문에 append 사용 불가
# append는 순서가 있는 자료형 맨 끝에 데이터 붙일 때 사용
# set은 순서가 없기 때문에 indecsing 또한 불가
```



#### bool

> bool - 논리형(True, False)

- 사용 가능 연산자

  and, or, not

```python
# or은 하나라도 참이면 True
print(True or False) # True
# and는 모든 게 참이면 True
print(True and False) # False
```

- python에서 다음은 False로 간주
  1. 빈 문자열 : ''
  2. 빈 리스트 : []
  3. 빈 tuple : ()
  4. 빈 dict : {}
  5. 숫자 0
  6. None



### 제어문

#### if

> if ~ elif ~ else 구문을 이용

```python
a = 20
if a % 3 == 0:
    print('3의 배수')
elif a % 5 == 0:
    print('5의 배수')
else:
    pass
```



### 반복문

#### for

> 반복횟수를 알고 있을 때 사용

- 표현: 

  for index in sequence(range, list, tuple)

```python
for i in [1, 2, 3, 4, 5]:
	print(i)
# => 1
#	 2
#	 3
#	 4
#	 5
```

- print()의 default 속성은 줄바꿈
- 가로로 출력하고 싶을 땐 : end 속성 이용

```python
for i in [1, 2, 3, 4, 5]:
    print(i, end = ' ')
# => 1 2 3 4 5
```



#### while

> 조건에 따라서 반복할 때 사용

```python
idx = 0
while idx < 10:
    print(idx, end = ' ')
    idx += 1
# => 0 1 2 3 4 5 6 7 8 9
```



#### list comprehension

> 리스트를 생성할 때 반복문과 조건문을 이용해서 생성
>
> list에만 해당함

```python
a = [1, 2, 3, 4, 5, 6, 7]

# 뒤에서부터 해석
# 리스트 a의 원소들을 반복하는데 원소가 짝수 인 것만 2를 곱해서 출력
list1 = [i*2 for i in a if i%2==0]

print(list1, end = ' ')
# => [4, 8, 12]

for i in list1:
    print(i, end = ' ')
# => 4 8 12
```



