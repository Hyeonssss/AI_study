# day3_Python_TIL

## python 기본

### python의 함수

> 함수는 특정 기능을 수행하는 코드 묶음

- 이름 : 소문자 + 밑줄(student_name)

- python 내장함수(print(), len())

#### 사용자 정의 함수

> user define function

```python
def my_sum(a, b, c, d):
    result = a + b + c +d
    return result

sum_result = my_sum('hello', ' my', ' world', ' error')
print(sum_result)
# => hello my world error
```

- 함수 사용 시 인자가 고정되어 있지 않은 경우

  인자의 개수가 가변적인 경우

```python
def my_sum2(*tmp):
    result = 0
    for index in tmp:
        result += index
    return result

print(my_sum2(1,2,6,8,9,7))
# => 합 출력
```

- 여러 개의 값을 리턴하는 함수

```python
def multi_return(a, b):
    result1, result2 = a + b, a* b
    return result1, result2

data1, data2 = multi_return(10, 20)
print(data1, data2)
# => 30 200
```

- default parameter

  맨 마지막 인자 하나만 default parameter 사용 가능

```python
def default_param(a, b, c=False):
    if c:
        result = a + b + c
    else:
        result = a * b
    return result

print(default_param(2,5,10))
# => 17
print(default_param(2,5))
# => 10
```

#### call-by-value, call-by-reference

> 넘겨준 인자값이 변경되지 않는 경우: call-by-value => immutable
>
> 넘겨준 인자값이 변경되는 경우: call-by-reference => mutable

```python
def my_func(tmp_value, tmp_list):
    tmp_value = tmp_value + 100
    tmp_list.append(100)
    

data_x = 10     # immutable 
                # 숫자, 문자열, tuple
data_list = [10, 20]    # mutable
                        # list, dict

my_func(data_x, data_list)
print(f"data_x : {data_x}")   # 숫자는 값을 복사만, 원본 변경 x       
# => data_x : 10
print(f"data_list : {data_list}")    # 리스트, 딕셔너리는 원본도 변경 o
# => data_list : [10, 20, 100]
```



#### local variable, global variable

> 지역변수, 전역 변수

```python
tmp = 100			# 모듈 스코프

def my_func(x):		# 함수 스코프
    global tmp		# 밖에 있는 tmp를 가져다 쓰겠다.
    tmp += x		# global이 없으면 tmp는 이름은 같지만 다른 변수로 간주
    return tmp

print(my_func(20))

# global keyword의 사용은 의존성에 있어서 좋지 않음
# 만들 때는 편할 수 있으나 유지보수에 좋지 않음, 피해야 하는 일
```



#### python의 내장함수

1. all(x) 

   반복 가능한 자료형에 모든 값이 True면 True

   만약 하나라도 False이면 False 처리를 해주는 함수

2. any(x)

   반복 가능한 자료형에 하나라도 True면 True

   모든 데이터가 False이면 False 처리를 해주는 함수

3. len(x) 

   입력값의 길이를 알려주는 함수

4. int(), float(), list(), tuple(), dict(), str(), set()

   안에 들어오는 인자를 해당 데이터 타입으로 변경하는 함수



#### lambda

> lambda는 한 줄로 함수를 정의하는 방법

- 함수처럼 사용되나 함수는 아님
- 함수의 이름이 없기 떄문에 anonymous function이라고도 함
- 또는 람다식(lambda expression)으로 부름

- 변수 = lambda 입력변수1, 입력변수2, ... : 대체 표현식

```python
f = lambda a, b, c: a + b + c
    # 별도의 독립 공간 없이 대체 표현식 사용
    # 리턴해서 결과값을 보여주는 게 아니라
    # 단순히 표현식을 바꿔주는 것

def my_sum(a, b, c):           
    # 함수는 별도의 스코프가 있기 때문에 return해야함
    # 별도의 독립적인 공간 사용
    return a + b + c 

print(f(10, 20, 30))
# => 60
print(my_sum(10, 20, 30))
# => 60
```



### 객체지향

#### 프로그래밍의 패러다임

- 과거에는 이진수로 프로그래밍 => 0100001101
- 어셈블리어 : ADD, REGISTER
- C를 기반으로 고급언어들이 등장

#### 절차적 프로그래밍

> 구조적 프로그래밍

- 대표적인 언어: C
- 프로그램을 기능으로 세분화
- 파악된 각각의 기능을 모듈로 구현(함수 이용)
- 장점
  1. 프로그램을 쉽고 빠르게 구현
  2. 프로그램의 설계를 빠르게 할 수 있음
  3. 누가 설계를 하든 거의 동일한 설계가 나옴
- 단점
  1. 프로그램의 규모가 커지게 되면 복잡, 유지보수가 힘듦
  2. 개발비용보다 유지보수 비용이 더 커지는 현상
  3. 기존 코드를 재 사용하는데 한계
  4. 함수 단위로 가져다 쓰든지, 코드를 복붙해서 사용
  5. 재사용성에 한계

- 이후 인터넷의 보급
- 유지보수의 요구사항이 급증

#### 객체지향 프로그래밍

- 대표적 언어: C++, Java

- 프로그램을 기능으로 세분화하지 않음

- 해결해야하는 문제를 그대로 프로그램으로 묘사

- 프로그램을 구성하는 주체를 파악

- 단점

  프로그램의 설계와 구현이 상대적으로 어렵다

- 장점

  프로그램의 유지보수성과 재사용성에 이점

- 현실세계의 개체들을 프로그램으로 모델링

1. 속성: 변수(속성, 필드, property, attribute)

2. 행위: 함수 => method(함수의 특별한 형태)

   특정한 개체 안에서 행위를 묘사하는 함수 => method

3. class

   객체를 모델링학 위해서 사용하는 프로그램 단위

   - 객체 모델링의 수단
   - 추상 데이터 타입
   - 속성과 메소드의 집합

   현실세계의 개체를 프로그램적으로 묘사하기 위해 사용하는 설명서 같은 개념

   class를 기반으로 프로그램에서 사용할 수 있는 메모리 영역 할당 가능

4. instance => 객체(class 내에서 할당받은 메모리 공간)

```python
class Student:
    # initializer(생성자-constructor)
    def __init__(self, name, dept, num, grade):
        self.name = name
        self.dept = dept
        self.num = num
        self.grade = grade
    
    def __str__(self):
        return f"제 이름은 {self.name}이고 {self.dept}과입니다."\
                f" 학번은 {self.num}이고 학점은 {self.grade}입니다."

# self: 객체를 의미    
# print(studaent[1].dept) # dept() 괄호가 있으면 메소드
                          # dept 괄호가 없으면 속성

# 이렇게 상속을 받으면 특정 클래스의 속성 및 method를 사용할 수 있다.
class MyClass(Student):
    pass
me = MyClass('김현석', '경영학', '201410671', 4.02)
print(me)

students = []
students.append(Student('홍길동', '철학', '20200111', 3.5))
students.append(Student('김길동', '영어영문', '20200113', 4.5))
students.append(Student('신사임당', '컴퓨터', '20200115', 1.5))

# 객체가 가지는 속성이나 method를 access할때는 연산자를 이용 '.'
# dot operator라고 함
```



```python
my_list = list() # list class의 instance를 생성하는 코드
print(type(my_list)) # type()이 하는 일은 인스턴스가 어떤 class로부터
                     # 파생되었는 지 알려줌
# python에서 사용되는 함수는 객체.
```

