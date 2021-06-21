# day4_Python_TIL

## python 기본

### class

```python
class Student:
    
    scholaship_rate = 3.0     # class variable
    
    def __init__(self, name, dept, grade):
        self.name = name    # instance variable
        self.__dept = dept    # instance variable  => private
        self.grade = grade  # instance variable
        
    def __str__(self):    
        return f"이름:{self.name}, 학과:{self.dept}, 학점:{self.grade}"
    
    def change_info(self, name, dept, grade):
        self.name = name
        self.__dept = dept
        if grade < 0 or grade > 4.5:
            print('학점의 범위를 벗어났습니다.')
        else:
            self.grade = grade
    
    @classmethod
    def change_scholaship_rate(cls, value):
        cls.scholaship_rate = value
        
    @staticmethod
    def hello():
        return "hello"
    
    def is_scholaship(self):
        if self.grade >= Student.scholaship_rate:
            return True
        else:
            return False
    
    
stu1 = Student('홍길동', '철학', 2.0)
stu2 = Student('신사임당','컴퓨터',4.0)
```



- 메소드 호출법

```python
print(stu1.is_scholaship) # => x
# 메모리 블록의 주소 출력
# method의 이름이 메모리 블록의 주소를 갖고 있음

print(stu1.is_scholaship)
# => False
```



#### namespace

> 객체들의 요소를 나누어서 관리하는 공간

- 상속(inheritance) = 객체지향의 꽃 => 재사용성 확보
  - 상속은 상위 클래스의 특징을 이어받아서 확장된 하위 클래스 생성 가능
  - 상속을 이용하면 class 간에 계층관계 성림
- 상위 class: 상속을 내려주는 클래스
  - super class, upper class, parent class, 부모 클래스

- 하위 class: 상속을 받아서 확장하는 클래스
  - sub class, child class, 자식 클래스
- namespace
  - 객체들의 요소를 나누어서 관리하는 공간
  - 속성이나 method 호출 시 계층구조에서 namespace를 따라가면서 속성과 method를 찾는다.
  - 인스턴스 네임스페이스 -> 클래스 네임스페이스 -> 슈퍼클래스 네임스페이스
  - 가장 하위클래스부터 상위클래스로 속성과 method를 찾는다.

```python
class Student:
    scholarship_rate = 3.0 # => class variable
    
stu1.scholarship_rate = 4.5 # => 클래스 변수의 값이 변하는 게 아니라 stu1 객체 안에 동일한 이름의 새로운 변수가 생성된다.
```

​		

- setter, getter 메소드
  - 인스턴스가 가지고 있는 속성은 외부에서 직접적인 변경이 불가능하도록 하는 게 좋다.
  - 메소드를 통해서 간접적으로 접근해 변경하도록 코드 작성

```python
class Student:
    def __init__(self, name, dept, grade):
        self.name = name    # instance variable
        self.dept = dept    # instance variable
        self.grade = grade  # instance variable
    
    # setter method
    def change_info(self, name, dept, grade):
        self.name = name
        self.dept = dept
        if grade < 0 or grade > 4.5:
            print('학점의 범위를 벗어났습니다.')
        else:
            self.grade = grade
            
stu1.name = '강감찬'
stu1.dept = '영어영문'
stu1.change_info('강감찬2', '영어영문2', -10)  # 맞는 방법

# 결과는 같으나 속성 자체를 액세스 하느냐, 메소드를 통해 간접적으로 액세스 하느냐의 차이
```



#### class method

- 인스턴스 메소드는 self를 인자로 받아서 사용
- 클래스 메소드는 cls를 인자로 받아서 사용

```python
class Student:
    
    scholaship_rate = 3.0     # class variable
    
	@classmethod
    def change_scholaship_rate(cls, value):
        cls.scholaship_rate = value
        
    # 클래스 변수인 scholaship_rate를 변경하는 클래스 메소드

Student.change_scholaship_rate(4.0)
print(Student.scholaship_rate)
# => 4.0
```



#### static method

1. instance variable

2. instance method

3. class variable

4. class method

5. static method

   => self나 cls를 인자로 받지 않음

   ​	일반적인 함수가 class 내부에 존재하는 경우

```python
class Student:
    
    @staticmethod
    def hello():
        return "hello"
```



#### public

> 어디에서나 속성과 함수를 사용할 수 있는 경우

```python
class Student:
	def __init__(self, name, dept, grade):
        self.name = name    # instance variable
        self.dept = dept    # instance variable  
        self.grade = grade  # instance variable

    def change_info(self, name, dept, grade):
        self.name = name
        self.dept = dept
        if grade < 0 or grade > 4.5:
            print('학점의 범위를 벗어났습니다.')
        else:
            self.grade = grade
            
stu1 = Student('홍길동', '철학', 4.0)
stu1.name = '이황' # => public
stu1.change_info('이이', '영어영문', 4.2) # => public
```

​	기본적으로 python은 public으로 지정



#### private

- 변수나 메소드 앞에 (__)를 붙이면 외부에서 접근 못하게 됨

  이유는 네이밍을 변경해버리기 때문

  변수나 메소드의 이름이 다르기 때문에 호출 시에 error

```python
class Student:
    def __init__(self, name, dept, grade):
        self.name = name    # instance variable
        self.__dept = dept    # instance variable  => private
        self.grade = grade  # instance variable
        
    def change_info(self, name, dept, grade):
        self.name = name
        self.__dept = dept
        if grade < 0 or grade > 4.5:
            print('학점의 범위를 벗어났습니다.')
        else:
            self.grade = grade
            
stu1 = Student('홍길동', '철학', 4.0)
stu1.__dept = '이황' # error => private
stu1.change_info('이이', '영어영문', 4.2) # error => public
```



### 상속

> inheritance

- 상속의 목적은 class의 재활용
- 코드의 반복을 줄이고 compact한 코드 작성
- 장점: 코드의 반복을 줄이고 재사용성을 높임
- 단점: 클래스를 재활용하려면 독립적인 class의 경우가 더 좋다
- 상속을 하게 되면 상위 클래스와 하위 클래스가 긴밀하게 연결(tightly copled)

```python
# 상위 클래스(super class, parent class, base class)
class Unit:
    def __init__(self, damage, life):
        self.utype = self.__class__.__name__
        self.damage = damage
        self.life = life
        
    def __str__(self):
        return f"{self.utype} - damage:{self.damage}, hp:{self.life}"
        
# 하위 클래스(sub class, child class)
class Marine(Unit):
    def __init__(self, damage, life, offense_upgrade):
        super().__init__(damage,life) # 부모클래스 __init__ 메소드 호출 
        self.offense_upgrade = offense_upgrade
    
    def __str__(self):
        return f"{self.utype} - damage:{self.damage}, hp:{self.life}, upgrade:{self.offense_upgrade}"
    
marine_1 = Marine(100, 200, 3)
print(marine_1)
# => Marine - damage:100, hp:200, upgrade:3
```



### magic function

> class 안에 들어있는 magic function

- 특징: 일반적으로 우리가 직접 호출하지 않는다.

  ​		  특정 상황에 자동적으로(내부적으로) 실행된다.

```python
# __init__(), __str__(), __del__(), __lt__(), ...

class Student:
    def __init__(self, name, dept, grade):
        print('객체가 생성돼요!!')
        self.name = name
        self.dept = dept
        self.grade = grade
        
    def __del__(self):     # instance가 메모리에서 삭제될 때 호출
        print('객체가 삭제돼요!!')
        # 객체가 삭제될 때 이 객체가 사용한 resource를 해제
    
    def __str__(self):
        return f"{self.name}, {self.dept}, {self.grade}"
    
    def __gt__(self, other):  # grate then
        return f'{self.grade > other.grade}, > 연산자에 의해 호출돼요!!'
    
    def __lt__(self, other):  # little then
        return f'{self.grade < other.grade}, < 연산자에 의해 호출돼요!!'

        
stu1 = Student('홍길동', '영어영문', 3.0)
stu2 = Student('신사임당', '철학', 4.0)

print(stu1)  # __str__ 메소드가 없으면 메모리 주소값이 출력
print(stu2)
print(stu1 < stu2)  # __lt__ 메소드 호출
print(stu1 > stu2)  # __gt__ 메소드 호출
del stu1

# =>
# 객체가 생성돼요!!
# 객체가 생성돼요!!
# 홍길동, 영어영문, 3.0
# 신사임당, 철학, 4.0
# True, < 연산자에 의해 호출돼요!!
# False, > 연산자에 의해 호출돼요!!
# 객체가 삭제돼요!!

# 객체가 생성되면 특정 메모리 주소에 메모리 공간이 할당(0x100)
# 두번째 실행에서 객체가 생성되면 특정 메모리 주소에 공간이 할당(0x200)
# 실행할 때마다 새로운 객체 생성
# 기존에 있던 객체는 파이썬이 알아서 삭제 처리
```



### module

> 함수나 변수 혹은 클래스를 모아놓은 파일을 지칭

- 확장자가 .py로 끝나는 python 소스코드는 모듈

  jupyter notebook을 이용한 .jpynb는 모듈이 아님

- 다른 python 파일을 불러와서 우리 코드에서 이용할 수 있도록 해주는 기능

  모듈을 이용하는 이유(파일을 나누어서 작성하는 이유)

  => 코드의 재사용성을 높이고 관리를 쉽게하기 위함

- import
  - 모듈을 불러들이는 키워드
  - 파일을 객체화 시켜서 우리 코드가 사용하는 메모리에 로드
  - import 시에는 파일의 이름만 이용(확장자 빼고)

- 윈도우의 환경변수
  - 환경변수 하나를 생성 - 특정 폴더 지정 - 파이썬 파일 작성
  - 이후에 이 파일을 모듈로 사용 가능

```python
# 다 같은 결과 module1.py의 기능들을 import

import module1 as m1

print(m1.my_pi)
# 모듈명을 앞에 붙이고 사용

from module1 import *
from module1 import Student, my_pi, my_sum

print(my_pi)
# 모듈의 변수나 함수 class를 직접 import 했기 때문에 모듈명 없이 바로 사용


# 같은 결과 여러 폴더 속 my_network_module.py를 import

import network.my_sub_folder.my_network_module as nm
from network.my_sub_folder import my_network_module

print(my_network_module.variable1)
# 파일을 import 했기 때문에 모듈명을 붙여서 사용

```

```python
# 다 같은 결과 module1.py의 기능들을 import

import module1 as m1

print(m1.my_pi)
# 모듈명을 앞에 붙이고 사용

from module1 import *
from module1 import Student, my_pi, my_sum

print(my_pi)
# 모듈의 변수나 함수 class를 직접 import 했기 때문에 모듈명 없이 바로 사용


# 같은 결과 여러 폴더 속 my_network_module.py를 import

import network.my_sub_folder.my_network_module as nm
from network.my_sub_folder import my_network_module

print(my_network_module.variable1)
# 파일을 import 했기 때문에 모듈명을 붙여서 사용

```