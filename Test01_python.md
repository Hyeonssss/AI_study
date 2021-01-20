# Test01_python

## python 연습문제



#### 1. 

- 3, 5 배수의 합

```python
result = 0
for i in range(1000):
    if i % 3 == 0:
        result += i
    elif i % 5 == 0:
        result += i
    else:
        pass
print(result)
```

- ```
  233168
  ```



#### 2.

- 피보나치 수열

```python
i = 1
previous = 1
current = 1
pivo = []
result = 0

while i <= 50:
    pivo.append(current)
    previous, current = current, previous + current
    i += 1

for i in pivo:
    if i % 2 == 0 and i <= 4000000:
        result += i

print(result)
```

- ```
  4613732
  ```



#### 3.

- 알파벳

```python
string1 = "This is a sample Program mississippi river"
string2 = "abcdabcdababccddcd"

up_string1 = string1.upper()
up_string2 = string2.upper()

result1 = 0
word1 = ''
for i in up_string1:
    if result1 < up_string1.count(i):
        result1 = up_string1.count(i)
        word1 = i
print(f'정답 : "{string1}" => {word1}')

result2 = 0
word2 = ''
for i in up_string2:
    if result2 < up_string2.count(i):
        result2 = up_string2.count(i)
        word2 = i
print(f'정답 : "{string2}" => {word2}')
```

- ```
  정답 : "This is a sample Program mississippi river" => I
  정답 : "abcdabcdababccddcd" => C
  ```



#### 4.

- lotto program

```python
# 쉬운 버전 로또
import random

my_lotto = []
lotto = []
for i in range(6):     # lotto 당첨 번호 여섯 개 뽑기
    a = random.randint(1,45)
    while a in lotto:  # 중복 없이 수 뽑기
        a = random.randint(1,45)
    lotto.append(a)     # 뽑은 랜덤 숫자 lotto 리스트에 삽입

for i in range(5):            # 나의 로또복권 5장 뽑기
    my_number = []
    for n in range(6):        # 나의 로또 번호 여섯 개 뽑기
        a = random.randint(1,45)
        while a in my_number: # 중복 없이 수 뽑기
            a = random.randint(1,45)
        my_number.append(a)    # 뽑은 랜덤 숫자 나의 로또 번호 리스트에 삽입
    my_lotto.append(my_number) # 나의 로또 번호 리스트를 나의 로또 복권 리스트에 삽입

one, two, three, four, fail = 0,0,0,0,0

for i in my_lotto:
    result =[]
    result = set(lotto) & set(i)
    
    if len(result) == 6:
        one += 1
    elif len(result) == 5:
        two += 1
    elif len(result) == 4:
        three += 1
    elif len(result) == 3:
        four += 1
    else:
        fail += 1

print(f"1등 : {one}개, 2등 : {two}개, 3등 : {three}개, 4등 : {four}개, 꽝 : {fail}개")
```

- ```
  1등 : 0개, 2등 : 0개, 3등 : 0개, 4등 : 1개, 꽝 : 4개
  ```



- 보너스 번호 추가 버전

```python
# 보너스 번호 추가 버전
import random

my_lotto = []
lotto = []

for i in range(6):     # lotto 당첨 번호 다섯 개 뽑기
    a = random.randint(1,45)
    while a in lotto:  # 중복 없이 수 뽑기
        a = random.randint(1,45)
    lotto.append(a)     # 뽑은 랜덤 숫자 lotto 리스트에 삽입

for i in range(5):            # 나의 로또복권 5장 뽑기
    my_number = []
    for n in range(6):        # 나의 로또 번호 여섯 개 뽑기
        a = random.randint(1,45)
        while a in my_number: # 중복 없이 수 뽑기
            a = random.randint(1,45)
        my_number.append(a)    # 뽑은 랜덤 숫자 나의 로또 번호 리스트에 삽입
    my_bonus = random.randint(1,45)  # 보너스 번호 추가
    while my_bonus in my_number:
        my_bonus = random.randint(1,45)
    my_number.append(my_bonus)       # 보너스 번호를 나의 로또 번호 리스트에 삽입
    my_lotto.append(my_number) # 나의 로또 번호 리스트를 나의 로또 복권 리스트에 삽입

one, two, thee, four, five, fail = 0,0,0,0,0,0

for i in my_lotto:
    result = []
    result = set(lotto) & set(i[:6])
    bonus = 0
    if i[-1] in lotto:
        bonus = 1
    
    if len(result) == 6:
        one += 1
    elif len(result) + bonus == 6:
        two += 1
    elif len(result) == 5:
        three += 1
    elif len(result) == 4:
        four += 1
    elif len(result) == 3:
        five += 1
    else:
        fail += 1

print(f"1등 : {one}개, 2등 : {two}개, 3등 : {three}개, 4등 : {four}개, 5등 : {five}개, 꽝 : {fail}개")
```

- ```
  1등 : 0개, 2등 : 0개, 3등 : 0개, 4등 : 0개, 5등 : 0개, 꽝 : 5개
  ```



#### 5.

- 소인수

```python
n_list = []
for i in range(1, 10000):
    n = 0
    for j in range(1, i + 1):
        if i % j == 0:
            n += 1
    if n == 2:
        n_list.append(i)

result = []
for i in n_list:
    if 600851475143 % i == 0:
        result.append(i)
        
print(max(result))
```

- ```
  6857
  ```



#### 6.

- 대칭수

```python
def is_palindrome(number):
    return number == number[::-1]

maxim = 0

for i in range(999, 1, -1):
    for j in range(999, 1, -1):
        n = j*i
        if is_palindrome(str(n)) == True:
            maxim = max(maxim, n)
            
print(maxim)
```

- ```
  906609
  ```



#### 7.

- 어떤 수로도 나눠지는 가장 작은 수

```python
n_list = [i for i in range(1,21)]

n = 232792500
count = 0
while count < len(n_list):
    n += 1
    count = 0
    for i in n_list:
        if n % i == 0:
            count += 1

print(n)
```

- ```
  232792560
  ```



#### 8.

- 합의 제곱과 제곱의 합

```python
def square_sum(x):
    square_sum = 0
    square_list = [i**2 for i in range(1, x+1)]
    for i in square_list:
        square_sum += i
    return square_sum

def sum_square(x):
    sum_square = 0
    sum_list = [i for i in range(1,x+1)]
    for i in sum_list:
        sum_square += i
    sum_square = sum_square ** 2
    return sum_square

print(sum_square(100) - square_sum(100))
```

- ```
  25164150
  ```



#### 9.

- 소수 10,001번째

```python
primes = [2]
number = 1
def IsPrimeNumber(number):
    for i in range(len(primes)):
        if number % primes[i] == 0:
            return False
    return True

while len(primes) < 10001:
    number += 2
    if IsPrimeNumber(number):
        primes.append(number)
        
print(primes[10000])
```

- ```
  104743
  ```



#### 10.

- 파타고라스

```python
for a in range(1, 1000):
    for b in range(1, 1000):
        c = 1000 - a - b
        if a * a + b * b == c * c and a < b < c:
            print(a * b * c)
```

- ```
  31875000
  ```

