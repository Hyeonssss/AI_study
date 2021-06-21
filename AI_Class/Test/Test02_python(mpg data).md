# Test02_python(mpg data)

## python data 처리 연습문제

### 데이터 로드

```python
class Mpg:
    def __init__(self, car_data):
        self.manufacturer = car_data[0]  # 제조업체 이름
        self.model = car_data[1]         # 모델명
        self.displ = car_data[2]         # 엔진 배기량(리터)
        self.year = car_data[3]          # 제조년도
        self.cyl = car_data[4]           # 실린더 수
        self.trans = car_data[5]         # 전송 유형
        self.drv = car_data[6]           # 드라이브 트레인의 유형
                                         # f: 전륜 구동, r: 후륜 구동, 4: 4wd
        self.cty = car_data[7]           # 도시 연비
        self.hwy = car_data[8]           # 고속도로 연비
        self.fl = car_data[9]            # 연료 종류
        self.car_class = car_data[10]    # 자동차의 유형
        
    def __lt__(self, other):             # little then
        return self.hwy < other.hwy

    def __str__(self):
        return f"{self.manufacturer}, {self.model}, {self.displ}, "\
            f"{self.year}, {self.cyl}, {self.trans}, {self.drv}, "\
            f"{self.cty}, {self.hwy}, {self.fl}, {self.car_class}"

    
my_file = open('mpg.txt', 'r')
my_file.readline()

car_list = []
while True:
    line = my_file.readline()
    if len(line) < 1:
        pass
    else:
        line2 = line.strip('\n')
        split_list = line2.split(',')
        car_list.append(Mpg(split_list))
    if not line:
        break
    
my_file.close() 

for i in car_list:
    print(i)
```

```
audi, a4, 1.8, 1999, 4, auto(l5), f, 18, 29, p, compact
audi, a4, 1.8, 1999, 4, manual(m5), f, 21, 29, p, compact
audi, a4, 2, 2008, 4, manual(m6), f, 20, 31, p, compact
audi, a4, 2, 2008, 4, auto(av), f, 21, 30, p, compact
audi, a4, 2.8, 1999, 6, auto(l5), f, 16, 26, p, compact
audi, a4, 2.8, 1999, 6, manual(m5), f, 18, 26, p, compact
audi, a4, 3.1, 2008, 6, auto(av), f, 18, 27, p, compact
audi, a4 quattro, 1.8, 1999, 4, manual(m5), 4, 18, 26, p, compact
audi, a4 quattro, 1.8, 1999, 4, auto(l5), 4, 16, 25, p, compact
audi, a4 quattro, 2, 2008, 4, manual(m6), 4, 20, 28, p, compact
audi, a4 quattro, 2, 2008, 4, auto(s6), 4, 19, 27, p, compact
audi, a4 quattro, 2.8, 1999, 6, auto(l5), 4, 15, 25, p, compact
audi, a4 quattro, 2.8, 1999, 6, manual(m5), 4, 17, 25, p, compact
```



#### 1.

- displ(배기량)이 4 이하인 자동차와 5 이상인 자동차 중 어떤 자동차의 hwy(고속도로 연비)가 평균적으로 더 높은지 확인하세요.

```python
low_count, sum_low_displ, high_count, sum_high_displ = 0,0,0,0

for i in range(len(car_list)):
    if float(car_list[i].displ) <= 4:
        sum_low_displ += int(car_list[i].hwy)
        low_count += 1
    elif float(car_list[i].displ) >= 5:
        sum_high_displ += int(car_list[i].hwy)
        high_count += 1
        
low_mean, high_mean = sum_low_displ / low_count, sum_high_displ / high_count

if low_mean > high_mean:
    print('배기량이 4 이하인 자동차의 고속도로 연비가 더 높습니다')
else:
    print('배기량이 5 이상인 자동차의 고속도로 연비가 더 높습니다')
print(f"배기량이 4 이하 자동차의 고속도로 연비 평균 : {low_mean}")
print(f"배기량이 5 이상 자동차의 고속도로 연비 평균 : {high_mean}")
```

- ```
  배기량이 4 이하인 자동차의 고속도로 연비가 더 높습니다
  배기량이 4 이하 자동차의 고속도로 연비 평균 : 25.96319018404908
  배기량이 5 이상 자동차의 고속도로 연비 평균 : 18.07894736842105
  ```



#### 2.

- "audi"와 "toyota" 중 어느 manufacturer(제조회사)의 cty(도시 연비)가 평균적으로 더 높은지 확인하세요.

```python
audi_count, sum_audi_cty, toyota_count, sum_toyota_cty =0, 0, 0, 0

for i in range(len(car_list)):
    if car_list[i].manufacturer == "audi":
        sum_audi_cty += int(car_list[i].cty)
        audi_count += 1
    elif car_list[i].manufacturer == "toyota":
        sum_toyota_cty += int(car_list[i].cty)
        toyota_count += 1
        
audi_mean, toyota_mean = sum_audi_cty/audi_count, sum_toyota_cty/toyota_count

if audi_mean > toyota_mean:
    print("audi의 도시 연비가 더 높습니다")
else:
    print("toyota의 도시 연비가 더 높습니다")
print(f"audi 자동차의 도시 연비 평균 : {audi_mean}")
print(f"toyota 자동차의 도시 연비 평균 : {toyota_mean}")
```

- ```
  toyota의 도시 연비가 더 높습니다
  audi 자동차의 도시 연비 평균 : 17.61111111111111
  toyota 자동차의 도시 연비 평균 : 18.529411764705884
  ```



#### 3.

- "chevrolet", "ford", "honda" 자동차의 고속도로 연비 평균

```python
count, sum_hwy = 0, 0
manufacturer_list = ["chevrolet", "ford", "honda"]

for i in range(len(car_list)):
    if car_list[i].manufacturer in manufacturer_list:
        sum_hwy += int(car_list[i].hwy)
        count += 1

mean = sum_hwy / count

print(f'"chevrolet", "ford", "honda"의 고속도로 연비 평균은 {mean}입니다')
```

- ```
  "chevrolet", "ford", "honda"의 고속도로 연비 평균은 22.50943396226415입니다
  ```



#### 4.

- "audi"에서 생산한 자동차 중 hwy가 1~5위

```python
audi_list = []
for i in range(len(car_list)):
    if car_list[i].manufacturer == "audi":
        audi_list.append(car_list[i])

audi_hwy_list = sorted(audi_list, reverse = True)

for i in range(5):
    print(f"{i+1}위: {audi_hwy_list[i]}")
```

- ```
  1위: audi, a4, 2, 2008, 4, manual(m6), f, 20, 31, p, compact
  2위: audi, a4, 2, 2008, 4, auto(av), f, 21, 30, p, compact
  3위: audi, a4, 1.8, 1999, 4, auto(l5), f, 18, 29, p, compact
  4위: audi, a4, 1.8, 1999, 4, manual(m5), f, 21, 29, p, compact
  5위: audi, a4 quattro, 2, 2008, 4, manual(m6), 4, 20, 28, p, compact
  ```



#### 5.

- 회사별로 "suv" 자동차의 평균 연비를 구한후 내림차순

```python
# suv 자동차의 리스트 생성
suv_list = []
for i in range(len(car_list)):
    if car_list[i].car_class == "suv":
        suv_list.append(car_list[i])


# suv 자동차의 리스트에서 제조회사명을 중복없이 추출
company_list = []
for i in range(len(suv_list)):
    if suv_list[i].manufacturer in company_list:
        pass
    else:
        company_list.append(suv_list[i].manufacturer)

        
# 회사별로 suv 자동차의 평균 연비를 추출
sum_list = []
for i in range(len(company_list)):
    temp_sum = 0
    count = 0
    for j in range(len(suv_list)):
        if suv_list[j].manufacturer == company_list[i]:
            temp_sum += (int(suv_list[j].cty) + int(suv_list[j].hwy)) / 2
            count += 1
    sum_list.append(temp_sum / count)
    

# 회사명과 suv 자동차의 평균 연비를 연결하는 dict 생성    
mean_dict = {}
for i in range(len(sum_list)):
    mean_dict[company_list[i]] = sum_list[i]


# 평균 연비 순위를 구해 mean_desc 리스트에 삽입
mean_desc = sorted(mean_dict.items(), key = (lambda x:x[1]), reverse = True)


# 1위부터 5위까지의 회사명과 연비 평균 데이터 출력
for i in range(5):
    print(f"{i+1}위- 회사명: {mean_desc[i][0]}, 연비평균: {mean_desc[i][1]}")
```

- ```
  1위- 회사명: subaru, 연비평균: 21.916666666666668
  2위- 회사명: toyota, 연비평균: 16.3125
  3위- 회사명: nissan, 연비평균: 15.875
  4위- 회사명: mercury, 연비평균: 15.625
  5위- 회사명: jeep, 연비평균: 15.5625
  ```



#### 6.

- class별 cty 평균을 구하고 cty 평균이 높은 순

```python
# 자동차의 유형을 중복없이 추출
class_list = []
for i in range(len(car_list)):
    if car_list[i].car_class in class_list:
        pass
    else:
        class_list.append(car_list[i].car_class)


# 자동차 유형별 도시연비 평균을 추출        
sum_list = []
for i in range(len(class_list)):
    temp_sum = 0
    count = 0
    for j in range(len(car_list)):
        if car_list[j].car_class == class_list[i]:
            temp_sum += int(car_list[j].cty)
            count += 1
    sum_list.append(temp_sum / count)
    
    
# 자동차 유형과 도시연비 평균을 연결하는 dict 생성    
mean_dict = {}
for i in range(len(sum_list)):
    mean_dict[class_list[i]] = sum_list[i]


# 도시연비 평균의 순위를 구해 mean_desc 리스트에 삽입
mean_desc = sorted(mean_dict.items(), key = (lambda x:x[1]), reverse = True)


# 도시연비 평균이 높은 순으로 자동차 유형과 도시연비 평균 데이터 출력
for i in range(len(class_list)):
    print(f"{i+1}위- 자동차 유형: {mean_desc[i][0]}, 도시연비평균: {mean_desc[i][1]}")
```

- ```
  1위- 자동차 유형: subcompact, 도시연비평균: 20.37142857142857
  2위- 자동차 유형: compact, 도시연비평균: 20.127659574468087
  3위- 자동차 유형: midsize, 도시연비평균: 18.75609756097561
  4위- 자동차 유형: minivan, 도시연비평균: 15.818181818181818
  5위- 자동차 유형: 2seater, 도시연비평균: 15.4
  6위- 자동차 유형: suv, 도시연비평균: 13.5
  7위- 자동차 유형: pickup, 도시연비평균: 13.0
  ```



#### 7.

- hwy(고속도로 연비) 평균이 가장 높은 회사 세 곳

```python
# 제조 회사명을 중복없이 추출
company_list = []
for i in range(len(car_list)):
    if car_list[i].manufacturer in company_list:
        pass
    else:
        company_list.append(car_list[i].manufacturer)


# 회사별 고속도로연비 평균을 추출        
sum_list = []
for i in range(len(company_list)):
    temp_sum = 0
    count = 0
    for j in range(len(car_list)):
        if car_list[j].manufacturer == company_list[i]:
            temp_sum += int(car_list[j].hwy)
            count += 1
    sum_list.append(temp_sum / count)
    
    
# 회사와 고속도로연비 평균을 연결하는 dict 생성    
mean_dict = {}
for i in range(len(sum_list)):
    mean_dict[company_list[i]] = sum_list[i]


# 고속도로연비 평균의 순위를 구해 mean_desc 리스트에 삽입
mean_desc = sorted(mean_dict.items(), key = (lambda x:x[1]), reverse = True)


# 고속도로연비 평균이 높은 순으로 회사명과 고속도로연비 평균 데이터 출력
for i in range(3):
    print(f"{i+1}위- 회사명: {mean_desc[i][0]}, 고속도로연비평균: {mean_desc[i][1]}")
```

- ```
  1위- 회사명: honda, 고속도로연비평균: 32.55555555555556
  2위- 회사명: volkswagen, 고속도로연비평균: 29.22222222222222
  3위- 회사명: hyundai, 고속도로연비평균: 26.857142857142858
  ```



#### 8.

- 각 회사별 "compact" 차종 수를 내림차순

```python
# compact 자동차의 리스트 생성
compact_list = []
for i in range(len(car_list)):
    if car_list[i].car_class == "compact":
        compact_list.append(car_list[i])


# compact 자동차의 리스트에서 제조회사명을 중복없이 추출
company_list = []
for i in range(len(compact_list)):
    if compact_list[i].manufacturer in company_list:
        pass
    else:
        company_list.append(compact_list[i].manufacturer)

        
# 회사별로 compact 자동차의 개수를 추출
sum_list = []
for i in range(len(company_list)):
    count = 0
    for j in range(len(compact_list)):
        if compact_list[j].manufacturer == company_list[i]:
            count += 1
    sum_list.append(count)


# 회사명과 compact 자동차의 개수를 연결하는 dict 생성    
cnt_dict = {}
for i in range(len(sum_list)):
    cnt_dict[company_list[i]] = sum_list[i]


# 차종 수를 구해 mean_desc 리스트에 삽입
cnt_desc = sorted(cnt_dict.items(), key = (lambda x:x[1]), reverse = True)


# compact 차종 개수가 높은 순으로 회사명과 compact 차종 개수 데이터 출력
for i in range(len(company_list)):
    print(f"{i+1}위- 회사명: {cnt_desc[i][0]}, compact 차종 수: {cnt_desc[i][1]}")
```

- ```
  1위- 회사명: audi, compact 차종 수: 15
  2위- 회사명: volkswagen, compact 차종 수: 14
  3위- 회사명: toyota, compact 차종 수: 12
  4위- 회사명: subaru, compact 차종 수: 4
  5위- 회사명: nissan, compact 차종 수: 2
  ```