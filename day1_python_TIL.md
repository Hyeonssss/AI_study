# TIL python day1

## Python 개요

### 특징

1. 배우기 쉽다.

2. 문법이 간결하다.

3. 코드의 가독성이 높다.

4. 다양한 분야에 사용 가능하다.

   웹프로그래밍, DB 프로그래밍, 데이터 분석, AI 등

   Mobile App을 만들기엔 적합하지 않다.

   시스템 프로그래밍엔 적합하지 않다.



### 개발환경 설정

1. 일반적인 개발환경(일반 python 프로그램, 웹 프로그램)

   => 파이참이라는 IDE 이용

   => 주피터노트북이라는 웹 IDE 이용

2. anaconda prompt 관리자 권한으로 실행

3. pip 최신버전으로 업데이트

   python -m pip install --upgrade pip

4. conda create -n  data_env python = 3.7 openssl

5. conda activate data_env

6. conda install nb_conda

7. jupyter notebook --generate-config

   폴더 경로를 찾아가기 .jupyter

   jupyter_notebook_config.py를 노트패드로 열기

   찾기: notebook_dir

   변경: 주석 제거, '' 안에 home 디렉토리 설정

   'D:/python_ML' 이후 저장

   D드라이브에 python_ML 폴더 생성 후 실행

8. jupyter notebook

## Python 기본

### Python의 Data Type

> (Built-in Data Type)

1. numeric
   1. int, float, complex
2. sequence
   1. list
   2. tuple
3. text sequence(문자열)
4. mapping
5. set
6. bool

