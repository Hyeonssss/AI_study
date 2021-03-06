# day17_web_TIL



### 설계

##### index.html

- 투표 주제 모아놓은 페이지

##### detail.html

- 특정 주제의 투표를 진행할 수 있는 페이지

##### result.html

- 특정 주제의 투표 결과를 보여주는 페이지



#### Table 설계

##### Question

- column
  - id - PK, not null
  - question_text
  - pub_date - 질문을 만든 날짜, 'yyyymmdd'

##### Choice

- column
  - id - PK
  - choice_text
  - votes - 투표 수
  - qustion_id - Foreign Key



### 프로젝트 생성

##### anaconda prompt

- cd.. cd.. 으로 루트 경로로 이동 후 cd python-Django

- django-admin startproject mysite

- Installed_apps

  polls.apps.pollsconfig 추가



- Django를 사용하는 이유: ORM을 사용할 수 있기 때문에

  class를 이용해서 table을 조작할 수 있다.



#### MVT pattern

##### Model

1. Model 생성 - Database Table 생성 (Class로 구현)

   polls/models.py (model을 정의하는 file) - Question, Choice

###### models.py

```python
from django.db import models

class Question(models.Model):
    # 이렇게 정의되는 class가 데이터베이스의 Table과 mapping
    # 그러면 Table의 column은 어떻게 정의하나요? => 속성으로 표현

    # 속성 정의
    # max_length : 글자수 최대 길이
    question_text = models.CharField(max_length=200)
    pub_date = models.DateTimeField('date published')

    def __str__(self):
        return self.question_text

class Choice(models.Model):
    Choice_text = models.CharField(max_length=200)
    votes = models.IntegerField(default=0)
    # Foreign key의 제약사항 : 참조무결성
    # on_delete=.models.CASCADE :
    # Question에서 record를 삭제할 때 Choice의 record도 함께 삭제
    question = models.ForeignKey(Question, on_delete=models.CASCADE)

    def __str__(self):
        return self.Choice_text
```



###### admin.py

- register your models here.

- admin.site.register() : admin site에 Table을 등록하는 명령어



- teminal 명령어 : 변경사항 저장

  python manage.py makemigrations : 변경사항 저장 파일 생성

  python manage.py migrate : 변경사항 파일 적용 

  python manage.py runserver : admin 페이지 작동

```python
from django.contrib import admin

# 이 파일 내에서 인식할 수 있도록 Qusetion, Choice class를 import
from polls.models import Question, Choice

# Register your models here.
admin.site.register(Question)
admin.site.register(Choice)


```



#### 구조

- URL (client request)

  (http://localhost:8000/polls/)

  - URLConf로 view에 접근
  - 클라이언트의 request를 모아서 하나의 객체로 자동으로 만들어서 보내준다(Django)

- View (로직)

  index() - 함수 (만들어야하는 함수)

  model을 이용해서 질문내용을 가져온다

  질문내용을 template에 보냄

  - view는 views.py에 들어있음
  - 하나의 객체로 모아진 request들이 view함수의 인자로 자동으로 전달된다

- Template (화면)

  index.html

  view에서 가져온 데이터를 받아 결과화면을 완성함 (클라이언트에게 보여줄 내용)

  - request + model data + index.html 을 합친 결과물을 클라이언트에게 제공



###### urls.py

```python
from django.contrib import admin
from django.urls import path
from polls import views

#  http://localhost:8000/polls/
urlpatterns = [
    path('admin/', admin.site.urls),
    # 특정 URL을 검색하면 2번째 인자인 함수가 실행된다.
    path('polls/', views.index, name='index')
]
```



###### views.py

```python
from django.shortcuts import render
from polls.models import Question


# Create your views here.
def index(request):
    # 데이터베이스를 뒤져서 설문목록을 가져온다
    # ( 테이블명 : polls_question, 클래스명 : Question )
    # objects.all() : 테이블 안에 모든 record 를 들고 오세요
    # order_by( '-x' ) : x 를 기준으로 내림차순 정렬
    question_list = Question.objects.all().order_by('-pub_date')[:5]

    # context : 데이터 전달용 dict 를 생성
    context = {'q_list': question_list}
    return render(request, 'index.html', context)
```



###### index.html

```django
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
</head>
<body>
{# {% python 로직 %}: python 로직을 표현 #}
{# {{ 문자열로 표현되는 값 }}: 문자열로 표현되는 값 #}
{# python의 if문과 비슷, template 언어 표현식 #}
    {% if q_list %} {# 리스트 안에 값이 있으면 실행 #}
        <ul>
            {% for question in q_list %}
                {# <a href=""></a>: 링크 걸기 #}
                {# question.id 값으로 링크 반복 #}
                <li><a href="/polls/{{ question.id }}">{{ question.question_text }}</a></li>
            {% endfor %}
        </ul>
    {% else %} {# 리스트 안에 아무것도 없으면 실행 #}
        <h1>데이터가 없어요!!</h1>
    {% endif %}
</body>
</html>
```



###### setting.py

```python
"""
Django settings for mysite project.

Generated by 'django-admin startproject' using Django 2.2.5.

For more information on this file, see
https://docs.djangoproject.com/en/2.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/2.2/ref/settings/
"""

import os

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/2.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'j=g_!fr@upyibc!q5xjrd@%oqv_2owv5j1-m61f5!+(ix+%_oq'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['localhost', '127.0.0.1']


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'polls.apps.PollsConfig'
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'mysite.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'mysite.wsgi.application'


# Database
# https://docs.djangoproject.com/en/2.2/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}


# Password validation
# https://docs.djangoproject.com/en/2.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/2.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'Asia/Seoul'

USE_I18N = True

USE_L10N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/2.2/howto/static-files/

STATIC_URL = '/static/'

```



### 프로젝트 생성 순서 정리

#### anaconda prompt

1. cd .. -> cd.. -> cd python-Django

   python-Djang 폴더로 경로 이동

2. django-admin startproject (프로젝트 이름)

   admin 권한으로 프로젝트를 생성

3. 탐색기에서 최상위 폴더의 이름을 변경

   mysite -> MyFirstWeb

   MyFirstWeb: 내 프로젝트의 최상위 프로젝트 폴더

   mysite: 프로젝트 전체 설정이 들어있는 폴더

4. prompt에서 경로를 MyFirstWeb으로 이동

   cd MyFirstWeb

5. python manage.py startapp (application 이름)

   설문조사하는 app polls 생성



#### pycharm

##### settings.py

1. Allowed_Hosts 변경

   ALLOWED_HOSTS = ['localhost', '127.0.0.1']

2. installed_apps에 polls application의 Config 추가

   INSTALLED_APPS = ['polls.apps.PollsConfig']

3. 시간 기준을 서울로 설정

   TIME_ZONE = 'Asia/Seoul'



##### models.py

1. 데이터 베이스의 Table로 지정될 class 생성 (models class 상속 받아서 생성)

```python
from django.db import models

# Create your models here.
class Question(models.Model):
    question_text = models.CharField(max_length=200)
    pub_date = models.DateTimeField('date published')
    
    def __str__(self):
        return self.question_text
    
class Choice(models.Model):
    choice_text = models.CharField(max_length=200)
    votes = models.IntegerField(default=0)
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    
    def __str__(self):
        return self.choice_text
```



##### admin.py

1. Question, Choice class를 import
2. admin.site.register(): admin site에 table을 등록하는 명령어

```python
from django.contrib import admin
from polls.models import Question, Choice

# Register your models here.
admin.site.register(Question)
admin.site.register(Choice)
```



##### teminal

1. python manage.py makemigrations : 변경사항 저장 파일 생성

2. python manage.py migrate : 변경사항 파일 적용 

3. python manage.py runserver : admin 페이지 작동



##### anaconda prompt

1. python manage.py createsuperuser

   localhost:8000/admin 페이지에 접속하기 위한 superuser 계정 생성



##### localhost:8000/admin

1. polls 폴더 속 Choices, Questions 테이블이 생성된 것을 확인
2. Questions에 데이터 입력
3. Choices에 데이터 입력



##### urls.py

1. localhost:8000/polls/ 로 연결되는 url을 생성
2. url에 들어가면 views.py 파일 속에 있는 index 함수가 실행되도록 설정

```python
from django.contrib import admin
from django.urls import path
from polls import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('polls/', views.index, name='index')
]
```



##### views.py

1. urls.py에서 명시했던 views.index 함수가 실행되도록 index() 함수 생성
2. index 함수는 Question의 모든 데이터를 내림차순에 5개만 받아오는 함수 
3. render(): Template에서 사용자의 request와 html, view의 데이터를 합쳐서 하나의 완성된 결과를 리턴해주는 함수

```python
from django.shortcuts import render
from polls.models import Question

# Create your views here.
def index(request):
    question_list = Question.objects.all().order_by('-pub_date')[:5]
    
    context = {'q_list': question_list}
    return render(request,'index.html', context)
```



##### index.html

1. polls 폴더 내에 templates 폴더 생성 후 그 안에 html 파일 생성

2. Django의 template 표현법

   {% 파이썬 로직 %}

   {{ 문자열 리턴 변수 }}

3. if문 

   {% if %}

   {% else %}

   {% endif %}

4. for문

   {% for i in x %}

   {% endfor %}

```django
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
</head>
<body>
    {% if q_list %}
        <ul>
            {% for question in q_list %}
                <li><a href="/polls/{{ question.id }}">{{ question.question_text }}</a></li>
            {% endfor %}
        </ul>
    {% else %}
        <h1>데이터가 없어요!</h1>
    {% endif %}
</body>
</html>
```