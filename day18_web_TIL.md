# day18_web_TIL



### 구조

- URL (client URL) -> View (함수) -> Template (html)



#### urls.py

- 프로젝트 전체에 대한 url 처리를 mysite에서 실행하면 과부화 가능성이 있고 복잡하다

###### mysite\urls.py

- include() 함수를 이용해서 url 처리를 각 application 내에서 실행

```python
from django.contrib import admin
from django.urls import path, include


urlpatterns = [
    path('admin/', admin.site.urls),

    # 여기에 모두 적는 비효율을 방지하기 위해 include() 함수 사용
    # urls를 application 단위로 하나씩 만들어서 사용
    path('polls/', include('polls.urls'))
]
```



###### polls\urls.py

- 네임스페이스 처리: 논리적인 이름들이 겹칠 수 있기 떄문에 논리적인 구조로 이름 저장
- 추후 polls:index 와 같은 형식으로 저장됨
- 꺽새와 int : 정수이며 변하는 값이다

```python
from django.urls import path
from . import views

# polls application 안에 들어오는 url만을 처리

# 기본값 : http://localhost:8000/polls/
app_name = 'polls' # 네임스페이스 처리
urlpatterns = [
    path('', views.index, name='index'),
    # polls/1
    # <int> : 특수표현, 정수이며 변하는 값(꺾새)이다
    path('<int:question_id>/', views.detail, name='detail'),
    path('<int:question_id>/vote/', views.vote, name='vote')
    # http://localhost:8000/polls/1/vote/ = polls:vote
]
```



#### templates

- application내에 templates html 파일은 templates 하위에 app 이름의 폴더 생성
- polls/index.html 로 네임스페이스의 역할을 함
- polls 폴더 -> templates 폴더 -> polls 폴더 -> index.html 파일 

###### settings.py

- settings.py에 TEMPLATES 부분 수정

```python
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        # 기본 templates 경로 설정
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
        # 어플리케이션마다 template 폴더를 만들어서 사용할 수 있다.
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
```



#### 순서

urls -> views

get_object_or_404: 딱 하나의 객체만 가져오는 함수, 데이터 없으면 404 에러 발생



#### views

- detail 함수 추가

```python
from django.shortcuts import render, get_object_or_404
from polls.models import Question, Choice

# Create your views here.
def index(request):
    question_list = Question.objects.all().order_by('-pub_date')[:5]

    context = {'q_list': question_list}
    # 단순 index.html만 적어놓으면 최상위 폴더의 index 함수가 호출됨
    return render(request,'polls/index.html', context)

def detail(request, question_id):
    # 숫자 하나가 question_id로 들어옴 설문에 대한 PK
    question = get_object_or_404(Question, pk=question_id)
    # choice_list = Choice.objects.all().ord
    context = {'selected_question': question}
    return render(request, 'polls/detail.html',context)

def vote(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    try:
        selected_choice = question.choice_set.get(pk=request.POST['my_choice'])
        # get(pk=request.POST['my_choice'] = question.id.POST['choice.id']
    # Choice table에 key가 존재하지 않는 에러가 발생할 때
    except(KeyError, Choice.DoesNotExist):
        return render(request, 'polls/detail.html', {
            'selected_question': question,
            'error_message': '아무것도 선택하지 않았어요!'
        })
    else:
        # 해당 객체의 votes 값 1 올리기
        selected_choice.votes += 1
        # save() : 이를 토대로 값을 변경해주세여
        selected_choice.save()

        context = {'selected_question': question}
        return render(request, 'polls/detail.html', context)
```



#### detail.html

###### 초기

- form : 사용자가 클릭하는 부분

  csrf_token : 크래킹을 막기위해 form 첫줄에 무조건 써야함

- radio 버튼의 name 값이 동일하면 단일선택
- label : 특정 id에 레이블을 지칭하는 태그

```django
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
</head>
<body>
    <h1>{{ selected_question.question_text }}</h1>
    <form action="#" method="#">
    {# csrf_token: 해킹 기법은 크래킹을 막기위해 보안 위해서 사용 #}
    {% csrf_token %} {# form의 첫줄에 무조건 써야함 #}
        {# radio button의 name 속성값이 같으면 단일선택 됨 #}
        <input type="radio" name="my_choice" id="kaka01">
        {# label : 특정 id에 레이블을 지칭하는 태그 #}
        <label for="kaka01">kaka01</label><br>
        <input type="radio" name="my_choice" id="kaka02">
        <label for="kaka02">kaka02</label><br>
        <input type="radio" name="my_choice" id="kaka03">
        <label for="kaka03">kaka03</label>

    </form>
</body>
</html>
```



###### 반복문으로 변경

- set을 사용하면 연결된 테이블의 집합을 가져온다, views.py 파일에 choice 테이블을 명시하지 않고도 데이터를 가져올 수 있다.
- forloop.counter : index를 지정

```django
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
</head>
<body>
    <h1>{{ selected_question.question_text }}</h1>
    <form action="#" method="#">
    {% csrf_token %}
        {# .choice_set.all: 대상과 연결된 choice 테이블의 집합을 모두 가져온다 #}
        {# set을 사용하면 views.py 파일에 choice 테이블을 명시하지 않아도 데이터를 가져올 수 있다 #}
        {% for choice in selected_question.choice_set.all %}
            <input type="radio" name="my_choice"
                   {# forloop.counter : index #}
                    {# choice.id 써도 됨 #}
                   id="kaka{{ forloop.counter }}">
            <label for="kaka{{ forloop.counter }}">
                {{ choice.choice_text }}
            </label><br>
        {% endfor %}
    
    </form>
</body>
</html>
```



###### url 추가

- url 'polls:vote' selected_qustion = 'http://localhost:8000/polls/1/vote/'
- template 코드에선 함수에 ()를 표현하지 않음

```django
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
</head>
<body>
    <h1>{{ selected_question.question_text }}</h1>
    {# form에서는 method를 거의 POST로 이용 #}
    {# 서버쪽 프로그램의 호출은 polls:vote url을 이용할 거에요 #}
    {# question id는 변하는 값이기에 인자로 넘겨준다 #}
    {# http://localhost:8000/polls{{ selected_question }}/vote/ #}
    <form action="{% url 'polls:vote' selected_question %}" 
          method="POST">
    {# csrf_token: 해킹 기법은 크래킹을 막기위해 보안 위해서 사용 #}
    {% csrf_token %} {# form의 첫줄에 무조건 써야함 #}
        {# .choice_set.all: 대상과 연결된 choice 테이블의 집합을 모두 가져온다 #}
        {# set을 사용하면 views.py 파일에 choice 테이블을 명시하지 않아도 데이터를 가져올 수 있다 #}
        {% for choice in selected_question.choice_set.all %}
            <input type="radio"
                   name="my_choice"
                   id="kaka{{ forloop.counter }}"
                   {# 서버쪽에 전송하는 데이터 #}
                    {# for문은 name=value의 형태로 전달된다 #}
                   value="{{ choice.id }}">
            <label for="kaka{{ forloop.counter }}">
                {{ choice.choice_text }}
            </label><br>
        {% endfor %}
        <br>
        {# submit : 특수한 기능을 하는 버튼 #}
        <input type="submit" value="투표"
    </form>
</body>
</html>
```



###### 최종

- 에러처리 구문 추가

```django
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
</head>
<body>
    <h1>{{ selected_question.question_text }}</h1>
    {# error 발생시 출력 #}
    {% if error_message %}
        <div>{{ error_message }}</div>
    {% endif %}
    
    <form action="{% url 'polls:vote' selected_question.id %}"
          method="POST">
    {% csrf_token %} 
        {% for choice in selected_question.choice_set.all %}
            <input type="radio"
                   name="my_choice"
                   id="kaka{{ forloop.counter }}"
                   value="{{ choice.id }}">
            <label for="kaka{{ forloop.counter }}">
                {{ choice.choice_text }}
            </label><br>
        {% endfor %}
        <br>
        <input type="submit" value="투표">
    </form>
</body>
</html>

```





### results 추가

#### polls.urls.py

```python
from django.urls import path
from . import views

# polls application 안에 들어오는 url만을 처리

# 기본값 : http://localhost:8000/polls/
app_name = 'polls' # 네임스페이스 처리
urlpatterns = [
    path('', views.index, name='index'),
    # polls/1
    # <int> : 특수표현, 정수이며 변하는 값(꺾새)이다
    path('<int:question_id>/', views.detail, name='detail'),
    path('<int:question_id>/vote/', views.vote, name='vote'),
    # http://localhost:8000/polls/1/vote/ = polls:vote
    path('<int:question_id>/results/', views.results, name='results')
    # http://localhost:8000/polls/1/results/ = polls:vote
]
```



#### views.py

```python
from django.shortcuts import render, get_object_or_404
from polls.models import Question, Choice
from django.http import HttpResponseRedirect
from django.urls import reverse

# Create your views here.
def index(request):
    question_list = Question.objects.all().order_by('-pub_date')[:5]

    context = {'q_list': question_list}
    # 단순 index.html만 적어놓으면 최상위 폴더의 index 함수가 호출됨
    return render(request,'polls/index.html', context)

def detail(request, question_id):
    # 숫자 하나가 question_id로 들어옴 설문에 대한 PK
    question = get_object_or_404(Question, pk=question_id)
    # choice_list = Choice.objects.all().ord
    context = {'selected_question': question}
    return render(request, 'polls/detail.html',context)

def vote(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    try:
        selected_choice = question.choice_set.get(pk=request.POST['my_choice'])
        # get(pk=request.POST['my_choice'] = question.id.POST['choice.id']
    # Choice table에 key가 존재하지 않는 에러가 발생할 때
    except(KeyError, Choice.DoesNotExist):
        return render(request, 'polls/detail.html', {
            'selected_question': question,
            'error_message': '아무것도 선택하지 않았어요!'
        })
    else:
        # 해당 객체의 votes 값 1 올리기
        selected_choice.votes += 1
        # save() : 이를 토대로 값을 변경해주세여
        selected_choice.save()
        # reverse(): urls.py(URLConf)에 있는 name 을 이용해서 url 형식으로 변환
        return HttpResponseRedirect(reverse('polls:results',
                                            # tuple 에 인자 하나라서 , 붙임
                                            args=(question.id,)))

def results(request, question_id):
    question = get_object_or_404(Question, pk=question_id)

    return render(request, 'polls/results.html', {
        'question': question
    })
```



#### results.html

```django
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
</head>
<body>
    <h1>{{ question.question_text }}</h1>
    <ul>
        {% for choice in question.choice_set.all %}
            <li>{{ choice.choice_text }} - {{ choice.votes }}</li>
        {% endfor %}
    </ul>
    <a href="{% url 'polls:detail' question.id %}">다시 투표하기!</a>
</body>
</html>
```



### 최종

#### settings.py

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
SECRET_KEY = '$hsr4y$2cx)ba!5z8=(^hhe^i5benjrkci-#8fys%h=)^2!3&y'

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
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
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



#### mysite\urls.py

```python
"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('polls/', include('polls.urls'))
]

```



#### models.py

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



#### admin.py

```python
from django.contrib import admin
from polls.models import Question, Choice

# Register your models here.
admin.site.register(Question)
admin.site.register(Choice)
```



#### polls\urls.py

```python
from django.urls import path
from . import views

app_name = 'polls'
urlpatterns = [
    path('', views.index, name='index'),
    path('<int:question_id>/', views.detail, name='detail'),
    path('<int:question_id>/vote/', views.vote, name='vote'),
    path('<int:question_id>/results/', views.results, name='results')
]
```



#### views.py

```python
from django.shortcuts import render, get_object_or_404
from polls.models import Question, Choice
from django.http import HttpResponseRedirect
from django.urls import reverse
# Create your views here.
def index(request):
    question_list = Question.objects.all().order_by('-pub_date')[:5]
    context = {'q_list': question_list}
    return render(request, 'polls/index.html', context)

def detail(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    context = {'selected_question': question}
    return render(request, 'polls/detail.html', context)

def vote(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    try:
        selected_choice = question.choice_set.get(
            pk=request.POST['my_choice']
        )
    except(KeyError, Choice.DoesNotExist):
        return render(request, 'polls/detail.html', {
            'selected_question': question,
            'error_message': '아무것도 선택하지 않았어요!'
        })
    else:
        selected_choice.votes += 1
        selected_choice.save()
        return HttpResponseRedirect(reverse('polls:results',
                                            args=(question.id,)))

def results(request, question_id):
    question = get_object_or_404(Question, pk=question_id)

    return render(request, 'polls/results.html', {
        'question': question
    })
```



#### polls\templates

##### index.html

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
                <li><a href="/polls/{{ question.id }}">
                    {{ question.question_text }}
                </a></li>
            {% endfor %}
        </ul>
    {% else %}
        <h1>데이터가 없어요!</h1>
    {% endif %}
</body>
</html>
```



##### detail.html

```django
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
</head>
<body>
    <h1>{{ selected_question.question_text }}</h1>
    {% if error_message %}
        <div>{{ error_message }}</div>
    {% endif %}

    <form action="{% url 'polls:vote' selected_question.id %}"
          method="POST">
    {% csrf_token %}
        {% for choice in selected_question.choice_set.all %}
            <input type="radio"
                   name="my_choice"
                   id="id_{{ forloop.counter }}"
                   value="{{ choice.id }}">
            <label for="id_{{ forloop.counter }}">
                {{ choice.choice_text }}
            </label><br>
        {% endfor %}<br>
        <input type="submit" value="투표하기">
    </form>
</body>
</html>
```



##### results.html

```django
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
</head>
<body>
    <h1>{{ question.question_text }}</h1>
    <ul>
        {% for choice in question.choice_set.all %}
            <li>
                {{ choice.choice_text }} - {{ choice.votes }}
            </li>
        {% endfor %}
    </ul>
    <a href="{% url 'polls:detail' question.id %}">
        다시 투표하기
    </a>
</body>
</html>
```