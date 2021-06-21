# Project01_WebProject



### 기본 정보

- 서비스명: i Pet (Infrastructure for Pet)
- 팀명: 반짝단짝 (반려동물의 단짝이 되자)
- 주제: 반려인들을 위한 서울시 반려동물 인프라 검색 및 커뮤니티



### Themes

- 담당부분
- 상세 부분은 압축파일 참조



##### urls.py

```python
from django.urls import path
from . import views

app_name = 'themes'

urlpatterns = [
    path('', views.themes, name='themes'),
    path('dataInsert/', views.data_insert, name='data_insert'),
    path('hotel/', views.hotel_list, name='hotel_list'),
    path('training/', views.training_list, name='training_list'),
    path('cafe/', views.cafe_list, name='cafe_list'),
    path('<int:local_id>/', views.t_object, name='t_object')
]

```



##### models.py

```python
from django.db import models

class Local(models.Model):
    title = models.CharField('이름', max_length=200)
    venue = models.CharField('주소', max_length=200)
    category = models.CharField('카테고리', max_length=200)
    tel = models.CharField('전화번호', max_length=200)
    img_url = models.URLField('썸네일 URL', max_length=400, blank=True)

    def __str__(self):
        return self.title

```



##### views.py

```python
from django.shortcuts import render, redirect, get_object_or_404
from selenium import webdriver
import time
from themes.models import Local
import json


def themes(request):
    return render(request, 'themes/themes.html')


def data_insert(request):


    #1. 웹드라이버 켜기
    driver = webdriver.Chrome("./chromedriver")
    #2. 네이버 지도 접속하기
    driver.get("https://v4.map.naver.com/")

    # !!)네이버 지도 업데이트로 추
    driver.find_elements_by_css_selector("button.btn_close")[1].click()

    #3. 검색창에 검색어 입력하기 // 검색창: input#search-input
    search_box = driver.find_element_by_css_selector("input#search-input")
    # 검색어 - 서울 반려동물, 서울 반려견 동반
    search_box.send_keys("서울 반려견 동반")
    #4. 검색버튼 누르기 // 검색버튼: button.spm
    search_button = driver.find_element_by_css_selector("button.spm")
    search_button.click()
    #5. 검색 결과 확인하기

    for n in range(1, 31):
        # 지연시간주기
        time.sleep(1)
        l_result = []
        stores = driver.find_elements_by_css_selector("div.lsnx")
        for s in stores:
            my_img = s.find_element_by_css_selector('div > img')
            img_url = my_img.get_attribute('src')
            title = s.find_element_by_css_selector("dl.lsnx_det > dt > a").text
            venue = s.find_element_by_css_selector("dl.lsnx_det > dd.addr").text
            category = s.find_element_by_css_selector("dl.lsnx_det > dd.cate").text

            try:
                tel = s.find_element_by_css_selector("dl.lsnx_det > dd.tel").text
            except:
                tel = "전화번호 없음"

            local_obj = {
                'title': title,
                'venue': venue,
                'category': category,
                'tel': tel,
                'img_url': img_url
            }
            l_result.append(local_obj)

        for l in l_result:
            locals = Local(
                title=l['title'],
                venue=l['venue'],
                category=l['category'],
                tel=l['tel'],
                img_url=l['img_url']
            )
            locals.save()

        # 페이지버튼 div.paginate > *
        page_bar = driver.find_elements_by_css_selector("div.paginate > *")

        try:
            if n%5 != 0:
                page_bar[n%5+1].click()
            else:
                page_bar[6].click()
        except:
            print("수집완료")
            break
    print('성공했어용')
    return redirect('themes:themes')


# 테마 종류 "반려동물 > 반려동물호텔", "반려동물 > 애견훈련"
# 음식점 > 카페,디저트, 음식점 > 카페
def hotel_list(request):
    locals = Local.objects.all()
    hotels = locals.filter(category="반려동물 > 반려동물호텔")
    context = {'hotels': hotels}
    return render(request, 'themes/hotel.html', context)


def training_list(request):
    locals = Local.objects.all()
    training = locals.filter(category="반려동물 > 애견훈련")
    context = {'training': training}
    return render(request, 'themes/training.html', context)


def cafe_list(request):
    locals = Local.objects.all()
    cafes = locals.filter(category="음식점 > 카페,디저트")
    context = {'cafes': cafes}
    return render(request, 'themes/cafe.html', context)


def t_object(request, local_id):
    local = get_object_or_404(Local, pk=local_id)

    mapdict = {'title': local.title,
               'venue': local.venue,
               'category': local.category,
               'tel': local.tel}

    map_json = json.dumps(mapdict)
    return render(request, 'themes/t_object.html', {
        'selected_local': local,
        'map_json': map_json
    })

```



##### themes.html

```html
{% extends 'base.html' %}

{% block html_header %}
    <script src="/static/js/themes.js"></script>
{% endblock %}

{% block html_body %}
{# block 부분을 치환해서 사용할 거에요 #}

    <div class="container">
        <h1>최초 1회만 실행</h1>
        <form action="/themes/dataInsert/" method="post">
            {% csrf_token %}
            <button class="w-100 btn btn-lg btn-primary" type="submit">데이터 가져오기</button>
        </form>
        <br>
        <div>
            <h5>모델 속 데이터를 json으로 파싱</h5>
            <a href="/themes/mapJson/">json paring!</a>
        </div>

    </div>
{% endblock %}
```



##### cafe.html

```html
{% extends 'base.html' %}

{% block html_header %}

    <style>

        img { display: block; margin: 0px auto; }

        header {
            background-image: url("/static/img/cafe_bg.jpg");
            background-size: 100%;
            background-position: 100%;
         }


    footer{
        position: relative;
    }
    </style>

{% endblock %}

{% block html_body %}

<!-- Header -->
  <header id="my_banner" class="bg-primary py-5 mb-5" >
    <div class="container h-100">
      <div class="row h-100 align-items-center">
        <div class="col-lg-12">
          <h2 id="my_infra" class="display-4 text-white text-align-center mt-5 mb-2">애견 동반 카페</h2>
          <p class="lead mb-5 text-white-50"></p>
        </div>
      </div>
    </div>
  </header>

  <body>

<main>

  <div class="album py-5">
    <div class="container">

      <div class="row row-cols-1 row-cols-sm-2 row-cols-md-3 g-3">

      {% for cafe in cafes %}

      <div class="col-lg-4 col-md-6 mb-5">
        <div class="card h-100">
          <img src="{{ cafe.img_url }}"
                   width="150"
                   height="150"
                   alt="{{ cafe.title }}">
          <div class="card-body">
            <h4 class="card-title">{{ cafe.title }}</h4>
            <p class="card-text">
                <ul>
                    <li>{{ cafe.category }}</li>
                    <li>{{ cafe.tel }}</li>
                    <li>{{ cafe.venue }}</li>
                </ul>
              </p>
          </div>
          <div class="card-footer">
            <a href="/themes/{{ cafe.id }}" class="btn btn-info">View</a>
          </div>

        </div>
      </div>

        {% endfor %}

      </div>
    </div>
  </div>

</main>

<footer class="text-muted py-5">
  <div class="container">
    <p class="float-end mb-1">
      <a href="#">Back to top</a>
  </div>
</footer>
  </body>
{% endblock %}


```



##### t_object.html

```html
{% extends 'base.html' %}

{% block html_header %}


    <!-- Custom styles for this template -->
    <link href="/static/css/product.css" rel="stylesheet">

    <script type="text/javascript" src="//dapi.kakao.com/v2/maps/sdk.js?appkey=d90250dcf255f14a97c9a9ce7407bf65&libraries=services"></script>

    <style>

        header {
            background-image: url("/static/img/obj_bg.jpg");
            background-size: 100%;
            background-position: 100%;
         }

    </style>

{% endblock %}

{% block html_body %}

<!-- Header -->
  <header id="my_banner" class="bg-primary py-5 mb-5" >
    <div class="container h-100">
      <div class="row h-100 align-items-center">
        <div class="col-lg-12">
          <h2 id="my_infra" class="display-4 text-white text-align-center mt-5 mb-2">{{ selected_local.title }}</h2>
          <p class="lead mb-5 text-white-50"></p>
        </div>
      </div>
    </div>
  </header>

  <body>

<main>

<div class="container">
  <div class="d-md-flex flex-md-equal w-100 my-md-3 ps-md-3">
    <div class="bg-light pt-md-5 pb-md-5" style="border-radius: 21px 0px 0px 21px;">
        <div class="my-3 py-3 text-center">
            <h2 class="display-5">{{ selected_local.title }}</h2>
            <p class="lead text-muted">
                <img src="{{ selected_local.img_url }}"
                       width="200"
                       height="200"
                       alt="{{ cafe.title }}">
            </p>
        </div>

    </div>
        <div class="bg-light pt-md-5 pb-md-5 ps-md-5 pe-md-5" style="border-radius: 0px 21px 21px 0px;">
          <div id="map" style="width:100%;height:350px;"></div>
        </div>
    </div>
    <div class="shadow-sm mx-auto" style="width: 80%; height: 100%; border-radius: 21px 21px 21px 21px;">
        <li>　　　　● 카테고리  　:　 {{ selected_local.category }}</li>
        <li>　　　　● 전화번호  　:　 {{ selected_local.tel }}</li>
        <li>　　　　● 주소　　  　:　 {{ selected_local.venue }}</li>
    </div>

</div>

<p style="margin-top:-12px"></p>
<script>
// json 형식으로 가져오기 = dictionary 형태
var parsedmap = JSON.parse("{{ map_json|escapejs }}");
console.log(parsedmap)

// 주소-좌표 변환 객체를 생성합니다
var geocoder = new kakao.maps.services.Geocoder();

// 주소로 좌표를 검색합니다
geocoder.addressSearch(parsedmap['venue'], function(result, status) {

    // 정상적으로 검색이 완료됐으면
     if (status === kakao.maps.services.Status.OK) {

        var coords = new kakao.maps.LatLng(result[0].y, result[0].x);

        var mapContainer = document.getElementById('map'), // 지도를 표시할 div
        mapOption = {
            center: new kakao.maps.LatLng(result[0].y, result[0].x), // 지도의 중심좌표
            level: 3 // 지도의 확대 레벨
        };

        // 지도를 생성합니다
        var map = new kakao.maps.Map(mapContainer, mapOption);

        // 결과값으로 받은 위치를 마커로 표시합니다
        var marker = new kakao.maps.Marker({
            map: map,
            position: coords
        });

        // 인포윈도우로 장소에 대한 설명을 표시합니다
        var infowindow = new kakao.maps.InfoWindow({
            content:'<div style="width:150px;text-align:center;padding:6px 0;">' + '{{selected_local.title}}' + '</div>'
        });
        infowindow.open(map, marker);

        // 지도의 중심을 결과값으로 받은 위치로 이동시킵니다
        map.setCenter(coords);
    }
});
</script>
<br>
<br>

<div class="container">
    <p class="float-end mb-1">
        <input class="btn btn-info" type="button" value="BACK" onClick="history.go(-1)">
    </p>
</div>

</main>
  </body>
{% endblock %}
```



##### base.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>i Pet</title>
    
    <!-- jQuery CDN-->
    <script src="https://code.jquery.com/jquery-2.2.4.min.js" integrity="sha256-BbhdlvQf/xTY9gja0Dq3HiwQF8LaCRTXxZKRutelT44=" crossorigin="anonymous"></script>

    <!-- Bootstrap CDN-->
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">

    <!-- CSS 정보-->
    <link href="/static/css/navbar.css" rel="stylesheet" type="text/css"/>
	<script src="https://code.jquery.com/jquery-1.12.3.js" type="text/javascript" ></script>
    <script src="/static/js/navbar.js" type="text/javascript" defer></script>

    <!-- Custom styles for this template -->
    <link href="/static/css/heroic-features.css" rel="stylesheet">

    <style>
        html{ position: relative;
            min-height: 150%;
            margin: 0;
            background: white;
        }

        body {
            min-height: 150%;
            }

        footer {
            position: absolute;
            left: 0;
            bottom: 0;
            width: 100%;
            padding: 15px 0;
        }

    /* 폰트 불러오기 */
    @font-face {src: url("/static/font/AppleSDGothicNeoH.ttf"); font-family: AppleSDGothicNeoH;}
    @font-face {src: url("/static/font/AppleSDGothicNeoL.ttf"); font-family: AppleSDGothicNeoL;}
    @font-face {src: url("/static/font/AppleSDGothicNeoM.ttf"); font-family: AppleSDGothicNeoM;}
    @font-face {src: url("/static/font/AppleSDGothicNeoEB.ttf"); font-family: AppleSDGothicNeoEB;}
    @font-face {src: url("/static/font/AppleSDGothicNeoEB.ttf"); font-family: AppleSDGothicNeoB;}

    #my_infra{font-family: AppleSDGothicNeoB;}
    h4{font-family: AppleSDGothicNeoH;}
    p{font-family: AppleSDGothicNeoEB;}
    a{font-family: AppleSDGothicNeoM;}

    </style>
    
    <!-- 추가적인 CSS 정보 -->
    {% block html_header %}

    {% endblock %}
</head>

<body>

    {% include 'navbar.html' %}

    {% block html_body %}
    {% endblock %}

    {% include 'foot.html' %}

</body>
</html>
```

