# day13_web_TIL



#### Network

- 컴퓨터 안에 있는 프로그램끼리 통신할 수 있는 망



#### Internet

- network of network: 네트워크와 네트워크를 연결하는 큰 네트워크

- 물리적인 Framework: 전세계적으로 구축된 네트워크 망
- ex) HTTP web service, SMTP Email service, FTP service



#### 복습

- web client(web client program)

  ex) Browser(chrome), 따로 구현 가능

- web server(web server program)

  ex) Apache, IIS, oracle web server, 따로 구현 x

  정적인 contents를 service => web server가 처리

  동적인 contents를 만들어서 service 할 수 없다 => WAS에 위임

  - 동적 contents: 서버에서 별로의 web program(server-side web application)을 실행시켜 나온 결과

- WAS (web application server)

  server-side web application을 보유하고 있음, 이를 생성하는 python Framework이 Django



#### Framework

- library

  특정 기능을 수행하는 코드 묶음

  ex) python의 module

- Framework

  전체적으로 시스템의 기초가 구성되어 있음, 개발에 용이

  기초 이해도가 있어야 개발 가능 

- platform

  다른 프로그램을 실행시킬 수 있는 환경이 되는 프로그램

  ex) window10, OS



#### 정리

- web client

  -> web server로 request

  ​	web server에게 response받은 결과물을 lendering

- web server

  -> 정적 contents는 바로 response

  ​	동적 contents는 WAS로 request

  ​	WAS에게 response 받으면 이를 다시 web client로 response

- WAS(web application server)

  -> server-side web application을 통해 결과를 web server로 response

  ​	이 때 필요한 경우 Database를 통해 데이터를 가져옴

- Database



#### response

- server쪽 프로그램을 실행시켜 도출되는 결과물

  ex) HTML+CSS(일반적인 형태), 동영상, 이미지, JSON



#### client-side web application

- CASE 1. Round Trip 방식

  response할 때 결과 값과 HTML+CSS 코드를 합쳐서 response

  server-side web application이 코드 묶음을 response 형태로 web client에 제공

  web server에 request 과정 없이 web client 내부에서 실행되는 프로그램

  - 단점: server쪽에서 모든 프로그램을 구현하다보니 데이터 전달량이 너무 방대해짐



- CASE 2. AJAX

  server-side web application과 client-side web application을 분리 구현

  web client는 미리 다른 server에서 HTML+CSS+Javascript로 이루어진 server-side web application를 받음

  이후 request에는 WAS에서 결과데이터를 HTML이 아닌 용량이 적은 CSV, XML, JSON 형식으로 response



- FrontEnd 개발: client-side web application 개발 작업(HTML, Javascript 사용)

  BackEnd 개발: server-side web application 개발 작업(Python, Java 사용)



- 패턴: FrontEnd 개발 Framework

  패턴에 입각해 개발하면 유지보수에 용이하다.

  ex) Angular, React, ve,js

  

### CASE 2 실습

- FrontEnd web application 구현

- HTML: 화면에 내용을 표시

  CSS: 화면을 디자인

  Javascript: 화면에 코드처리

  

### Javascript

- 변수선언(파이썬과 비슷)

  var, let = 변수를 선언할 때 쓰는 예약어

- 한 문장이 끝나면 ;을 찍는다

  ; : 여기에서 이 문장이 종료됐다는 지시어

```javascript
// JavaScript
// 1. 변수선언(파이썬과 비슷)
//  var = 옛버전, 변수 선언할 때 쓰는 예약어(파이썬 def와 비슷)
//  let = 버전업에서 변수 선언할 때 쓰는 예약어
// 한 문장이 끝나면 ;을 찍는다
// ; : 여기에서 이 문장이 종료됐다는 지시어
let tmp1 = 'sample';    // string
let tmp2 = 3.14;        // number
let tmp3 = true;        // boolean, 소문자 true, false
let tmp4 = [1, 2, 3, 4];// array(파이썬 list와 비슷)
```



- 실행환경: browser 내에서 실행된다.

- 변수 출력

  alert(), console.log() 이용

  - alert():

    경고창을 호출하는 함수

    blocking method로 이 문단을 읽을 경우 코드의 실행이 일시 정지되고 확인을 눌러야 다음으로 넘어감

  - console.log():

    f12를 누른 후 console을 누르면 console.log를 볼 수 있다.

- javascript 객체

  파이썬 dict와 같은 구조, 표현 방식은 JSON으로

- 함수 만들기

  function name { }이용 

```javascript
// 실행환경: browser 내에서 실행된다
// 변수를 출력하고 싶어요
alert(), console.log()

// alert()는 blocking method: 여기에서 코드의 실행이 일시 중지
//  확인을 눌러야 다음을로 넘어감
alert(tmp1)

// f12를 누른 후 console을 누르면 console.log를 볼 수 있다
console.log('변수의 값 : ' + tmp4)

// javascript 객체 (파이썬의 dict와 같은 구조 => 데이터 표현방식은 JSON으로)
let obj = { name : '홍길동',
            age : 25 }
console.log(obj.name, obj.age)

// 함수 만들기
function add(x,y) {
    return x + y
}
alert(add(10,20))
```





### jQuery

- jQuery를 사용할 수 있도록 불러들이는 코드, title 밑에 삽입

```html
	<script
            src="https://code.jquery.com/jquery-2.2.4.min.js"
            integrity="sha256-BbhdlvQf/xTY9gja0Dq3HiwQF8LaCRTXxZKRutelT44="
            crossorigin="anonymous">
    </script>
```



#### html

- element: HTML 구성요소

  시작 태그와 끝 태그 영역을 모두 포함

- tag: <>fh 구성되는 HTML 요소 (<> 부분만 지칭)

- element의 종류는 크게 2 가지

  block level element: element가 한 라인 전체를 차지

  inline element: element가 해당 내용만 영역을 차지



- 같은 단계에 있으면 형제 관계, 포함관계면 부모자식 관계

  부모의 부모와는 관계가 없음



- ul: unorderd list

  li의 값들이 ● 기호로 화면에 출력됨

- ol: ordered list

  li의 값드이 숫자로 화면에 출력됨



- 영역을 잡는 태그: div, span

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
    <script
            src="https://code.jquery.com/jquery-2.2.4.min.js"
            integrity="sha256-BbhdlvQf/xTY9gja0Dq3HiwQF8LaCRTXxZKRutelT44="
            crossorigin="anonymous">
    </script>
    <script src="js/jQuery_exec.js"></script>
</head>
<body>
    <!-- element: HTML 구성 요소(시작 태그와 끝 태그 영역 모두 포함하는 영역) -->
    <!-- tag: <>로 구성되는 HTML 요소(<h1> 부분만 지칭) -->
    <!-- element의 종류는 크게 2가지
         block level element: element가 한 라인 전체를 차지
         inline element (span, img): element가 해당 내용만 영역을 차지 -->

    <!-- 같은 단계에 있으면 형제 관계, 포함관계면 부모자식 관계 -->
    <!-- 부모의 부모와는 관계가 없다 -->
    <h1>여기는 h1입니다.</h1>
    <ul> <!-- unordered list -->
        <!--  li 값들이 ● 기호로 화면에 출력됨  -->
        <li class="region">서울</li>
        <li id="inchon">인천</li>
        <li class="region">부산</li>
    </ul>
    <ol> <!-- ordered list -->
        <!-- li 값들이 숫자로 화면에 출력됨 -->
        <li>아이유</li>
        <li>김연아</li>
        <li>홍길동</li>
    </ol>
    <!-- 영역잡는 태크: div, span -->
    <div class="region">이것은 소리없는 아우성!!</div>
    <span class="region">여기는 span 영역입니다.</span>
    <img src="img/car.jpg">
    <br><br>
    <input type="button" value="클릭클릭!!"
           onclick="my_func()">
</body>
</html>
```



#### jQuery

1. selector

   element를 지칭하는 특수한 표기법

   문자열을 이용

   jQuery는 $ 기호로 시작한다.

   1. 전체 선택자: *

   2. 태그 선택자: 태그명을 이용해서 선택

   3. 아이디 선택자: ID 속성을 이용해서 선택

      ID 앞엔 # 붙여야 함

   4. 클래스 선택자: class 속성을 이용해서 선택 (파이썬 class와 연관 없음)

      ID는 유니크한 값, class는 그룹을 지정해서 css 적용 가능

      클래스를 지정하는 기호 . (dot)을 붙여야 함

   5. 구조 선택자: 부모, 자식, 형제 관계를 이용해서 선택

      '>'는 (부모 > 자식)의 태그 관계를 표현

- 함수

  1. css():

     jQuery가 가진 함수 중 스타일을 지정하는 함수

     background-color: 배경에 색깔 삽입

  2. remove():

     요소를 삭제하는 함수

  3. text():

     문자열을 변경하는 함수

     

```javascript
function my_func() {
    // 버튼을 누르면 호출되는 my_func함수
    // jQuery 사용법
    // 1. selector
    //     selector는 HTML element를 지칭하는 특수한 표기법을 의미
    //     selector는 문자열을 이용한다 (' ') 필요
    //     jQuery는 $로 시작한다
    //     $('selector').method() - selector 찾아서 method 적용
    //     1. 전체 선택자: *
               $('*').css('color', 'red')
    //         css(): jQuery가 가진 함수 중 스타일을 지정하는 함수
    //     2. 태그 선택자: 태그명을 이용해서 선택
               $('span').remove()
    //         remove(): 요소를 삭제하는 함수
               $('li').css('background-color', 'yellow')
    //         background-color: 배경에 색깔 삽입
    //     3. 아이디 선택자: ID속성을 이용해서 선택
               $('#inchon').text('소리없는 아우성!!')
    //         id 앞엔 # 붙여야 함
    //     4. 클래스 선택자: class 속성을 이용해서 선택 (파이썬 class와 연관 없음)
    //         id는 유니크한 값, class는 그룹을 지정해서 css 적용 가능
               $('.region').css('color', 'blue')
    //         클래스를 지칭하는 기호 . (dot)
    //     5. 구조 선택자: 부모, 자식, 형제 관계를 이용해서 선택
               $('ol > li').css('background-color', 'yellow')
    //         > 는 (부모 > 자식)의 태그 관계를 표현


}
```







### boxoffice 실습

- html

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
    <!-- ctrl + / = HTML 주석 처리 -->
    <!-- jQuery를 이용하기 위해서 CDN 방식을 이용할 거에요!-->
    <script
            src="https://code.jquery.com/jquery-2.2.4.min.js"
            integrity="sha256-BbhdlvQf/xTY9gja0Dq3HiwQF8LaCRTXxZKRutelT44="
            crossorigin="anonymous">
    </script>

    <!-- src javascript는 my_script.js 파일 그대로 사용 -->
    <script src="js/my_script.js">
    </script>
</head>
<body>
일일 박스오피스 순위를 알아보아요!!
<br><br>
<!-- 입력 상자를 생성하는 함수 input -->
key : <input type="text" id="userKey">
<br><br>
날짜 : <input type="text" id="userDate">
<br><br>
<input type="button" value="조회!!"
        onclick="hello()">
</body>
</html>
```



- javascript

```javascript
function hello() {
    alert('버튼이 클릭되었어요!')
    // $ Jquery를 시작할 때 사용하는 기호
    // # Id를 지칭하는 기호
    // val() 사용자가 입력한 value를 호출하는 함수
    user_key = $('#userKey').val()
    // 682ca358e95538a21f1702baa2dd****(마스킹처리)

    user_date = $('#userDate').val()
    open_api = 'http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json'

    // api query = open_api URL + ?key=key값&targetDt=targetDt값
    my_url = open_api + '?key=' + user_key + '&targetDt=' + user_date

    // location: 현재 웹 브라우저 페이지를 의미
    // .href: 현재 웹 브라우저를 의미, 이를 변경하면 웹페이지가 변경됨
    location.href = my_url
}

```

