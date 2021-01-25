# day14_web_TIL



## webstorm



#### jQuery

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
    //         $('*').css('color', 'red')
    //         css(): jQuery가 가진 함수 중 스타일을 지정하는 함수
    //     2. 태그 선택자: 태그명을 이용해서 선택
    //         $('span').remove()
    //         remove(): 요소를 삭제하는 함수
    //         $('li').css('background-color', 'yellow')
    //         background-color: 배경에 색깔 삽입
    //     3. 아이디 선택자: ID속성을 이용해서 선택
    //         $('#inchon').text('소리없는 아우성!!')
    //         id 앞엔 # 붙여야 함
    //     4. 클래스 선택자: class 속성을 이용해서 선택 (파이썬 class와 연관 없음)
    //         id는 유니크한 값, class는 그룹을 지정해서 css 적용 가능
    //         $('.region').css('color', 'blue')
    //         클래스를 지칭하는 기호 . (dot)
    //     5. 구조 선택자: 부모, 자식, 형제 관계를 이용해서 선택
    //         $('ol > li').css('background-color', 'yellow')
    //         > 는 (부모 > 자식)의 태그 관계를 표현
    //         (공백) : 후손, + : 바로 다음에 나오는 형제, ~ : 뒤에 나오는 모든 형제
    //         $('ol ~ span').css('color', 'red')
    //     6. 속성 선택자
    //         $('input[type]') => input tag에 type이라는 속성이 있는 element
    //         $('input[type=button]').disable() - 버튼을 비활성화 시키는 것
}
```



#### jQuery_sample01

- html

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
    <script src="js/jQuery_sample01.js"></script>
</head>
<body>
    <div>
        <ul>
            <li id="apple">사과</li>
            <li id="pineapple">파인애플</li>
            <li class="myList">참외</li>
        </ul>
        <!-- 사용자 입력 양식(form) : 사용자로부터 데이터를 입력받아서 서버로 전송 -->
        <!-- action : 서버쪽 프로그램, method : 어떤 방식으로 호출할 것인가 -->
        <form action="#" method="post">
            <input type="text" id="uId" size="20">
        </form>
        <ol>
            <li class="myList">고양이</li>
            <li class="myList">호랑이</li>
            <li class="myList">강아지</li>
        </ol>
        <input type="button" value="click"
               onclick="my_func()"
    </div>
</body>
</html>
```



- jQuery



- text() : 태그 사이 글자를 가져오는 함수
- val() : 입력 상자 안에 있는 값을 가져오는 함수
- attr() : 속성 값을 가져오는 함수
- eq() : 순서를 지정해서 가져오는 함수

```javascript
function my_func() {
    // text() : 태크 사이의 글자를 가져오는 함수
    console.log( $('#apple').text() )
    console.log( $('ul').text() )
    console.log( $('#pineapple').text() )
    console.log( $('ul > .myList').text() )

    // val() : 입력 상자 안에 있는 값을 가져오는 함수
    console.log( $('#uId').val() )

    // attr() : 속성 값을 가져오는 함수
    console.log( $('input[type=text]').attr('id') )
    $('input[type=text]').attr('size', 10) // 속성값을 변경
    console.log( $('ol > li:first').text() )
    console.log( $('ol > li:last').text() )
    console.log( $('ol > li:first + li').text() )

    // eq() : 순서 지정해서 가져오는 함수
    console.log( $('ol > li:eq(1)').text() )
    console.log( $('ol').text() )

    $('ol > li').each(function(idx, item) {
        console.log( (idx+1) +'번째 ' + $(item).text() + '입니다.' )
    })


}

// function() : 이름이 없는 함수 => 묵시적 함수 => lambda 함수
// 함수를 독자적으로 사용은 불가능하고 변수 같은 곳에 저장해서 사용
// 함수를 하나의 값으로 인식 => first class
// 함수가 하나의 값으로 사용되기 때문에 함수를 다른 함수의 인자로 사용 가능
// let kaka = function () { }
```



#### jQuery_sample02

- html

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
    <script src="js/jQuery_sample02.js"></script>
    <style>
        .myStyle {
            width: 300px;
            height: 100px;
            background-color: yellow;
        }
    </style>
</head>
<body>
    <div>이것은 소리없는 아우성!!</div>
    <ul>
        <li>김연아</li>
        <li>이순신</li>
        <li>강감찬</li>
    </ul>

    <input type="button" value="click"
           onclick="my_func()">

</body>
</html>


<!--    <div class="myStyle">-->
<!--        <ol>-->
<!--            <li>홍길동</li>-->
<!--            <li>김길동</li>-->
<!--        </ol>-->
<!--    </div>-->

<!--    <input type="button" value="can i click?"-->
<!--           disabled="disabled">-->

<!--    <input type="button" value="del style"-->
<!--           onclick="remove_func()">-->

```



- jQuery



- addClass() : class 추가
- removeAttr() : 속성 지우는 함수
- remove() : 지우는 함수
- empty() : 자신은 삭제하지 않고 자신의 후손을 모두 삭제
- append(), prepend(), after(), before()

```javascript
function remove_func() {
    // removeClass : class를 제거
    $('div').removeClass('myStyle')
}

function my_func() {
    $('div').css('color', 'red')
    $('div').css('background-color', 'yellow')
    
    // addClass : class 추가
    $('div').addClass('myStyle')
    // removeAttr() : 속성 지우는 함수
    $('input[type=button]:first').removeAttr('disabled')
    $('div.myStyle').remove()
    
    // 자신은 삭제하지 말고 자신의 후손을 모두 삭제
    $('div.myStyle').empty()
    
    // 그럼 없는 element를 만들려면 어떻게 해야 하나요?

    // 새로운 tag 생성하기
    // <div>소리없는 아우성</div>
    let my_div = $('<div></div>').text('소리없는 아우성')

    // <img src='img'/car.jpg>
    let my_img = $('<img />').attr('src','img/car.jpg')

    // 새로운 element를 원하는 위치에 붙여야 함
    let my_li = $('<li></li>').text('아이유')
    
    // 4종류의 함수로 element를 원하는 위치에 가져다 붙일 수 있다
    // 1. append() : 맨 마지막 자식으로 붙임
          $('ul').append(my_li)
    // 2. prepend() : 맨 처음 자식으로 붙임
          $('ul').prepend(my_li)
    // 3. after() : 바로 다음 형제로 붙임
          $('ul > li:eq(1)').after(my_li)
    // 4. before() : 바로 이전 형제로 붙임
          $('ul > li:last').before(my_li)

}
```



### dailyBoxOfficeSearch

- Bootstrap sample의 html 코드를 가져와 실습 진행

- html

```html

<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>dailyBoxOfficeSearch</title>

    <script
            src="https://code.jquery.com/jquery-2.2.4.min.js"
            integrity="sha256-BbhdlvQf/xTY9gja0Dq3HiwQF8LaCRTXxZKRutelT44="
            crossorigin="anonymous">
    </script>


    <!-- Bootstrap core CSS -->
    <!-- Bootstrap도 CDN 방식으로 이용 -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-giJF6kkoqNQ00vy+HMDP7azOuL0xtbfIcaT9wjKHr8RbDVddVHyTfAAsrekwKmP1" crossorigin="anonymous">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta1/dist/js/bootstrap.bundle.min.js" integrity="sha384-ygbV9kiqUc6oa4msXn9868pTtWMgiQaeYH7/t7LECLbyPA2x65Kgf80OJFdroafW" crossorigin="anonymous"></script>

    <style>
        .bd-placeholder-img {
            font-size: 1.125rem;
            text-anchor: middle;
            -webkit-user-select: none;
            -moz-user-select: none;
            user-select: none;
        }

        @media (min-width: 768px) {
            .bd-placeholder-img-lg {
                font-size: 3.5rem;
            }
        }
    </style>


    <!-- Custom styles for this template -->
    <link href="css/dashboard.css" rel="stylesheet">

    <script src="js/dailyBoxOfficeSearch.js"></script>

</head>

<body>

<header class="navbar navbar-dark sticky-top bg-dark flex-md-nowrap p-0 shadow">
    <a class="navbar-brand col-md-3 col-lg-2 me-0 px-3" href="#">BoxOffice</a>
    <button class="navbar-toggler position-absolute d-md-none collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#sidebarMenu" aria-controls="sidebarMenu" aria-expanded="false" aria-label="Toggle navigation">
        <span class="navbar-toggler-icon"></span>
    </button>
    <!-- search 검색창, placeholder : 처음 뜨는 힌트 문구 -->
    <input class="form-control form-control-dark w-100"
           type="text" placeholder="날짜를 입력하세요(yyyymmdd)"
           id="userInputDate" aria-label="Search">

    <ul class="navbar-nav px-3">
        <li class="nav-item text-nowrap">
            <a class="nav-link" href="#"
                onclick="my_func()">Search</a>
        </li>

    </ul>
</header>

<div class="container-fluid">
    <div class="row">
        <nav id="sidebarMenu" class="col-md-3 col-lg-2 d-md-block bg-light sidebar collapse">
            <div class="position-sticky pt-3">
                <ul class="nav flex-column">
                    <li class="nav-item">
                        <a class="nav-link active" aria-current="page" href="#">
                            <span data-feather="home"></span>
                            순위 조회
                        </a>
                    </li>
                </ul>
            </div>
        </nav>

        <main class="col-md-9 ms-sm-auto col-lg-10 px-md-4">

            <!-- <a href="http://www.naver.com">네이버로 이동</a>-->
            <!-- a element는 hyperlink를 만들어주는 element
                href 속성 다음에 있는 URL로 GET방식의 request 전송 -->

            <h2>DailyBoxOffice Search Rank</h2>
            <div class="table-responsive">
                <table class="table table-striped table-sm">
                    <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Name</th>
                        <th>누적관객수</th>
                        <th>누적매출액</th>
                        <th>개봉일</th>
                    </tr>
                    </thead>
                    <tbody>
                    <tr>
                        <td>1,001</td>
                        <td>Lorem</td>
                        <td>ipsum</td>
                        <td>dolor</td>
                        <td>sit</td>
                    </tr>
                    </tbody>
                </table>
            </div>
        </main>
    </div>
</div>

</body>
</html>

```



- jQuery



- AJAX : 화면이 refresh 되지 않도록 해주는 데이터 통신 방식

  -     $.ajax({
            url : open_api,     	 	// 호출할 서버쪽 프로그램의 URL, Query string 제외
            type : 'GET',       			// 서버쪽 프로그램에 대한 request 방식
            dataType : "json",    	// 서버 프로그램이 결과로 보내주는 데이터 형식
            data : {            			   // 서버 프로그램 호출할 때 넘어가는 데이터
                key : user_key,
                targetDt : user_date
            },
            success : function () {
                alert('success')
            },
            error : function () {
                alert('error!')
            }

```javascript
function my_func() {
    // 사용자가 입력한 날짜를 가져와서
    // 해당 날짜에 대한 boxoffice 순위를 알려주는
    // 서버 쪽 웹 프로그램을 호출하고 그 결과를 화면에 출력
    let user_date = $('#userInputDate').val()
    let user_key = '682ca358e95538a21f1702baa2ddb218'
    let open_api = 'http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json'

    let my_url = open_api + '?key=' + user_key + '&targetDt=' + user_date

    // 이렇게 하면 화면이 refresh돼서 원하는 작업을 할 수 없다.
    // location.href = my_url

    // JavaScript가 가지고 있는 특별한 통신방법 이용
    // AJAX : 화면이 refresh 되지 않도록 해주는 데이터 통신 방식
    // 순수 JavaScript의 AJAX 코드는 구현하기 어렵고 힘들다
    // jQuery를 이용해서 AJAX를 구현
    $.ajax({
        url : open_api,     // 호출할 서버쪽 프로그램의 URL, Query string 제외
        type : 'GET',       // 서버쪽 프로그램에 대한 request 방식
        dataType : "json",  // 서버 프로그램이 결과로 보내주는 데이터 형식
        data : {            // 서버 프로그램 호출할 때 넘어가는 데이터
            key : user_key,
            targetDt : user_date
        },
        success : function () {
            alert('success')
        },
        error : function () {
            alert('error!')
        }

    })

}

```



- css
- 실습 데이터에서 가져온 css 파일

```css
body {
  font-size: .875rem;
}

.feather {
  width: 16px;
  height: 16px;
  vertical-align: text-bottom;
}

/*
 * Sidebar
 */

.sidebar {
  position: fixed;
  top: 0;
  /* rtl:raw:
  right: 0;
  */
  bottom: 0;
  /* rtl:remove */
  left: 0;
  z-index: 100; /* Behind the navbar */
  padding: 48px 0 0; /* Height of navbar */
  box-shadow: inset -1px 0 0 rgba(0, 0, 0, .1);
}

@media (max-width: 767.98px) {
  .sidebar {
    top: 5rem;
  }
}

.sidebar-sticky {
  position: relative;
  top: 0;
  height: calc(100vh - 48px);
  padding-top: .5rem;
  overflow-x: hidden;
  overflow-y: auto; /* Scrollable contents if viewport is shorter than content. */
}

.sidebar .nav-link {
  font-weight: 500;
  color: #333;
}

.sidebar .nav-link .feather {
  margin-right: 4px;
  color: #727272;
}

.sidebar .nav-link.active {
  color: #007bff;
}

.sidebar .nav-link:hover .feather,
.sidebar .nav-link.active .feather {
  color: inherit;
}

.sidebar-heading {
  font-size: .75rem;
  text-transform: uppercase;
}

/*
 * Navbar
 */

.navbar-brand {
  padding-top: .75rem;
  padding-bottom: .75rem;
  font-size: 1rem;
  background-color: rgba(0, 0, 0, .25);
  box-shadow: inset -1px 0 0 rgba(0, 0, 0, .25);
}

.navbar .navbar-toggler {
  top: .25rem;
  right: 1rem;
}

.navbar .form-control {
  padding: .75rem 1rem;
  border-width: 0;
  border-radius: 0;
}

.form-control-dark {
  color: #fff;
  background-color: rgba(255, 255, 255, .1);
  border-color: rgba(255, 255, 255, .1);
}

.form-control-dark:focus {
  border-color: transparent;
  box-shadow: 0 0 0 3px rgba(255, 255, 255, .25);
}

```

