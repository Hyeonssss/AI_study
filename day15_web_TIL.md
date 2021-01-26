# day15_web_TIL



### dailBoxOfficeSearch

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
                    <tbody id="my_tbody">

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

```javascript
function my_func() {
    // 사용자가 입력한 날짜를 가져와서
    // 해당 날짜에 대한 boxoffice 순위를 알려주는
    // 서버 쪽 웹 프로그램을 호출하고 그 결과를 화면에 출력
    let user_date = $('#userInputDate').val()
    let user_key = '682ca358e95538a21f1702baa2ddb218'
    let open_api = 'http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json'

    let my_url = open_api + '?key=' + user_key + '&targetDt=' + user_date

    // 이렇게 하면 화면이 refresh 돼서 원하는 작업을 할 수 없다.
    // location.href = my_url

    // JavaScript 가 가지고 있는 특별한 통신방법 이용
    // AJAX : 화면이 refresh 되지 않도록 해주는 데이터 통신 방식
    // 순수 JavaScript 의 AJAX 코드는 구현하기 어렵고 힘들다
    // jQuery 를 이용해서 AJAX 를 구현
    $.ajax({
        url : open_api,     // 호출할 서버쪽 프로그램의 URL, Query string 제외
        type : 'GET',       // 서버쪽 프로그램에 대한 request 방식
        dataType : "json",  // 서버 프로그램이 결과로 보내주는 데이터 형식
        data : {            // 서버 프로그램 호출할 때 넘어가는 데이터
            key : user_key,
            targetDt : user_date
        },
        // success 의 function 인자로 결과 json 데이터가 들어옴
        // 단순 문자열인 json 데이터를 javascript 객체로 변환
        // dataType 에 json 을 명시하면 자동으로 객체로 변환해줌
        success : function (result) {
            // 현재 데이터로 갱신하기0
            $('#my_tbody').empty()
            let movie_list = result["boxOfficeResult"]["dailyBoxOfficeList"]
            // jQuery for문 사용
            // 필요한 요소들을 각각 저장 
            for(let i=0; i<movie_list.length; i++) {
                let m_name = movie_list[i].movieNm
                let m_rank = movie_list[i].rank
                let m_audi = movie_list[i].audiAcc
                let m_sales = movie_list[i].salesAcc
                let m_openDt = movie_list[i].openDt
                // 데이터를 가져왔으니 HTML element 를 생성
                //     <tr>
                //         <td>1,001</td>
                //         <td>Lorem</td>
                //         <td>ipsum</td>
                //         <td>dolor</td>
                //         <td>sit</td>
                //     </tr>
                // <tr>    </tr>
                let tr = $('<tr></tr>')
                let rank_td = $('<td></td>').text(m_rank)
                let name_td = $('<td></td>').text(m_name)
                let audi_td = $('<td></td>').text(m_audi)
                let sales_td = $('<td></td>').text(m_sales)
                let open_td = $('<td></td>').text(m_openDt)
                // tr 태그로 모두 통합
                tr.append(rank_td)
                tr.append(name_td)
                tr.append(audi_td)
                tr.append(sales_td)
                tr.append(open_td)
                // tbody 밑에 tr태그 통째로 삽입
                $('#my_tbody').append(tr)

            }
        },
        error : function () {
            alert('error!')
        }

    })

}
```



### jQuery_sample03

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
    <script src="js/jQuery_sample03.js"></script>
    <style>
        .myStyle {
            background-color: yellow;
            color: red;
        }
    </style>
</head>
<body>
    <!-- jQuery Event -->
    <!-- 가장 쉬운 이벤트처리 방법: html 이 가지고 있는 이벤트 관련 속성 사용 -->
    <!-- <h1 onclick="my_func()">여기는 h1 영역입니다.</h1>-->
    <!-- <h1 onmouseover="set_style()"-->
    <!-- onmouseleave="release_style()">여기는 h1 영역입니다.</h1>-->
    <h1>사용자이름 : 아이유</h1>
    <h1>사용자이름 : 김연아</h1>
    <!-- <input type="button" value="click"-->
    <!--        onclick="add_event()">-->
</body>
</html>
```



- jQuery
- on('ready'): html 렌더링이 모두 끝나고 난 후 실행 되는 함수
- this : python의 객체지향에서 self와 동일
- on(): element 에 이벤트 처리능력을 부여

```javascript
// on('ready'): html 렌더링이 끝나고 나면
$(document).on('ready', function () {

    $('h1').on('click', function () {
        // alert('클릭되었어요')
        // 이벤트가 발생했을 때 어떤 element 에서 event 가 발생했는지를 파악
        // this : python 의 self 와 동일
        alert( $(this).text() )
    })

})

// 축약해서 이렇게도 표현 가능
// $(document).ready(function() {
//
// })

function my_func() {
    alert('success')
}

function set_style() {
    $('h1').addClass('myStyle')
}
function release_style() {
    $('h1').removeClass('myStyle')
}
function add_event() {
    // h1 을 찾아서 해당 element 에 event 처리 능력을 부여
    // on이라는 함수를 이용해서
    $('h1').on('click', function (event) {
        alert('h1 이 클릭되었어요!')
    })
}
```



### jQuery_sample04

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
    <script src="js/jQuery_sample04.js"></script>
</head>
<body>

    <div id="myDiv"></div>
    <input type="button", value="검색엔진 호출"
           onclick="my_search()">
</body>
</html>
```



- jQuery
- 다음 검색 엔진 프로그램 API 사용

```javascript
function my_search() {
    // 버튼 누르면 호출돼요
    let keyword = '건축학개론'
    // AJAX 방식으로 서버프로그램 호출
    // Daum KAKAO 의 이미지 검색 프로그램 호출
    $.ajax({
        url : "https://dapi.kakao.com/v2/search/image",
        type : 'GET',
        dataType : "json",
        data : {
            query : keyword
        },
        headers : {
            Authorization: 'KakaoAK 112d09d48415ab826f9766541c782f9d',
        },
        success : function (result) {
            alert('success!')
            let img_list = result["documents"]
            let img_url = img_list[0].thumbnail_url
            let img_temp = $('<img />').attr('src',img_url)
            $('#myDiv').append(img_temp)

        },
        error : function () {
            alert('error!')
        }
    })
}

// for(let i=0; i<img_list.length; i++) {
//     let img_url = img_list[i].thumbnail_url
//     let img_temp = $('<img />').attr('src',img_url)
//     $('#myDiv').append(img_temp)
// }
```



### BoxOffice 실습 최종

- html
- bootstrap  example 사용

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
    <link href="css/dashboard.css" rel="stylesheet">

    <script src="js/BoxOffice_T.js"></script>

</head>

<body>

<header class="navbar navbar-dark sticky-top bg-dark flex-md-nowrap p-0 shadow">
    <a class="navbar-brand col-md-3 col-lg-2 me-0 px-3" href="#">BoxOffice</a>
    <button class="navbar-toggler position-absolute d-md-none collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#sidebarMenu" aria-controls="sidebarMenu" aria-expanded="false" aria-label="Toggle navigation">
        <span class="navbar-toggler-icon"></span>
    </button>
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
            <h2>DailyBoxOffice Search Rank</h2>
            <div class="table-responsive">
                <table class="table table-striped table-sm">
                    <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Name</th>
                        <th>포스터</th>
                        <th>누적매출액</th>
                        <th>포스터보기</th>
                    </tr>
                    </thead>
                    <tbody id="my_tbody">

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
- 영화진흥위원회 open API와 kakao 검색 open API 사용
- 2중으로 ajax 호출

```javascript
function my_func() {
    let user_date = $('#userInputDate').val()
    let user_key = '682ca358e95538a21f1702baa2ddb218'
    let open_api = 'http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json'

    $.ajax({
        url : open_api,
        type : 'GET',
        dataType : "json",
        data : {
            key : user_key,
            targetDt : user_date
        },
        success : function (result) {
            $('#my_tbody').empty()
            let movie_list = result["boxOfficeResult"]["dailyBoxOfficeList"]
            for(let i=0; i<movie_list.length; i++) {
                let m_name = movie_list[i].movieNm
                let m_rank = movie_list[i].rank
                let m_sales = movie_list[i].salesAcc

                let tr = $('<tr></tr>')
                
                let rank_td = $('<td></td>').text(m_rank)
                let name_td = $('<td></td>').text(m_name)
                let img_td = $('<td></td>')
                let sales_td = $('<td></td>').text(m_sales)
                let poster_td = $('<td></td>')
                
                let poster_btn = $('<input />').attr('type', 'button')
                    .attr('value', '포스터보기')
                
                poster_td.append(poster_btn)
                
                poster_btn.on('click',function () {
                    let keyword = m_name + ' 포스터'
                    $.ajax({
                        url : "https://dapi.kakao.com/v2/search/image",
                        type : 'GET',
                        dataType : "json",
                        data : {
                            query : keyword
                        },
                        headers : {
                            Authorization: 'KakaoAK 112d09d48415ab826f9766541c782f9d',
                        },
                        success : function (result) {
                            img_td.empty()
                            let img_list = result["documents"]
                            let img_url = img_list[0].thumbnail_url
                            let img_temp = $('<img />').attr('src',img_url)
                            img_td.append(img_temp)
                        },
                        error : function () {
                            alert('error!')
                        }
                    })
                })
                tr.append(rank_td)
                tr.append(name_td)
                tr.append(img_td)
                tr.append(sales_td)
                tr.append(poster_td)
                $('#my_tbody').append(tr)
            }
        },
        error : function () {
            alert('error!')
        }
    })
}
```

