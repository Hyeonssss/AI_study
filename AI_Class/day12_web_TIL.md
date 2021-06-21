# day12_web_TIL

# web programming 기본개념



### web programming

- HTTP protocol로 통신하는 클라리언트 프로그램과 서버 프로그램을 개발



#### protocol

- 데이터 통신을 위해 지켜야할 약속, 규약

- 웹 클라이언트 프로그램, 웹 서버 프로그램

  각자 따로 개발할 수도 있고 함께 개발도 가능



- web program은 cs구조로 되어있다

  client : 능동적으로 서비스를 요청

  server : 클라이언트 요청에 대해 service 제공

- web client : web server에 접속하는 클라이언트 프로그램

  ex) 브라우저(크롬, IE, safari, 파이어폭스)

- web server : web client의 요청을 받아 서비스를 제공하는 프로그램

  ex) Apache web server



#### 우리가 만드는 것

- web client application

- web server application

  ex) Django(python), servlert(java)



- protocol : HTTP

- port : 0 ~ 65535 안에 있는 하나의 숫자(포트 번호)

  0 ~ 1024: reserved(예약 번호)

  1025 ~ 65535: 사용자가 이용

  포트 번호: 하나의 process(프로그램)

  - ip address와 함께 port 번호를 이용하면 외부에서 내 컴퓨터 내의 프로그램을 사용할 수 있다

  - port는 프로그램을 지칭

    

- ip주소: network에 연결되어있는 각종 기기에 부여되는 논리적인 주소

  4자리로 구성 됨 (192.235.37.64)

- MAC address: network에 연결되는 각종 기기에 부여되는 물리적인 주소

  6자리로 구성 됨 (34.37.128.34.2.76)



- URL = HTTP://	192.168.0.34	:	4000

  ​		 (protocol)	(IP address)		(port)



- web Client(브라우저, chrome)

- web server(포트 번호: 80): 기본적으로 포트번호 80이 예약되어있음

  web client -> web server: 요청 HTTP(Request)

  web server -> web client: 응답 HTTP(Response)

  

- HTTP protocol의 특징

  client와 server는 request와 response를 주고받고 나면 연결을 끊음

  request할 때 연결하고 response 보내고 나면 연결 끊음

  서버는 1개 클라이언트는 다수: 로드 과부하를 막기 위해서 클라이언트가 요청할 때만 잠시 연결

  

- stateless: 연결을 끊을 때 정보가 사라지기 때문에 서버가 client를 구별할 수 없다



#### web client

> web client program

- 우리가 직접 작성할 수 있다

  브라우저



#### web server

> web server program

- performance가 중요, 우리가 작성하지 않음

  Apache, IIS, oracle web server



#### protocol

- HTTP(web 전용 protocol)
- FTP(file 전송 전용 protocol)
- SMTP(e-mail 전용 protocol)



#### IP, Port

- Protocol, IP, Port => URL 생성 가능



#### Request, Response

- web client -> web server: 요청 HTTP(Request)

  web server -> web client: 응답 HTTP(Response)



### HTTP Protocol

> HyperText Transfer Protocol

- 웹 서버와 웹 클라이언트가 서로 데이터를 주고 받기 위해 사용하는 통신규약

  TCP/IP protocol stack 위에서 동작하는 protocol => IP address를 기반으로 통신

- HTTP -> Text뿐 아니라 동영상, 이미지, pdf 등 여러 종류의 Data를 주고 받을 수 있음



#### HTTP Request message

- GET / 특정 URL, 현재 protocol 버전, host URL

  빈 줄

  body

- GET: 데이터를 가져올 때

  POST: 새로운 데이터를 등록할 때

  PUT: 기존 데이터를 변경할 때

  DELETE: 서버 쪽  데이터를 삭제할 때 

- REST: 요청 방식 네가지를 합쳐서 통칭, Request 작성 원칙



- GET, POST

  두 가지 방식으로도 모든 CRUD 작업이 가능

- GET 방식 -> 내가 전달할 데이터를 Query String 형식으로 URL 뒤에 붙여서 보냄

  ​				=> 일일 박스 office 순위 => Open API 보내는 것과 동일

  ​				=> ~~~~?key=----------&targetDt=20200101 

  장점: 사용하기 쉽다

  단점: 보안성이 없다, URL 길이제한이 있다

- POST방식 -> request message body 부분에 데이터를 포함하여 전송

  장점: 보안성이 있고, 보내려는 데이터에 길이 제한이 없다



#### WAS

- web server vs WAS(web application server)

- web client(browser) -- request(GET, POST) --> web server(80)
  - web server는 정적 resource에 대한 response만 줄 수 있다

    현재 가지고 있는 HTML, 이미지, 동영상 데이터는 줄 수 있음

    그러나 특정 프로그램을 호출할 수 있는 능력은 없음 

- 그래서 WAS에 위임: web server -- request --> WAS(web application server)
  - WAS는 DB에서 데이터 가져옴



- web client -> web server -> WAS -> Database

  웹의 기본 구조



- web client(브라우저: chrome) ----> web server
  1. web client가 a.html request(GET)
  2. web server가 가지고 있던 a.html response
  3. web client는 받아서 렌더링 후 화면에 출력



### HTML

- HTML (내용)
- CSS (모양)
- Javascript (프로그램 쿼리)



### WebStorm

1. web server program 실행

   (webstorm 안에 있는 내장 web server 이용)

3. web server가 우리 project를 인식해야 나중에 client가 요청했을 때 파일을 찾아서 제공할 수 있음 -> configure
4. 우리 project를 web에서 사용할 수 있도록 전개 -> deploy

5. web client(browser) 실행, URL 입력(request)
6. 해당 a.html 파일을 받아 렌더링



#### HTML 태그

- br - 줄 바꿈

- CSS

  ```html
  <style>
      div {
          background-color: red;
          color: yellow;
      }
  </style>
  
  <div> </div>로 감싸진 부분을 데코레이션
  ```