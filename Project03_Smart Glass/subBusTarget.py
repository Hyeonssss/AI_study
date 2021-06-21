import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish
import torch
import pandas as pd
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pytesseract
import re
import os
import threading
from PIL import Image 
# import Image


# ------------------------- yolov5s 모델 import -------------------------
from pathlib import Path

from models.yolo import Model, attempt_load
from utils.general import check_requirements, set_logging
from utils.google_utils import attempt_download
from utils.torch_utils import select_device

check_requirements(Path(__file__).parent / 'requirements.txt', exclude=('tensorboard', 'pycocotools', 'thop'))
# set_logging(verbose=verbose)
device = None

name1 = 'yolov5_custom_pink.pt'
fname1 = Path(name1).with_suffix('.pt')  # checkpoint filename
model1 = attempt_load(fname1, map_location=torch.device('cpu'))
model1 = model1.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
model1.conf = 0.5

name2 = 'yolov5_custom.pt'
fname2 = Path(name2).with_suffix('.pt')  # checkpoint filename
model2 = attempt_load(fname2, map_location=torch.device('cpu'))
model2 = model2.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
model2.conf = 0.5

device = select_device('0' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
# ------------------------- 모델 import fin -------------------------




"""
on_connect는 subscriber가 브로커에 연결하면서 호출할 함수
rc가 0이면 정상접속이 됐다는 의미
"""

# ------------------------- 구조물 인식 함수 -------------------------
def notice(img):
    # 현재 이미지 불러오기
    results_img = model1(img)
    results_img_xy_df = results_img.pandas().xyxy[0]
    results_img_xy_df = results_img_xy_df.sort_values(by=['confidence'], ascending=False)
    results_img_xy_df = results_img_xy_df.iloc[0:1]

    if len(results_img_xy_df) != 0 :
        print('******************if**********************')
        x_center = (results_img_xy_df[results_img_xy_df['name'] == 'pink_point'].iloc[:, 2] + results_img_xy_df[results_img_xy_df['name'] == 'pink_point'].iloc[:, 0]) / 2
 
    else:
        print('******************else********************')
        return 'X'

    # 카메라 중점
    print('\n')
    print(x_center)
    cam_view_center = 360
    if cam_view_center*0.6 < float(x_center) < cam_view_center*1.4:
        print("C")
        return 'C'
    elif cam_view_center < float(x_center):
        print("R")
        return 'R'
    elif cam_view_center > float(x_center):
        print("L")
        return 'L'


# ------------------------- yolov5s 모델로 input 이미지 디텍팅 함수 -------------------------
def objectDetection(img):
    results_img = model2(img)
    results_img.crop()
#     results_img.save()


# ------------------------- OCR_v1 -------------------------
# OCR 버전마다 전처리 과정이 다름
def OCR_pn(img):
    img = cv2.imread(img) # 이미지 로드
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # 배경에 그레이 적용
    gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC) # 사이즈 정규화(키우기)
    blur = cv2.GaussianBlur(gray, (5,5), 0) # 이미지에 블러 처리

    # ------------------- 기울기 조정 start -------------------
    canny = cv2.Canny(blur, 700, 350, apertureSize = 5, L2gradient = True) # 이미지 외곽선만 추출
    lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 50, minLineLength = 3, maxLineGap = 150) # 직선 찾기

    angle = 0
    maxdim = 0
    # 각도 조정
    if not (lines is None):
        for i in lines:
            xdim = i[0][2] - i[0][0]
            ydim = i[0][3] - i[0][1]
            iangle = math.atan2(ydim, xdim)*180/np.pi
            dim = math.sqrt((xdim * xdim) + (ydim * ydim))
            if abs(angle) < 40 and maxdim < dim:
                maxdim = dim
                angle =iangle

    roih, roiw, roic = img.shape
    matrix = cv2.getRotationMatrix2D((roiw/2, roih/2), angle, 1)
    roi = cv2.warpAffine(img, matrix, (roiw, roih))
    # ------------------- 기울기 조정 fin -------------------

    roi = cv2.resize(roi, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    blur_2 = cv2.GaussianBlur(roi, (5,5), 0)
    
    plate_num = ""
    try:
        # tesseract OCR 적용
        text = pytesseract.image_to_string(blur_2, config='-c tessedit_char_whitelist=0123456789 --psm 7 --oem 1') # whitelist: 숫자만 인식
        plate_num = re.sub('[\W_]+', '', text) # 특수문자 제거
    except:
        text = None
    
    plate_num = re.sub('[\W_]+', '', text)   
    return plate_num[-4:]


# ------------------------- OCR_v2 -------------------------
def OCR_pn2(img):
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    plate_num = ""
    try:
        text = pytesseract.image_to_string(blur, config='-c tessedit_char_whitelist=0123456789 --psm 7 --oem 1')
        plate_num = re.sub('[\W_]+', '', text)            
    except:
        text = None

    plate_num = re.sub('[\W_]+', '', text)   
    return plate_num[-4:]


# ------------------------- OCR_v3 -------------------------
def OCR_pn3(img):
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # ------------------- 기울기 조정 start -------------------
    canny = cv2.Canny(blur, 700, 350, apertureSize = 5, L2gradient = True)
    lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 50, minLineLength = 3, maxLineGap = 150)

    angle = 0
    maxdim = 0
    if not (lines is None):
        for i in lines:
            xdim = i[0][2] - i[0][0]
            ydim = i[0][3] - i[0][1]
            iangle = math.atan2(ydim, xdim)*180/np.pi
            dim = math.sqrt((xdim * xdim) + (ydim * ydim))
            if abs(angle) < 40 and maxdim < dim:
                maxdim = dim
                angle =iangle

    roih, roiw, roic = img.shape
    matrix = cv2.getRotationMatrix2D((roiw/2, roih/2), angle, 1)
    roi = cv2.warpAffine(img, matrix, (roiw, roih))
    # ------------------- 기울기 조정 fin -------------------

    gray_2 = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    gray_2 = cv2.resize(gray_2, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    blur_2 = cv2.GaussianBlur(gray_2, (5,5), 0)

    ## 이미지 흑백 대조하기, 최적 임계값을 자동으로 추출하는 Otsus 사용
    ret, thresh = cv2.threshold(blur_2, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    ## 확장을 위한 커널 생성
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    ## 글씨 이미지 크기 확장
    dilation = cv2.dilate(thresh, rect_kern, iterations = 1)

    blur_3 = cv2.GaussianBlur(dilation, (5,5), 0)
    
    plate_num = ""
    try:
        text = pytesseract.image_to_string(blur_3, config='-c tessedit_char_whitelist=0123456789 --psm 7 --oem 1')
        plate_num = re.sub('[\W_]+', '', text)            
    except:
        text = None
    
    plate_num = re.sub('[\W_]+', '', text)   
    return plate_num[-4:]
            
        
# ------------------------- OCR_v4 -------------------------
def OCR_bn(img):
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    ## 이미지 흑백 대조하기, 최적 임계값을 자동으로 추출하는 Otsus 사용
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    ## 확장을 위한 커널 생성
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    ## 글씨 이미지 크기 확장
    dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
    blur_2 = cv2.GaussianBlur(dilation, (5,5), 0)
    
    plate_num = ""
    try:
        text = pytesseract.image_to_string(blur_2, config='-c tessedit_char_whitelist=0123456789 --psm 7 --oem 1')            
    except:
        text = None
        
    plate_num = re.sub('[\W_]+', '', text)   
    return plate_num[-4:]


# ------------------------- 목표 버스 일치 여부 확인 함수 -------------------------
def tfBusNum(store_img, busNum, busLicenseNum):
    for img in store_img:
        try:
            preBusNum = OCR_bn(img) # OCR_v1 output
            preBusNum2 = OCR_pn(img) # OCR_v1 output
            preBusNum3 = OCR_pn2(img) # OCR_v1 output
            preBusNum4 = OCR_pn3(img) # OCR_v1 output
            print(f'busNum: {busNum}') # 목표 버스 번호 
            print(f'busLicenseNum: {busLicenseNum}') # 목표 버스 차량번호
            print('---------------예측-------------------')
            print(f'prediction busNum: {preBusNum}')
            print(f'prediction busNum2: {preBusNum2}')
            print(f'prediction busNum3: {preBusNum3}')
            print(f'prediction busNum4: {preBusNum4}')

            # 목표 버스 번호와 일치 여부 확인
            if preBusNum == busNum:
                print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                return True
            if preBusNum2 == busNum:
                print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                return True
            if preBusNum3 == busNum:
                print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                return True
            if preBusNum4 == busNum:
                print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                return True

            # 목표 버스 차량번호와 일치 여부 확인
            if preBusNum == busLicenseNum:
                print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                return True
            if preBusNum2 == busLicenseNum:
                print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                return True
            if preBusNum3 == busLicenseNum:
                print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                return True
            if preBusNum4 == busLicenseNum:
                print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                return True
        except:
            continue
    
    
# ------------------------- 버스 문 인식 함수 -------------------------
def door(img):
    # 현재 이미지 불러오기
    results_img = model2(img)
    results_img_xy_df = results_img.pandas().xyxy[0]

    if len(results_img_xy_df[results_img_xy_df['name']=='door_open']) != 0:
        print('***************if-1-door**************')
        x_center = (results_img_xy_df[results_img_xy_df['name']=='door_open'].iloc[:,2] + results_img_xy_df[results_img_xy_df['name']=='door_open'].iloc[:,0])/2

    if len(results_img_xy_df[results_img_xy_df['name']=='door']) != 0:
        print('***************if-2-door**************')
        x_center = (results_img_xy_df[results_img_xy_df['name']=='door'].iloc[:,2] + results_img_xy_df[results_img_xy_df['name']=='door'].iloc[:,0])/2
        
    else:
        print('***************else-door**************')
        # 인식 안됨
        return 'X'

    # 카메라 중점
    print(x_center)
    cam_view_center = 360
    if cam_view_center*0.6 < float(x_center) < cam_view_center*1.4:
        return 'C'
        
    elif cam_view_center < float(x_center):
        return 'R'
        
    elif cam_view_center > float(x_center):
        return 'L'
    

#연결 확인
def on_connect(client, userdata, flags, rc):
    print("connect.." + str(rc))
    if rc == 0:
        client.subscribe("eyeson/#")
        print(1)
    else:
        print("연결실패")


def ai(uuid,img_pink,img_bus,store_img, busNum, busLicenseNum):
    print('333')
    
    
    # ------------------------- 표지판 인식 while문(Center가 30번 나올때까지 반복) -------------------------
    pre = None # 직전 표지판 방향
    cnt = 0
    while True:
        try:
            direction = notice(img_pink) # 현재 표지판 방향
        except:
            direction = None
            
        # R: 오른쪽에 표지판이 있습니다
        # L: 왼쪽에 표지판이 있습니다
        # C: 정면에 표지판이 있습니다
        if direction == 'R':
            cnt += 1
            if pre != direction or cnt >= 30: # 직전 표지판 방향과 현재 표지판 방향이 다르거나 cnt가 30 이상이면 퍼블리싱
                publish.single("eyeson/" + uuid,'ai/noticeInfo/R', hostname="15.164.46.54")
                cnt = 0
            pre = direction
            continue
        if direction == 'L':
            cnt += 1
            if pre != direction or cnt >= 30:
                publish.single("eyeson/" + uuid,'ai/noticeInfo/L', hostname="15.164.46.54")
                cnt = 0
            pre = direction
            continue
        if direction == 'C':
            cnt += 1
            print(f'<<<<<정면입니다>>>>> cnt: {cnt}')
            if pre != direction:
                publish.single("eyeson/" + uuid, 'ai/noticeInfo/C', hostname="15.164.46.54")
                cnt = 0
            if cnt >= 30:
                # center가 30번 누적으로 나오면 알림 주고 break
                # correctC: 정면에 표지판이 있습니다. 이 방향으로 다가가주세요
                publish.single("eyeson/" + uuid, 'ai/noticeInfo/correctC', hostname="15.164.46.54")
                break
            pre = direction
            continue
        if direction == 'X':
            cnt += 1
            if pre != direction or cnt >= 30:
                # X: 표지판이 인식되지 않았습니다. 차도 주위를 둘러봐주세요
                publish.single("eyeson/" + uuid, 'ai/noticeInfo/X', hostname="15.164.46.54")
                cnt = 0
            pre = direction
            continue
        else:
            cnt = 0
            continue


    # ------------------------- 버스 번호 인식 함수(표지판 인식이 끝난 후 실행) -------------------------
    while True:
        try:
            objectDetection(img_bus) # yolov5s 모델로 input image 디텍팅 후 크롭된 이미지 저장
        except:
            pass
        # 버스 번호 crop 이미지 저장

        # OCR 통해 버스 번호 일치 여부 확인, True 나오면 break
        if tfBusNum(store_img, busNum, busLicenseNum):
            print(uuid + "타겟버스")
            publish.single("eyeson/" + uuid,"ai/targetBus", hostname="15.164.46.54")
            break


    # ------------------------- 문 방향 인식 함수(목표 버스가 일치함을 확인 후 실행) -------------------------            
    pre = None
    cnt = 0
    while True:
        try:
            direction = door(img_bus)
        except:
            direction = None

        # R: 오른쪽에 문이 있습니다
        # L: 왼쪽에 문이 있습니다
        # C: 정면에 문이 있습니다
        if direction == 'R':
            cnt += 1
            if pre != direction or cnt >= 30:
                publish.single("eyeson/" + uuid, 'ai/doorInfo/R', hostname="15.164.46.54")
                cnt = 0
            pre = direction
            continue
        if direction == 'L':
            cnt += 1
            if pre != direction or cnt >= 30:
                publish.single("eyeson/" + uuid, 'ai/doorInfo/L', hostname="15.164.46.54")
                cnt = 0
            pre = direction
            continue
        if direction == 'C':
            cnt += 1
            if pre != direction:
                publish.single("eyeson/" + uuid, 'ai/doorInfo/C', hostname="15.164.46.54")
                cnt = 0
            if cnt >= 30:
                # center가 30번 누적으로 나오면 알림 주고 break
                # correctC: 정면에 문이 있습니다. 이 방향으로 다가가주세요
                publish.single("eyeson/" + uuid, 'ai/doorInfo/correctC', hostname="15.164.46.54")
                break
            pre = direction
            continue
        if direction == 'X':
            cnt += 1
            if pre != direction or cnt >= 30:
                # X: 문이 인식되지 않았습니다. 차도 주위를 둘러봐주세요
                publish.single("eyeson/" + uuid, 'ai/doorInfo/X', hostname="15.164.46.54")
                cnt = 0
            pre = direction
            continue
        else:
            cnt = 0
            continue
    

# 메시지가 도착됐을때 처리할 일들 - 여러가지 장비 제어하기, Mongodb에 저장
def on_message(client, userdata, msg): # 최초 1번
    try:
        #global img_bus, busNum, busLicenseNum, base, store_img, uuid, img_pink
        myval = msg.payload.decode("utf-8")
        myval = myval.replace(" ", "")
        myval = myval.split("/")
        mytopic = msg.topic.split("/")
        uuid = mytopic[1]
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>" + uuid + "uuid 확인")
        print(myval)
        base = '/home/lab08/yolov5/runs/hub/exp/crops/'
#         img_pink = '/home/lab19/ai_env_mino/yolov5/' + uuid + '.jpg'
        img_pink = '/home/lab08/yolov5/'+uuid+'.jpg'
#         img_pink = '/home/lab08/yolov5/test_pink.jpg'

#         img_bus = './test_bus.jpg'
        img_bus = '/home/lab08/yolov5/'+uuid+'.jpg'
#         img_bus = './'+uuid+'.jpg'
        store_img = [base+'bus_num_fr/'+uuid+'.jpg', base+'bus_num_side/'+uuid+'.jpg', base+'license_plate/'+uuid+'.jpg']

         
        if myval[0] == "android":
            if myval[1] == "ai":
                busNum = myval[2]
                print(busNum)
                busLicenseNum = myval[3]
                print(busLicenseNum)
            #     uuid = '12345'
#                 busNum = '5714'
#                 busLicenseNum = '2914'
                print("bigData last 확인")


                # 표지판 인식 센터 잡기 -> 목표 버스 확인 -> 도어 인식 센터 잡기
                # subCamera.py 에선 실시간으로 바이트로 넘어오는 파일을 이미지로 저장(특정 폴더에)
                # subBusTarget.py 는 최초 1번 정보를 받고 while문으로 갱신되는 이미지를 예측
                
                # 쓰레드 설정 
                print('111')
                thread = threading.Thread(target=ai, args=(uuid,img_pink,img_bus,store_img,busNum,busLicenseNum,))
                # 
                print('222')
                # 쓰레드 시작
                thread.start()
    except:
        pass
        
mqttClient = mqtt.Client()  # 클라이언트 객체 생성
# 브로커에 연결이되면 내가 정의해놓은 on_connect함수가 실행되도록 등록
mqttClient.on_connect = on_connect

# 브로커에서 메시지가 전달되면 내가 등록해 놓은 on_message함수가 실행
mqttClient.on_message = on_message

# 브로커에 연결하기
mqttClient.connect("15.164.46.54", 1883, 60)

# 토픽이 전달될때까지 수신대기
mqttClient.loop_forever()
