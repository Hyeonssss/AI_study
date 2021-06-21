# Pday04_0415_NLP_EDA



```python
# 파일 처리
import os

# Data 처리
import pandas as pd
import numpy as np

# ML, DNN, CNN 관련 라이브러리
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, GlobalAveragePooling2D, Softmax

import efficientnet.tfkeras as efn
from sklearn.neighbors import NearestNeighbors
import math
##############################################################

# 이미지 및 그래프 출력
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

##############################################################

# 해쉬(phash) 값 처리
import imagehash

##############################################################

# Text Data NLP 처리
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import re
import nltk
nltk.download('popular')

##############################################################

# 메모리 관리
import gc

# 경고메시지 지우기
import warnings
warnings.filterwarnings(action='ignore')

# 상태바 진행상태
from tqdm import tqdm

# Text Color
from termcolor import colored
```



```python
# 전체 Data 요약

BASE_DIR = './data/'

# CSV 파일
train_df = pd.read_csv(BASE_DIR + 'train.csv')
test_df = pd.read_csv(BASE_DIR + 'test.csv')
sample_df = pd.read_csv(BASE_DIR + 'sample_submission.csv')

IMG_SIZE = [512, 512]

BATCH_SIZE = 10

N_CLASSES = len(train_df['label_group'].unique()) # 11011
```



```python
text_model = TfidfVectorizer(stop_words='english', binary=True, max_features=25000)
text_embeddings = text_model.fit_transform(train_df['title']).toarray()
text_embeddings.shape

# (34250, 24939)
```



```python
# title 단어 preprocessing
def text_preprocessing(text, flg_stem, flg_lemm):
    
    stopwords_list = nltk.corpus.stopwords.words('english')
    
    # 특수기호 제거, 모든 문자 소문자, 양옆 공백 제거
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    
    # text 문자에서 stopwords_list 에 있는 단어 제거하기
    text_list = text.split()
    
    if stopwords_list is not None :
        text_list = [word for word in text_list 
                    if word not in text_list]
        
    # -ing, -ly, ... 같은 접미어 제거 하기 (가지치기)
    if flg_stem == True :
        ps = nltk.stem.porter.PorterStemmer()
        text_list = [ps.stem(word) for word in text_list]
        
    # 뿌리 단어로 바꾸기
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        text_list = [lem.lemmatize(word) for worf in text_list]
        
    # 문자열로 되돌려놓기
    clean_text = " ".join(text_list)
    return text
```



```python
stopwords_list = nltk.corpus.stopwords.words('english')
stopwords_list
```

![image-20210415174139554](md-images/image-20210415174139554.png)



```python
# Title preprocessing
train_df['clean_title'] = train_df['title'].map(lambda x : text_preprocessing(x, True, True))

# Clean_title 글자 개수(띄어쓰기 포함)
train_df['clean_title_len'] = train_df['clean_title'].map(lambda x: len(x))

# Clean_title 단어 개수
train_df['clean_title_word_count'] = train_df['clean_title'].map(lambda x: len(str(x).split(" ")))

# Clean_title 철자(스펠링) 개수(띄어쓰기 제외)
train_df['clean_title_char_count'] = train_df['clean_title'].map(lambda x: sum(len(word) for word in str(x).split(" ")))

# Clean_title 단어 대비 글자 개수
train_df['clean_title_avg_word_length'] = train_df['clean_title_char_count'] / train_df['clean_title_word_count']

train_df.iloc[:10,3:]
```

![image-20210415174256254](md-images/image-20210415174256254.png)



```python
knn=3
model = NearestNeighbors(n_neighbors=knn)
model.fit(text_embeddings)

distance, index_num = model.kneighbors(text_embeddings)
```



```python
for i in range(5):
    plt.figure()
    plt.plot(np.arange(knn), distance[i], 'o-')
    plt.rc('font', family='NanumGothic')
    plt.title('{}번째 Title Text 거리'.format(i+1), size=20)
    plt.ylabel('Text 거리', size=18)
    plt.xlabel('최근접 이웃(KNN) 수', size=18)
    plt.show()
    
    print(train_df.loc[index_num[i,:3], ['title', 'label_group']])
```

![image-20210415174438455](md-images/image-20210415174438455.png)

```python

preprocess_text_model = TfidfVectorizer(stop_words='english', binary=True, max_features=25000)
preprocess_text_embeddings = preprocess_text_model.fit_transform(train_df['clean_title']).toarray()
preprocess_text_embeddings.shape
```



```python

knn=10
model = NearestNeighbors(n_neighbors=knn)
model.fit(preprocess_text_embeddings)

distance, index_num = model.kneighbors(preprocess_text_embeddings)
```



```python

for i in range(5):
    plt.figure(figsize=(20,5))
    plt.plot(np.arange(knn), distance[i], 'o-')
    plt.rc('font', family='NanumGothic')
    plt.title('{}번째 Clean Title Text 거리'.format(i+1), size=20)
    plt.ylabel('Text 거리', size=18)
    plt.xlabel('최근접 이웃(KNN)', size=18)
    plt.show()
    
    print(train_df.loc[index_num[i,:knn], ['title', 'clean_title', 'label_group']])

```

![image-20210415174549722](md-images/image-20210415174549722.png)