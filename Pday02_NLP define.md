# Pday02_NLP define



### NLP based EDA

#### 라이브러리 모음

```python
# 파일 처리
import os

# Data 처리
import pandas as pd
import numpy as np

# 이미지 및 그래프 출력
import matplotlib.pyplot as plt
import seaborn as sns

# 경고메시지 지우기
import warnings
warnings.filterwarnings(action='ignore')

# 상태바 상태
from tqdm import tqdm

# NLP based EDA
from nltk.corpus import stopwords
import nltk
from wordcloud import WordCloud, STOPWORDS
import re, string
```



#### 전체 데이터 요약

```python
BASE_DIR = './data/'

train_df = pd.read_csv(BASE_DIR + 'train.csv')
test_df = pd.read_csv(BASE_DIR + 'test.csv')
sample_df = pd.read_csv(BASE_DIR + 'sample_submission.csv')
```

```python
train_df.head()
```

![image-20210409173146723](md-images/image-20210409173146723.png)



### title EDA

#### World Cloud 출력

```python
stopwords_wc = set(STOPWORDS)
token_text = ''

for i in tqdm(title_text):
    token_l = i.split()
    token_text += " ".join(token_l) + " "
```

![image-20210409173305477](md-images/image-20210409173305477.png)



```python
wordcloid = WordCloud(width = 800, height = 800,
                      background_color= 'white',
                      stopwords= stopwords_wc,
                      min_font_size= 10).generate(token_text)

plt.figure(figsize = (8,8), facecolor = None)
plt.imshow(wordcloid)
plt.axis("off")
plt.tight_layout(pad = 0)

plt.show()
```

![image-20210409173338043](md-images/image-20210409173338043.png)



#### title 데이터 전처리

```python
def preprocess_text(title, flg_stemm=False, flg_lemm=True):
    stop = stopwords.words('english')
    title = [x for x in title.split() if not x in stop]
    title = " ".join(title)
    title = title.lower()
    title = re.sub(r"\-","",title)
    title = re.sub(r"\+","",title)
    title = re.sub (r"&","and",title)
    title = re.sub(r"\|","",title)
    title = re.sub(r"\\","",title)
    title = re.sub(r"\W"," ",title)
    for p in string.punctuation :
        title = re.sub(r"f{p}","",title)
    title = re.sub(r"\s+"," ",title)
    
    lst_text = title.split()
    ## remove Stopwords
    if stop is not None:
        lst_text = [word for word in lst_text if word not in stop]
                
    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
                
    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()    
        lst_text = [lem.lemmatize(word) for word in lst_text]
            
    title = " ".join(lst_text)

    return title
```



##### title 데이터 전처리 컬럼 추가

```python
train_df['clean_title'] = train_df['title'].apply(
    lambda x : preprocess_text(x, flg_stemm=False, flg_lemm=True, ))
```

```python
display(train_df.head())
```

![image-20210409173425485](md-images/image-20210409173425485.png)




```python
#Length of Title
train_df['clean_title_len'] = train_df['clean_title'].apply(lambda x: len(x))
```

```python
#Word Count
train_df['clean_title_word_count'] =train_df["clean_title"].apply(lambda x: len(str(x).split(" ")))
```

```python
#Character Count
train_df['clean_title_char_count'] = train_df["clean_title"].apply(lambda x: sum(len(word) for word in str(x).split(" ")))
```

```python
#Average Word Length
train_df['clean_title_avg_word_length'] = train_df['clean_title_char_count'] / train_df['clean_title_word_count']
```

```python
display(train_df.head())
```

![image-20210409173525703](md-images/image-20210409173525703.png)



#### 그래프로 확인

```python
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

def plot_distribution(x, title):

    fig = px.histogram(
    train_df, 
    x = x,
    width = 800,
    height = 500,
    title = title
    )
    
    fig.show()
```

```python
plot_distribution(x = 'clean_title_len', title = 'Title Length Distribution')
```

![image-20210409173558549](md-images/image-20210409173558549.png)

```python
plot_distribution(x = 'clean_title_word_count', title = 'Word Count Distribution')
```

![image-20210409173626456](md-images/image-20210409173626456.png)

```python
plot_distribution(x = 'clean_title_char_count', title = 'Character Count Distribution')
```

![image-20210409173649216](md-images/image-20210409173649216.png)

```python
plot_distribution(x = 'clean_title_avg_word_length', title = 'Average Word Length Distribution')
```

![image-20210409173705715](md-images/image-20210409173705715.png)