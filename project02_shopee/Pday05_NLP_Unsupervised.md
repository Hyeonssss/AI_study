# Pday05_0415_NLP_Unsupervised



```python
# DATA_PATH = '../input/'
DATA_PATH = './data/'

import psutil
```



```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2, matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import gc

def getMetric(col):
    def f1score(row):
        n = len( np.intersect1d(row.target,row[col]) )
        return 2*n / (len(row.target)+len(row[col]))
    return f1score
```



```python
COMPUTE_CV = True

test = pd.read_csv(DATA_PATH + 'test.csv')
if len(test)>3: COMPUTE_CV = False
else: print('this submission notebook will compute CV score, but commit notebook will not')

# COMPUTE_CV = False

if COMPUTE_CV:
    train = pd.read_csv(DATA_PATH + 'train.csv')
    train['image'] = DATA_PATH + 'train_images/' + train['image']
    tmp = train.groupby('label_group').posting_id.agg('unique').to_dict()
    train['target'] = train.label_group.map(tmp)
    # train_gf = cudf.read_csv(DATA_PATH + 'train.csv')
else:
    train = pd.read_csv(DATA_PATH + 'test.csv')
    train['image'] = DATA_PATH + 'test_images/' + train['image']
    # train_gf = cudf.read_csv(DATA_PATH + 'test.csv')
    
print('train shape is', train.shape )
train.head()
```

![image-20210415173747439](md-images/image-20210415173747439.png)



```python
from sklearn.feature_extraction.text import TfidfVectorizer
model = TfidfVectorizer(stop_words=None, binary=True, max_features=55000)
text_embeddings = model.fit_transform(train.title).toarray()
print('text embeddings shape',text_embeddings.shape)
```

![image-20210415173821106](md-images/image-20210415173821106.png)



```python
import torch
text_embeddings = torch.from_numpy(text_embeddings)
text_embeddings = text_embeddings.cuda()
```



```python
preds = []
CHUNK = 1024

print('Finding similar titles...')
CTS = len(train)//CHUNK
if len(train)%CHUNK!=0: CTS += 1
CTS_index = 0
for j in range( CTS ):
    
    a = j*CHUNK
    b = (j+1)*CHUNK
    b = min(b,len(train))
    print('chunk',a,'to',b)
    
    # COSINE SIMILARITY DISTANCE
    # cts = np.dot( text_embeddings, text_embeddings[a:b].T).T
    cts = torch.matmul(text_embeddings, text_embeddings[a:b].T).T
    cts = cts.data.cpu().numpy()
    print(cts.shape)
    for k in range(b-a):
        # IDX = np.where(cts[k,]>0.7)[0]
        IDX = np.where(cts[k,]>0.7)[0]
        o = train.iloc[IDX].posting_id.values
        preds.append(o.tolist())
        CTS_index += 1
# del model, text_embeddings
```

![image-20210415173902995](md-images/image-20210415173902995.png)

![image-20210415173848175](md-images/image-20210415173848175.png)



```python
train['oof_text'] = preds

if COMPUTE_CV:
    train['f1'] = train.apply(getMetric('oof_text'),axis=1)
    print('CV score for baseline =',train.f1.mean())
    
    

```

![image-20210415173919837](md-images/image-20210415173919837.png)