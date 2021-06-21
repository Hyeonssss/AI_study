# Pday03_NLP



```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

SUBMIT = False

if SUBMIT:
    train = pd.read_csv('./data/test.csv', usecols=['posting_id', 'title'])
else:
    train = pd.read_csv('./data/train.csv', usecols=['posting_id', 'title', 'label_group'])
    tmp = train.groupby('label_group').posting_id.agg('unique').to_dict()
    train['target'] = train.label_group.map(tmp)
```



```python
import string

title = train['title'].apply(lambda s : s.translate(
    str.maketrans(string.punctuation, ' ' * len(string.punctuation))))

tfidf_vec = TfidfVectorizer(stop_words='english',
                            binary=True,
                            max_features=30000)
text_embeddings = tfidf_vec.fit_transform(title).toarray().astype(np.float32)
```



```python
from tqdm.notebook import tqdm
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

text_tensor = torch.from_numpy(text_embeddings).to(device)

out_preds = []
chunk = 64
for i in tqdm(list(range(0, train.shape[0], chunk)) + [train.shape[0]-chunk]):
    arr = text_tensor[i : i + chunk] @ text_tensor.T
    
    indices = torch.nonzero(arr > 0.525)
    
    preds = dict()
    for k in range(arr.shape[0]):
        preds[k] = []
    for ind in range(indices.size(0)):
        preds[indices[ind, 0].item()].append(indices[ind, 1].item())
        
    out_preds.extend([(train.iloc[k].posting_id, train.iloc[v].
                       posting_id.tolist()) for k,v in preds.items()])

out_preds = out_preds[:train.shape[0]]
df = pd.DataFrame(out_preds, columns=['index','pred'])
df.set_index('index')

if not SUBMIT:
    df['true'] = train['target']
    
    f1 = []
    for index, row in df[['true', 'pred']].iterrows():
        f1.append((2 * len(set(row['true']) & set(row['pred']))) /
                 (len(row['true']) + len(row['pred'])))
        
    print(f'F1: {np.mean(f1)}')
else:
    df['posting_id'] = train['posting_id']
    df['matches'] = df['pred'].apply(lambda x : ' '.join(x))
    df[['posting_id', 'matches']].to_csv('submission.csv', index=False)
```

![image-20210413162536896](md-images/image-20210413162536896.png)