# 데이터 불러오기


```python
import pandas as pd
from glob import glob
```


```python
data = glob('./한영번역/*.xlsx')
```


```python
data
```




    ['./한영번역/5_문어체_조례.xlsx',
     './한영번역/2_대화체.xlsx',
     './한영번역/1_구어체(2).xlsx',
     './한영번역/1_구어체(1).xlsx',
     './한영번역/3_문어체_뉴스(2).xlsx',
     './한영번역/3_문어체_뉴스(3).xlsx',
     './한영번역/3_문어체_뉴스(1).xlsx',
     './한영번역/4_문어체_한국문화.xlsx',
     './한영번역/3_문어체_뉴스(4).xlsx',
     './한영번역/6_문어체_지자체웹사이트.xlsx']




```python
df = pd.DataFrame(columns = ['원문','번역문'])
path = './한영번역/'

file_list = [ '2_대화체.xlsx',
 '1_구어체(2).xlsx',
 '1_구어체(1).xlsx',
 '3_문어체_뉴스(2).xlsx',
 '3_문어체_뉴스(3).xlsx',
 '3_문어체_뉴스(1).xlsx',
 '4_문어체_한국문화.xlsx',
 '3_문어체_뉴스(4).xlsx']

for data in file_list:
    temp = pd.read_excel(path+data)
    df = pd.concat([df,temp[['원문','번역문']]])
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>원문</th>
      <th>번역문</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>이번 신제품 출시에 대한 시장의 반응은 어떤가요?</td>
      <td>How is the market's reaction to the newly rele...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>판매량이 지난번 제품보다 빠르게 늘고 있습니다.</td>
      <td>The sales increase is faster than the previous...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>그렇다면 공장에 연락해서 주문량을 더 늘려야겠네요.</td>
      <td>Then, we'll have to call the manufacturer and ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>네, 제가 연락해서 주문량을 2배로 늘리겠습니다.</td>
      <td>Sure, I'll make a call and double the volume o...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>지난 회의 마지막에 논의했던 안건을 다시 볼까요?</td>
      <td>Shall we take a look at the issues we discusse...</td>
    </tr>
  </tbody>
</table>
</div>




```python
import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.legacy.datasets import TranslationDataset, Multi30k
from torchtext.legacy.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math
import time
```

## 토크나이저 만들기


```python
from torchtext.legacy import data 
from konlpy.tag import Okt

tokenizer = Okt()
```


```python

def tokenize_kor(text):
    """한국어를 tokenizer해서 단어들을 리스트로 만듦"""
    return [text_ for text_ in tokenizer.morphs(text)]

def tokenize_en(text):
    """영어를 split tokenizer해서 단어들을 리스트로 만듦"""
    return [text_ for text_ in text.split()]

# 필드 정의

SRC =data.Field(tokenize = tokenize_kor,
                init_token = '<sos>',
                eos_token = '<eos>',batch_first = True,lower = True)

TRG =data.Field(tokenize = tokenize_en,
                init_token = '<sos>',
                eos_token = '<eos>',batch_first = True,
                lower = True)
```

## 데이터 셋 만들기 (전체 데이터중 100,000개만 사용)


```python
df_shuffled=df.sample(frac=1).reset_index(drop=True)
```


```python
from sklearn.model_selection import KFold
df_ = df_shuffled[:10000]
kf = KFold(n_splits = 5, shuffle = True, random_state = 42)

for i,(trn_idx,val_idx) in enumerate(kf.split(df_['원문'])):
    trn = df_.iloc[trn_idx]
    val = df_.iloc[val_idx]
```


```python
print('trn size: ',len(trn))
print('val size: ',len(val))
```

    trn size:  8000
    val size:  2000
    


```python
trn.to_csv(path + 'trn.csv',index = False)
val.to_csv(path + 'val.csv',index = False)
```


```python
from torchtext.legacy.data import TabularDataset

train_data, validation_data =TabularDataset.splits(
     path='./한영번역/', train='trn.csv',validation= 'val.csv', format='csv',
        fields=[('원문', SRC), ('번역문', TRG)], skip_header=True)
```


```python
print('훈련 샘플의 개수 : {}'.format(len(train_data)))
print('검증 샘플의 개수 : {}'.format(len(validation_data)))
```

    훈련 샘플의 개수 : 8000
    검증 샘플의 개수 : 2000
    


```python
# 학습 데이터 중 하나를 선택해 출력
print(vars(train_data.examples[30])['원문'])
print(vars(train_data.examples[30])['번역문'])
```

    ['토트넘', '의', '손흥민', '선수', '가', '아시안컵', '합류', '전', '마지막', '경기', '인', '맨체스터', '유나이티드', '전', '에서', '풀타임', '을', '뛰었지만', ',', '팀', '패배', '를', '막지', '못', '했습니다', '.']
    ["tottenham's", 'son', 'heung-min', 'played', 'full', 'time', 'in', 'the', 'last', 'game', 'of', 'manchester', 'united', 'before', 'joining', 'the', 'asian', 'cup,', 'but', 'failed', 'to', 'stop', 'the', 'team.']
    

## Vocab


```python
SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)
```


```python
print(f"len(SRC): {len(SRC.vocab)}")
print(f"len(TRG): {len(TRG.vocab)}")
```

    len(SRC): 10759
    len(TRG): 10038
    

## data loader


```python
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)
BATCH_SIZE = 128
```

    cuda:1
    


```python
from torchtext.legacy.data import Iterator
train_iterator = Iterator(dataset = train_data, batch_size = BATCH_SIZE)
valid_iterator = Iterator(dataset = validation_data, batch_size = BATCH_SIZE)
```

## multi head attention
---
1. query 행렬과 key 행렬간의 내적을 계산
2. 1번 값에 key 행렬의 차원의 제곱근으로 나눔
3. 2번 값에 소프트맥스 함수를 적용해 정규화 진행
4. 3번 값에 value 행렬을 곱해 attention 출력


```python
import torch.nn as nn

class MultiHeadAttentionLayer(nn.Module):
    """
    encoder와 decoder의 multi head attention 부분
    
    임베딩 된 sequence + positional encoding (or 이전 layer의 output) 을 이용해 
    self attention 을 수행하고 다음 layer(residual, normalization)로 보냄
        
    output: [batch size, seq_len, hidden_size] -> input과 차원이 같이야 여러 layer를 쌓을수 있음.
    """
    def __init__(self, hidden_dim, n_heads, dropout_ratio, device):
        super().__init__()

        assert hidden_dim % n_heads == 0

        self.hidden_dim = hidden_dim # 임베딩 차원
        self.n_heads = n_heads # 헤드(head)의 개수: 서로 다른 어텐션(attention) 컨셉의 수
        self.head_dim = hidden_dim // n_heads # 각 헤드(head)에서의 임베딩 차원 -> 각 헤드의 차원 * 헤드의 hidden dim  = input hidden dim 이여야함

        self.fc_q = nn.Linear(hidden_dim, hidden_dim) # Query 값에 적용될 FC 레이어
        self.fc_k = nn.Linear(hidden_dim, hidden_dim) # Key 값에 적용될 FC 레이어
        self.fc_v = nn.Linear(hidden_dim, hidden_dim) # Value 값에 적용될 FC 레이어

        self.fc_o = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_ratio)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query,key, value, mask = None):

        batch_size = src.shape[0]

        # query : [batch_size, query_len, hidden_dim]
        # key : [batch_size, query_len, hidden_dim]
        # value : [batch_size, query_len, hidden_dim]
        
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q: [batch_size, query_len, hidden_dim]
        # K: [batch_size, key_len, hidden_dim]
        # V: [batch_size, value_len, hidden_dim]

        # hidden_dim → n_heads X head_dim 형태로 변형
        # n_heads(h)개의 서로 다른 어텐션(attention) 컨셉을 학습하도록 유도
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q: [batch_size, n_heads, query_len, head_dim]
        # K: [batch_size, n_heads, key_len, head_dim]
        # V: [batch_size, n_heads, value_len, head_dim]

        # Attention Energy 계산
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale # 기울기 소실 방지

        # energy: [batch_size, n_heads, query_len, key_len]

        # 마스크(mask)를 사용하는 경우
        if mask is not None:
            # 마스크(mask) 값이 0인 부분을 -1e10으로 채우기
            energy = energy.masked_fill(mask==0, -1e10)

        # 어텐션(attention) 스코어 계산: 각 단어에 대한 확률 값
        attention = torch.softmax(energy, dim=-1)

        # attention: [batch_size, n_heads, query_len, key_len]

        # 여기에서 Scaled Dot-Product Attention을 계산
        x = torch.matmul(self.dropout(attention), V)

        # x: [batch_size, n_heads, query_len, head_dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x: [batch_size, query_len, n_heads, head_dim]

        x = x.view(batch_size, -1, self.hidden_dim)

        # x: [batch_size, query_len, hidden_dim]

        x = self.fc_o(x)

        # x: [batch_size, query_len, hidden_dim]

        return x, attention
```

## feedforward layer
---
피드포워드 레이어는 2개의 dense layer과 ReLU함수로 구성됨.   



```python
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hidden_dim, pf_dim, dropout_ratio):
        super().__init__()

        self.fc_1 = nn.Linear(hidden_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):

        # x: [batch_size, seq_len, hidden_dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        # x: [batch_size, seq_len, pf_dim]

        x = self.fc_2(x)

        # x: [batch_size, seq_len, hidden_dim]

        return x
```

## transformer 의 encoder
---
1. 입력 sequence를 임베딩 + positional embedding 값을 첫번째 인코더의 입력값으로 넣음
2. 인코더 1은 입력값을 받아 multi head attention을 통해 attention 행렬 값 출력
3. 어텐션 행렬을 FeedForward로 입력
4. 그 다음 인코더 1의 출력값을 그 위에 있는 인코더(인코더 2)에 입력
5. 인코더2는 위 과정을 반복.


```python
class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, pf_dim, dropout_ratio, device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hidden_dim, pf_dim, dropout_ratio)
        self.dropout = nn.Dropout(dropout_ratio)

    
    def forward(self, src, src_mask):

        # src: [batch_size, src_len, hidden_dim]
        # src_mask: [batch_size, src_len]

        # self attention
        # 필요한 경우 마스크(mask) 행렬을 이용하여 어텐션(attention)할 단어를 조절 가능
        _src, _ = self.self_attention(src, src, src, src_mask) # 하나의 입력값을 통해 query, key, value 값 생성 , decoder는 다름.

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # src: [batch_size, src_len, hidden_dim]

        # position-wise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        # src: [batch_size, src_len, hidden_dim]

        return src
```


```python
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, n_heads, pf_dim, dropout_ratio, device, max_length=100):
        super().__init__()

        self.device = device
        
        self.tok_embedding = nn.Embedding(input_dim, hidden_dim)
        self.pos_embedding = nn.Embedding(max_length, hidden_dim)
        # layer를 쌓음
        self.layers = nn.ModuleList([EncoderLayer(hidden_dim, n_heads, pf_dim, dropout_ratio, device) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout_ratio)
        
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

    def forward(self, src, src_mask):

        # src: [batch_size, src_len]
        # src_mask: [batch_size, src_len]

        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        # pos: [batch_size, src_len]

        # 소스 문장의 임베딩과 위치 임베딩을 더한 것을 사용
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        # src: [batch_size, src_len, hidden_dim]

        # 모든 인코더 레이어를 차례대로 거치면서 순전파(forward) 수행
        for layer in self.layers:
            src = layer(src, src_mask)

        # src: [batch_size, src_len, hidden_dim]

        return src # 마지막 레이어의 출력을 반환
```

## transformer의 decoder
---
1. 디코더에 대한 입력 문장을 임베딩 행렬로 변환한 다음 위치 인코딩 정보를 추가하고 디코더(디코더 1)에 입력
2. 디코더는 입력을 가져와 마스크된 멀티 헤드 어텐션 레이어에 보내고, 출력으로 어텐션 행렬 M을 반환
3. `어텐션 행렬M`, `인코딩 표현 R`을 입력받아 멀티헤드 어텐션 레이어(인코더- 디코더 어텐션 레이어)에 값을 입력하고, 출력으로 새로운 어텐션 행렬 생성
4. 어텐션 행렬을 피드포워드 레이어에 입력하여 값을 출력
5. 디코더 1의 출력값을 다음 디코더(디코터 2)의 입력값으로 사용
6. 디코더 2는 해당 내용 반복하며, 타깃 문장에 대한 디코더 표현을 반환.


```python
class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, pf_dim, dropout_ratio, device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio, device)
        self.encoder_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hidden_dim, pf_dim, dropout_ratio)
        self.dropout = nn.Dropout(dropout_ratio)

    # 인코더의 출력 값(enc_src)을 어텐션(attention)하는 구조
    def forward(self, trg, enc_src, trg_mask, src_mask):

        # trg: [batch_size, trg_len, hidden_dim]
        # enc_src: [batch_size, src_len, hidden_dim]
        # trg_mask: [batch_size, trg_len]
        # src_mask: [batch_size, src_len]

        # self attention
        # 자기 자신에 대하여 어텐션(attention)
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        # trg: [batch_size, trg_len, hidden_dim]

        # encoder attention
        # 디코더의 쿼리(Query)를 이용해 인코더를 어텐션(attention)
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)

        # dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        # trg: [batch_size, trg_len, hidden_dim]

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        # trg: [batch_size, trg_len, hidden_dim]
        # attention: [batch_size, n_heads, trg_len, src_len]

        return trg, attention
```


```python
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, n_layers, n_heads, pf_dim, dropout_ratio, device, max_length=100):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hidden_dim)
        self.pos_embedding = nn.Embedding(max_length, hidden_dim)

        self.layers = nn.ModuleList([DecoderLayer(hidden_dim, n_heads, pf_dim, dropout_ratio, device) for _ in range(n_layers)])

        self.fc_out = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout_ratio)

        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        # trg: [batch_size, trg_len]
        # enc_src: [batch_size, src_len, hidden_dim]
        # trg_mask: [batch_size, trg_len]
        # src_mask: [batch_size, src_len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos: [batch_size, trg_len]

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        # trg: [batch_size, trg_len, hidden_dim]

        for layer in self.layers:
            # 소스 마스크와 타겟 마스크 모두 사용
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        # trg: [batch_size, trg_len, hidden_dim]
        # attention: [batch_size, n_heads, trg_len, src_len]

        output = self.fc_out(trg)

        # output: [batch_size, trg_len, output_dim]

        return output, attention
```


```python
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    # 소스 문장의 <pad> 토큰에 대하여 마스크(mask) 값을 0으로 설정
    def make_src_mask(self, src):

        # src: [batch_size, src_len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask: [batch_size, 1, 1, src_len]

        return src_mask

    # 타겟 문장에서 각 단어는 다음 단어가 무엇인지 알 수 없도록(이전 단어만 보도록) 만들기 위해 마스크를 사용
    def make_trg_mask(self, trg):

        # trg: [batch_size, trg_len]

        """ (마스크 예시)
        1 0 0 0 0
        1 1 0 0 0
        1 1 1 0 0
        1 1 1 0 0
        1 1 1 0 0
        """
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        # trg_pad_mask: [batch_size, 1, 1, trg_len]

        trg_len = trg.shape[1]

        """ (마스크 예시)
        1 0 0 0 0
        1 1 0 0 0
        1 1 1 0 0
        1 1 1 1 0
        1 1 1 1 1
        """
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()

        # trg_sub_mask: [trg_len, trg_len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask: [batch_size, 1, trg_len, trg_len]

        return trg_mask

    def forward(self, src, trg):

        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # src_mask: [batch_size, 1, 1, src_len]
        # trg_mask: [batch_size, 1, trg_len, trg_len]

        enc_src = self.encoder(src, src_mask)

        # enc_src: [batch_size, src_len, hidden_dim]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        # output: [batch_size, trg_len, output_dim]
        # attention: [batch_size, n_heads, trg_len, src_len]

        return output, attention
```

## 모델 학습


```python
# 하이퍼 파라미터 설정
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HIDDEN_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
```


```python
# 모델 정의
SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

# 인코더(encoder)와 디코더(decoder) 객체 선언
enc = Encoder(INPUT_DIM, HIDDEN_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
dec = Decoder(OUTPUT_DIM, HIDDEN_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)

# Transformer 객체 선언
model = Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
```


```python
# 가중치 초기화
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

model.apply(initialize_weights)
```




    Transformer(
      (encoder): Encoder(
        (tok_embedding): Embedding(10759, 256)
        (pos_embedding): Embedding(100, 256)
        (layers): ModuleList(
          (0): EncoderLayer(
            (self_attn_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (ff_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (self_attention): MultiHeadAttentionLayer(
              (fc_q): Linear(in_features=256, out_features=256, bias=True)
              (fc_k): Linear(in_features=256, out_features=256, bias=True)
              (fc_v): Linear(in_features=256, out_features=256, bias=True)
              (fc_o): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (positionwise_feedforward): PositionwiseFeedforwardLayer(
              (fc_1): Linear(in_features=256, out_features=512, bias=True)
              (fc_2): Linear(in_features=512, out_features=256, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): EncoderLayer(
            (self_attn_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (ff_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (self_attention): MultiHeadAttentionLayer(
              (fc_q): Linear(in_features=256, out_features=256, bias=True)
              (fc_k): Linear(in_features=256, out_features=256, bias=True)
              (fc_v): Linear(in_features=256, out_features=256, bias=True)
              (fc_o): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (positionwise_feedforward): PositionwiseFeedforwardLayer(
              (fc_1): Linear(in_features=256, out_features=512, bias=True)
              (fc_2): Linear(in_features=512, out_features=256, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (2): EncoderLayer(
            (self_attn_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (ff_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (self_attention): MultiHeadAttentionLayer(
              (fc_q): Linear(in_features=256, out_features=256, bias=True)
              (fc_k): Linear(in_features=256, out_features=256, bias=True)
              (fc_v): Linear(in_features=256, out_features=256, bias=True)
              (fc_o): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (positionwise_feedforward): PositionwiseFeedforwardLayer(
              (fc_1): Linear(in_features=256, out_features=512, bias=True)
              (fc_2): Linear(in_features=512, out_features=256, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (decoder): Decoder(
        (tok_embedding): Embedding(10038, 256)
        (pos_embedding): Embedding(100, 256)
        (layers): ModuleList(
          (0): DecoderLayer(
            (self_attn_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (enc_attn_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (ff_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (self_attention): MultiHeadAttentionLayer(
              (fc_q): Linear(in_features=256, out_features=256, bias=True)
              (fc_k): Linear(in_features=256, out_features=256, bias=True)
              (fc_v): Linear(in_features=256, out_features=256, bias=True)
              (fc_o): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (encoder_attention): MultiHeadAttentionLayer(
              (fc_q): Linear(in_features=256, out_features=256, bias=True)
              (fc_k): Linear(in_features=256, out_features=256, bias=True)
              (fc_v): Linear(in_features=256, out_features=256, bias=True)
              (fc_o): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (positionwise_feedforward): PositionwiseFeedforwardLayer(
              (fc_1): Linear(in_features=256, out_features=512, bias=True)
              (fc_2): Linear(in_features=512, out_features=256, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): DecoderLayer(
            (self_attn_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (enc_attn_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (ff_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (self_attention): MultiHeadAttentionLayer(
              (fc_q): Linear(in_features=256, out_features=256, bias=True)
              (fc_k): Linear(in_features=256, out_features=256, bias=True)
              (fc_v): Linear(in_features=256, out_features=256, bias=True)
              (fc_o): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (encoder_attention): MultiHeadAttentionLayer(
              (fc_q): Linear(in_features=256, out_features=256, bias=True)
              (fc_k): Linear(in_features=256, out_features=256, bias=True)
              (fc_v): Linear(in_features=256, out_features=256, bias=True)
              (fc_o): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (positionwise_feedforward): PositionwiseFeedforwardLayer(
              (fc_1): Linear(in_features=256, out_features=512, bias=True)
              (fc_2): Linear(in_features=512, out_features=256, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (2): DecoderLayer(
            (self_attn_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (enc_attn_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (ff_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (self_attention): MultiHeadAttentionLayer(
              (fc_q): Linear(in_features=256, out_features=256, bias=True)
              (fc_k): Linear(in_features=256, out_features=256, bias=True)
              (fc_v): Linear(in_features=256, out_features=256, bias=True)
              (fc_o): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (encoder_attention): MultiHeadAttentionLayer(
              (fc_q): Linear(in_features=256, out_features=256, bias=True)
              (fc_k): Linear(in_features=256, out_features=256, bias=True)
              (fc_v): Linear(in_features=256, out_features=256, bias=True)
              (fc_o): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (positionwise_feedforward): PositionwiseFeedforwardLayer(
              (fc_1): Linear(in_features=256, out_features=512, bias=True)
              (fc_2): Linear(in_features=512, out_features=256, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (fc_out): Linear(in_features=256, out_features=10038, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )




```python
import torch.optim as optim

# Adam optimizer로 학습 최적화
LEARNING_RATE = 0.0005
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# 뒷 부분의 패딩(padding)에 대해서는 값 무시
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
```


```python
# 모델 학습(train) 함수
def train(model, iterator, optimizer, criterion, clip):
    model.train() # 학습 모드
    epoch_loss = 0

    # 전체 학습 데이터를 확인하며
    for i, batch in enumerate(iterator):
        
        src = batch.원문.to(device)
        trg = batch.번역문.to(device)
    

        optimizer.zero_grad()

        # 출력 단어의 마지막 인덱스(<eos>)는 제외
        # 입력을 할 때는 <sos>부터 시작하도록 처리
        output, _ = model(src, trg[:,:-1])

        # output: [배치 크기, trg_len - 1, output_dim]
        # trg: [배치 크기, trg_len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        # 출력 단어의 인덱스 0(<sos>)은 제외
        trg = trg[:,1:].contiguous().view(-1)

        # output: [배치 크기 * trg_len - 1, output_dim]
        # trg: [배치 크기 * trg len - 1]

        # 모델의 출력 결과와 타겟 문장을 비교하여 손실 계산
        loss = criterion(output, trg)
        loss.backward() # 기울기(gradient) 계산

        # 기울기(gradient) clipping 진행
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # 파라미터 업데이트
        optimizer.step()

        # 전체 손실 값 계산
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

```


```python
# 모델 평가(evaluate) 함수
def evaluate(model, iterator, criterion):
    model.eval() # 평가 모드
    epoch_loss = 0


    # 전체 평가 데이터를 확인하며
    for i, batch in enumerate(iterator):
        src = batch.원문.to(device)
        trg = batch.번역문.to(device)

        # 출력 단어의 마지막 인덱스(<eos>)는 제외
        # 입력을 할 때는 <sos>부터 시작하도록 처리
        output, _ = model(src, trg[:,:-1])

        # output: [배치 크기, trg_len - 1, output_dim]
        # trg: [배치 크기, trg_len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        # 출력 단어의 인덱스 0(<sos>)은 제외
        trg = trg[:,1:].contiguous().view(-1)

        # output: [배치 크기 * trg_len - 1, output_dim]
        # trg: [배치 크기 * trg len - 1]

        # 모델의 출력 결과와 타겟 문장을 비교하여 손실 계산
        loss = criterion(output, trg)

        # 전체 손실 값 계산
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)
```


```python
import math
import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
```


```python
import time
import math
import random

N_EPOCHS = 10
CLIP = 1
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time() # 시작 시간 기록

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time() # 종료 시간 기록
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'transformer_german_to_english.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}')
    print(f'\tValidation Loss: {valid_loss:.3f} | Validation PPL: {math.exp(valid_loss):.3f}')
```

```
Epoch: 01 | Time: 1m 21s
	Train Loss: 6.062 | Train PPL: 429.227
	Validation Loss: 5.091 | Validation PPL: 162.625
Epoch: 02 | Time: 1m 18s
	Train Loss: 4.867 | Train PPL: 129.979
	Validation Loss: 4.585 | Validation PPL: 98.025
Epoch: 03 | Time: 1m 18s
	Train Loss: 4.251 | Train PPL: 70.178
	Validation Loss: 4.257 | Validation PPL: 70.629
Epoch: 04 | Time: 1m 18s
	Train Loss: 3.709 | Train PPL: 40.806
	Validation Loss: 4.054 | Validation PPL: 57.608
Epoch: 05 | Time: 1m 19s
	Train Loss: 3.248 | Train PPL: 25.739
	Validation Loss: 3.979 | Validation PPL: 53.444
Epoch: 06 | Time: 1m 19s
	Train Loss: 2.864 | Train PPL: 17.523
	Validation Loss: 3.973 | Validation PPL: 53.169
Epoch: 07 | Time: 1m 19s
	Train Loss: 2.537 | Train PPL: 12.636
	Validation Loss: 4.039 | Validation PPL: 56.793
Epoch: 08 | Time: 1m 18s
	Train Loss: 2.265 | Train PPL: 9.635
	Validation Loss: 4.111 | Validation PPL: 61.019
Epoch: 09 | Time: 1m 18s
	Train Loss: 2.040 | Train PPL: 7.694
	Validation Loss: 4.218 | Validation PPL: 67.921
Epoch: 10 | Time: 1m 18s
	Train Loss: 1.853 | Train PPL: 6.379
	Validation Loss: 4.330 | Validation PPL: 75.955
```
## 직접 텍스트 데이터를 입력하여 번역 수행
---
- transformer의 구조상 한번에 모든 단어를 예측 할 수없음
- `<eos>` 토큰이 나올 때 까지 계속 decoder layer를 반복해야함.


```python
# 번역(translation) 함수
def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50, logging=True):
    model.eval() # 평가 모드

    if isinstance(sentence, str):        
        tokens = [text_ for text_ in tokenizer.morphs(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # 처음에 <sos> 토큰, 마지막에 <eos> 토큰 붙이기
    tokens = ['<sos>'] + tokens + ['<eos>']
    
    # text -> token
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    # 소스 문장에 따른 마스크 생성
    src_mask = model.make_src_mask(src_tensor)

    # 인코더(endocer)에 소스 문장을 넣어 출력 값 구하기
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    # 처음에는 <sos> 토큰 하나만 가지고 있도록 하기
    trg_indexes = [trg_field.vocab.stoi['<sos>']]

    # 한 단어씩 생성
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        # 출력 문장에 따른 마스크 생성
        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        # 출력 문장에서 가장 마지막 단어만 사용
        pred_token = output.argmax(2)[:,-1].item()
        trg_indexes.append(pred_token) # 출력 문장에 더하기

        # <eos>를 만나는 순간 끝
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    # 각 출력 단어 인덱스를 실제 단어로 변환
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    # 첫 번째 <sos>는 제외하고 출력 문장 반환
    return trg_tokens[1:], attention
```


```python
example_idx = 10

src = input()

translation, attention = translate_sentence(src, SRC, TRG, model, device, logging=True)

print('')
print("모델 번역 결과:", " ".join(translation))

```
```
나는 제주도 여행을 계획 중이다.

모델 번역 결과: i am planning to travel abroad while traveling island. <eos>
```