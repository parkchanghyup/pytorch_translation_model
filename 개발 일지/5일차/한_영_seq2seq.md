# 데이터 병합


```python
import pandas as pd
from glob import glob
```


```python
data = glob('*.xlsx')
data
```




    ['5_문어체_조례.xlsx',
     '2_대화체.xlsx',
     '1_구어체(2).xlsx',
     '1_구어체(1).xlsx',
     '3_문어체_뉴스(2).xlsx',
     '3_문어체_뉴스(3).xlsx',
     '3_문어체_뉴스(1).xlsx',
     '4_문어체_한국문화.xlsx',
     '3_문어체_뉴스(4).xlsx',
     '6_문어체_지자체웹사이트.xlsx']




```python
# 전체 데이터 중 문어체_조레, 문어제_지자체웝사이트 파일 제외하고 데이터 프레임 생성

df = pd.DataFrame(columns = ['원문','번역문'])

file_list = [ '2_대화체.xlsx',
 '1_구어체(2).xlsx',
 '1_구어체(1).xlsx',
 '3_문어체_뉴스(2).xlsx',
 '3_문어체_뉴스(3).xlsx',
 '3_문어체_뉴스(1).xlsx',
 '4_문어체_한국문화.xlsx',
 '3_문어체_뉴스(4).xlsx']

for data in file_list:
    temp = pd.read_excel(data)
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
    """한국어를 tokenizer해서 단어들을 리스트로 만든 후 reverse"""
    return [text_ for text_ in tokenizer.morphs(text)][::-1]

def tokenize_en(text):
    """영어를 split tokenizer해서 단어들을 리스트로 만듦"""
    return [text_ for text_ in text.split()]

# 필드 정의

SRC = data.Field(tokenize = tokenize_kor,
                init_token = '<sos>',
                eos_token = '<eos>')

TRG = data.Field(tokenize = tokenize_en,
                init_token = '<sos>',
                eos_token = '<eos>',
                lower = True)

```

# 데이터셋 만들기 (전체 데이터중 100,000개만 사용)



```python
df
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
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>199995</th>
      <td>복무기간 단축안은 10월 전역자부터 2주 단위로 하루씩 단축해 육군·해병대·의무경찰...</td>
      <td>The proposed reduction of the service period w...</td>
    </tr>
    <tr>
      <th>199996</th>
      <td>실제로 이번 인사에서 인천지검 특수부장, 서울중앙지검 증권범죄합동수사단장 등을 지낸...</td>
      <td>In fact, the vice chief of the Seoul Eastern D...</td>
    </tr>
    <tr>
      <th>199997</th>
      <td>29일 서울 서초구 한국산업기술진흥협회 중회의실에서 열린 ‘이공계 우수인재 양성 및...</td>
      <td>On the 29th, at a meeting of experts on "how t...</td>
    </tr>
    <tr>
      <th>199998</th>
      <td>광주시교육청은 “지난 1일과 2일 한유총 광주지회로부터 장휘국 교육감 면담 요청이 ...</td>
      <td>The Gwangju Office of Education said, “There w...</td>
    </tr>
    <tr>
      <th>199999</th>
      <td>과학기술정보통신부 과학기술혁신본부장에는 김성수(58) 한국화학연구원장, 행정안전부 ...</td>
      <td>The appointment for the head of the Science an...</td>
    </tr>
  </tbody>
</table>
<p>1402033 rows × 2 columns</p>
</div>




```python
df_shuffled=df.sample(frac=1).reset_index(drop=True)
```


```python
from sklearn.model_selection import KFold
# 우선 전제 데이터중 10만개만 사용
df_ = df_shuffled[:100000]
kf = KFold(n_splits = 5, shuffle = True, random_state = 42)

for i,(trn_idx,val_idx) in enumerate(kf.split(df_['원문'])):
    trn = df_.iloc[trn_idx]
    val = df_.iloc[val_idx]
```


```python
print('trn size: ',len(trn))
print('val size: ',len(val))
```

    trn size:  80000
    val size:  20000
    


```python
trn.to_csv('trn.csv',index = False)
val.to_csv('val.csv',index = False)
```


```python
from torchtext.legacy.data import TabularDataset

train_data, validation_data =TabularDataset.splits(
     path='', train='trn.csv',validation= 'val.csv', format='csv',
        fields=[('원문', SRC), ('번역문', TRG)], skip_header=True)
```


```python
print('훈련 샘플의 개수 : {}'.format(len(train_data)))
print('검증 샘플의 개수 : {}'.format(len(validation_data)))
```

### Vocab 


```python
# 말뭉치 생성
SRC.build_vocab(train_data, min_freq = 2, max_size = 50000)
TRG.build_vocab(train_data, min_freq = 2, max_size = 50000)
```

### data loader 


```python
# data loader 생성
from torchtext.legacy.data import Iterator

# 하이퍼파라미터
batch_size = 128
lr = 0.001
EPOCHS = 20

train_loader = Iterator(dataset = train_data, batch_size = batch_size)
val_loader = Iterator(dataset = validation_data, batch_size = batch_size)
```


```python
print('훈련 데이터의 미니 배치 수 : {}'.format(len(train_loader)))
print('검증 데이터의 미니 배치 수 : {}'.format(len(val_loader)))
```

    훈련 데이터의 미니 배치 수 : 625
    검증 데이터의 미니 배치 수 : 157
    

# 모델 설계

### encoder
- 2 layer RNN
- back-bone으로 GRU 사용
- Layer 1: 독일어 토큰의 임베딩을 입력으로 받고 은닉상태 출력
- Layer 2 : Layer1의 은닉상태를 입력으로 받고 새로운 은닉상태 출력
- 각 layer마다 초기 은닉상태 h_0 필요 (0으로 초기화 ?)
- 각 layer마다 context vector 'z'를 출력


```python
# encoder 


class Encoder(nn.Module):
    """
    seq2seq의 encoder

    input_dim : input 데이터의 vocab size 
    단어들의 index가 embedding 함수로 넘겨짐

    emb_dim : embedding layer의 차원
    embedding 함수 : one-hot vector를 emb_dim 길이의 dense vector로 변환
    hid_dim : 은닉 상태의 차원
    n_layers : RNN 안의 레이어 개수 (여기선 2개)
    dropout : 사용할 드롭아웃의 양 (오버피팅 방지하는 정규화 방법)
    """
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout = 0.2):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU( emb_dim, hid_dim, n_layers, dropout = dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        
        #src = [src len, batch_size)]
        embedded = self.dropout(self.embedding(src))
        #embeded = [src len, batch size, emb dim]
        outputs, hidden = self.rnn(embedded)
        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]

        
        return hidden

```

### decoder

- Layer 1 : 직전 time-stamp로 부터 은닉 상태(s)와 cell state를 받고, 이들과 embedded token인 y_t를 입력으로 받아 새로운 은닉상태와 cell state를 만들어냄
- Layer 2 : Layer 2의 은닉 상태(s)와 Layer 2에서 직전 time-stamp의 은닉 상태(s)와 cell state를 입력으로 받아 새로운 은닉 상태와 cell state를 만들어냄
- Decoder Layer1의 `첫 은닉상태(s)와 cell state` = `context vector (z)` = `Encoder Layer 1의 마지막 은닉상태(h)와 cell state`
- Decoder RNN/LSTM의 맨 위 Layer의 은닉 상태를 Linear Layer인 f에 넘겨서 다음 토큰이 무엇일지 예측함
- 여기서는 GRU를 사용했기 때문에 cell state는 없음.
 


```python
class Decoder(nn.Module) : 
    """
    seq2seq의 Decoder

    output_dim : eocoder에서 넘어온 output_dim
    
    emb_dim : embedding layer의 차원
        hid_dim : 은닉 상태의 차원
    n_layers : RNN 안의 레이어 개수 (여기선 2개)
    dropout : 사용할 드롭아웃의 양 (오버피팅 방지하는 정규화 방법)
    """
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # Decoder에서 항상 n directions = 1
        # 따라서 hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]
        
        # input = [1, batch size]
        input = input.unsqueeze(0)
        
        # embedded = [1, batch size, emb dim]
        embedded = self.dropout(self.embedding(input))
        
        output, hidden = self.rnn(embedded, hidden)
        
        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
 
        # Decoder에서 항상 seq len = n directions = 1 
        # 한 번에 한 토큰씩만 디코딩하므로 seq len = 1
        # 따라서 output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        
        # prediction = [batch size, output dim]
        prediction = self.fc_out(output.squeeze(0))
        
        return prediction, hidden
        

```


```python
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        # Encoder와 Decoder의 hidden dim이 같아야 함 
        assert encoder.hid_dim == decoder.hid_dim, "encoder와 decoder의 hidden dim이 다름."
        assert encoder.n_layers == decoder.n_layers, "encoder와 decoder의 n_layers이 다름."

    def forward(self, src, trg ,teacher_forcing_ratio = 0.5):

        # src = [src len, batch size]
        # trg = [trg len, batch size]

        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        # decoder 결과를 저장할 텐서
        outputs = torch.zeros([trg_len,batch_size , trg_vocab_size])

        # encoder의 마지막 은닉 상태가 Deocder의 초기 은닉상태로 쓰임
        hidden = self.encoder(src)

        # decoder에 들어갈 첫 input은 <sos>토큰
        input = trg[0,:]

        #target length만큼 반복
        # range(0,trg_len)이 아니라 range(1,trg_len)인 이유 : 0번째 trg는 항상 <sos>라서 그에 대한 output도 항상 0 
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[t] = output

            teacher_force = random.random() < teacher_forcing_ratio

            # 확률 가장 높게 예측한 토큰
            top1 = output.argmax(1)

            #teacher_force = 1 = true 면 trg[t]를 아니면 top1을 input으로 사용
            input = trg[t] if teacher_force else top1
        return outputs


```


```python
device = torch.device('cuda:1')
```


```python
input_dim = len(SRC.vocab)
output_dim = len(TRG.vocab)

# Encoder embedding dim
enc_emb_dim = 256
# Decoder embedding dim
dec_emb_dim = 256

hid_dim=512
n_layers=2

enc_dropout = 0.5
dec_dropout=0.5

enc = Encoder(input_dim, enc_emb_dim, hid_dim, n_layers, enc_dropout)
dec = Decoder(output_dim, dec_emb_dim, hid_dim, n_layers, dec_dropout)

model = Seq2Seq(enc, dec, device).to(device)
```


```python
optimizer = optim.Adam(model.parameters())

# <pad> 토큰의 index를 넘겨 받으면 오차 계산하지 않고 ignore하기
# <pad> = padding
trg_pad_idx = TRG.vocab.stoi[TRG.pad_token]

criterion = nn.CrossEntropyLoss(ignore_index = trg_pad_idx)
```


```python
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss=0
    
    for i, batch in enumerate(iterator):
        src = batch.원문.to(device)
        trg = batch.번역문.to(device)
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]
        output_dim = output.shape[-1]
        
        # loss 함수는 2d input으로만 계산 가능 
        output = output[1:].view(-1, output_dim).to(device)
        trg = trg[1:].view(-1)
        
        # trg = [(trg len-1) * batch size]
        # output = [(trg len-1) * batch size, output dim)]
        loss = criterion(output, trg)
        
        loss.backward()
        
        # 기울기 폭발 막기 위해 clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss+=loss.item()
        
    return epoch_loss/len(iterator)
```


```python
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.원문.to(device)
            trg = batch.번역문.to(device)
            
            # teacher_forcing_ratio = 0 (아무것도 알려주면 안 됨)
            output = model(src, trg, 0)
            
            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]
            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim).to(device)
            trg = trg[1:].view(-1)
            
            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]
            
            loss = criterion(output, trg)
            
            epoch_loss+=loss.item()
        
        return epoch_loss/len(iterator)
```


```python
# function to count training time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
```


```python
N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = train(model, train_loader, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, val_loader, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
```
