# 영어 - 독일어 번역 모델 생성 (seq2seq)
---
[참고블로그](https://codlingual.tistory.com/91)


```python
!pip install torchtext==0.4
```


```python
# 라이브러리 로딩
import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math
import time
```


```python
!python -m spacy download en

!python -m spacy download de
```

## Tokenizer


```python
# 각 언어에 맞는 tokenizer 불러오기

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')
```


```python
def tokenize_de(text):
    # 독일어 tokenize해서 단어들을 리스트로 만든 후 reverse 
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]
    
def tokenize_en(text):
    # 영어 tokenize해서 단어들을 리스트로 만들기
    return [tok.text for tok in spacy_en.tokenizer(text)]
```


```python
# Field 선언

#input
SRC  = Field(tokenize = tokenize_de, init_token= '<sos>', eos_token = '<eos>', lower =True)

#output
TRG  = Field(tokenize = tokenize_en, init_token= '<sos>', eos_token = '<eos>', lower =True)
```


```python
# exts : 어떤 언어 사용할지 명시 (input 언어를 먼저 씀)
# filed = (입력, 출력)

train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC,TRG))
```


```python
# 데이터 확인
# 독일 단어는 역순임.
print(vars(train_data.examples[0]))
```

    {'src': ['.', 'büsche', 'vieler', 'nähe', 'der', 'in', 'freien', 'im', 'sind', 'männer', 'weiße', 'junge', 'zwei'], 'trg': ['two', 'young', ',', 'white', 'males', 'are', 'outside', 'near', 'many', 'bushes', '.']}
    

## Vocab 


```python
#최소 2번은 등장해야 vocab에 포함

SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)
```


```python
# Iterator 
batch_size = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), batch_size=batch_size)
```

## 모델링

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

### Seq2seq


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
input_dim = len(SRC.vocab)
output_dim = len(TRG.vocab)

# Encoder embedding dim 
enc_emb_dim = 256

# Decoder embedding dim 
dec_emb_dim = 256

hid_dim = 512
n_layers = 2

enc_dropout = 0.5
dec_dropout = 0.5

enc = Encoder(input_dim, enc_emb_dim, hid_dim, n_layers,enc_dropout)
dec = Decoder(output_dim, dec_emb_dim, hid_dim, n_layers,dec_dropout)

device = torch.device('cuda')
model = Seq2Seq(enc, dec, device)
```

### 가중치초기화
(-0.08, 0.08) 범위의 정규분포에서 모든 가중치 초기화


```python
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
model = model.apply(init_weights)
```

### Opimizer/ Loss


```python
optimizer = optim.Adam(model.parameters())

# <pad> 토큰의 index를 넘겨 받으면 오차 계산하지 않고 ignore하기
# <pad> = padding
trg_pad_idx = TRG.vocab.stoi[TRG.pad_token]

criterion = nn.CrossEntropyLoss(ignore_index = trg_pad_idx)
```

## 학습 코드


```python
def train(model, iterator, optimizer, criterion, clip):
    """
    모델을 학습하는 코드
    """
    model.train()
    epoch_loss=0
    
    for i, batch in enumerate(iterator):
        src = batch.src # [25,128]
        trg = batch.trg # [29,128]
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]
        output_dim = output.shape[-1]
        
        # loss 함수는 2d input으로만 계산 가능 
        output = output[1:].view(-1, output_dim)
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
    """
    학습된 모델을 평가하는 코드
    """
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            
            # teacher_forcing_ratio = 0 (아무것도 알려주면 안 됨)
            output = model(src, trg, 0)
            
            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]
            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
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
import time
N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
```
