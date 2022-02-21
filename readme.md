# 개요

한국어 번역 모델 toy project 입니다. 개인 공부 할 겸 재미로 만들어 보았습니다.  
사용한 데이터는 아래 링크에서 다운 받으 실 수 있습니다.

- [Ai hub의 기계독해 데이터 사용(한-영 번역)](https://aihub.or.kr/aidata/86)


# requirement
```python
torch == '1.10.0+cu111
torchtext == '0.11.0
```

# how to use
```python
from utils.translate import TranslateSentence

# 모델 선언 코드는 transformer_training_inference.ipynb에 있음.

model = Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
model.load_state_dict(torch.load('transformer_korea_to_english.pt'))

src = input()

translation, attention = TranslateSentence(
    src, SRC, TRG, model, device, logging=True)

print('번역 할 문장 : ')
print("모델 번역 결과:", " ".join(translation))
```
```
번역 할 문장 : 나는 제주도 여행을 계획중입니다.
모델 번역 결과: i am planning to travel on jeju island. <eos>
```

# Reference
  - https://codlingual.tistory.com/91
  - https://github.com/graykode/nlp-tutorial
  - https://github.com/ndb796
  - https://github.com/YutaroOgawa/pytorch_advanced
