# 한국어 번역 모델 toy project

- [Ai hub의 기계독해 데이터 사용(한-영 번역)](https://aihub.or.kr/aidata/86)
- 참고 블로그 및 github
    - https://codlingual.tistory.com/91
    - https://github.com/graykode/nlp-tutorial
    - https://github.com/ndb796



## 1일차

- torchtext 사용법 익히기
- 네이버 영화 평점 분류 모델 설계 (gru 활용)

## 2일차

- 데이콘 뉴스 토픽 분류 모델 개발
    - test data의 data loader르 만드는 과정에서 `shuffle = True`로 선언하여 많이 헤맴. . 
    - gru로 모델 설계하여 정확도 75% 정도 되는 모델 설계 완료
- 독일어 -> 영어 번역 모델 설계 (seq2seq)

## 3일차

- rnn 계열 모델에 대한 이해가 부족하다고 판단 -> rnn, lstm 구현하는 것 부터 연습
- 데이콘 뉴스 토픽 분류 모델 
    - lstm으로 설계.

## 4일차
- 독일어 -> 영어 번역 모델 설계 완료 (seq2seq, gru)

## 5일차
- 한국어 -> 영어 번역 모델 설계 완료 (seq2seq, gru)
- 모델 학습 중 -> 2~3일 소요 예상

## 6일차
- 학습 된 모델을 바탕으로 번역 수행 했지만 생각보다 결과물이 좋지 않음
- transformer 모델 설계 후 해당 모델로 결과물 다시 볼 예정.

## 7일차
- transformer 구조에 대한 이해와 [나동빈님 코드](https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice/blob/master/code_practices/Attention_is_All_You_Need_Tutorial_(German_English).ipynb)를 바탕으로 한국어 영어 번역 모델 개발
- 학습 중.

## 8일차
- transformer 모델 활용하여 번역모델 설계완료
    - 성능을 최적화 시키기 위해 파라미터 튜닝이나, 모델 구조에 특정 기법을 적용하지 않음.
    - transformer의 구조를 익히는데 목적을 둔 코드.
- 생각보다 번역이 잘됨 -> 번역 task는 그렇게 어려운 task가 아니라고 함. :[
