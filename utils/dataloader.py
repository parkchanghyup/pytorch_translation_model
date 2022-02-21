#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from glob import glob
from torchtext import data
from konlpy.tag import Okt
from torchtext.data import Iterator


def Get_Dataset_and_Field(data_size=100000, batch_size=256):
    """
    데이터를 불러와 dataloader를 반환하는 함수
    """

    def data_concat():
        # 전체 데이터 중 문어체_조레, 문어제_지자체웝사이트 파일 제외하고 데이터 프레임 생성
        data = glob('*.xlsx')

        df = pd.DataFrame(columns=['원문', '번역문'])

        file_list = ['2_대화체.xlsx',
                     '1_구어체(2).xlsx',
                     '1_구어체(1).xlsx',
                     '3_문어체_뉴스(2).xlsx',
                     '3_문어체_뉴스(3).xlsx',
                     '3_문어체_뉴스(1).xlsx',
                     '4_문어체_한국문화.xlsx',
                     '3_문어체_뉴스(4).xlsx']

        for data in file_list:
            temp = pd.read_excel(data)
            df = pd.concat([df, temp[['원문', '번역문']]])

        return df

    def random_data_extraction(df, data_size):
        """
        데이터 중 10만개만 사용
        """

        df_shuffled = df.sample(frac=1).reset_index(drop=True)

        df_ = df_shuffled[:data_size]

        train_df = df_[:data_size]
        test_df = df_[data_size+5000:]
        print('trn size: ', len(train_df))
        print('test size: ', len(test_df))

        train_df.to_csv('train_df.csv', index=False)
        test_df.to_csv('test_df.csv', index=False)

    def tokenize_kor(text):
        """
        한국어를 tokenizer해서 단어들을 리스트로 만든 후 reverse하여 반환
        """
        return [text_ for text_ in tokenizer.morphs(text)][::-1]

    def tokenize_en(text):
        """
        영어를 split tokenizer해서 단어들을 리스트로 만드는 함수
        """
        return [text_ for text_ in text.split()]

    df = data_concat()
    random_data_extraction(df, data_size)
    tokenizer = Okt()

    # 필드 정의
    SRC = data.Field(tokenize=tokenize_kor,
                     init_token='<sos>',
                     eos_token='<eos>', batch_first=True, lower=True)

    TRG = data.Field(tokenize=tokenize_en,
                     init_token='<sos>',
                     eos_token='<eos>', batch_first=True,
                     lower=True)

    # 데이터 불러오기
    train_data, test_data = TabularDataset.splits(
        path='', train='train_df.csv', test='test_df.csv', format='csv',
        fields=[('원문', SRC), ('번역문', TRG)], skip_header=True)

    # 학습 데이터와 검증데이터셋 분리
    train_data, validation_data = train_data.split(
        split_ratio=0.8, random_state=random.seed(323))

    print('훈련 샘플의 개수 : {}'.format(len(train_data)))
    print('검증 샘플의 개수 : {}'.format(len(validation_data)))

    # 말뭉치(vocab) 생성
    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)
    print(f"len(SRC): {len(SRC.vocab)}")
    print(f"len(TRG): {len(TRG.vocab)}")

    # 데이터 로더 생성
    train_iterator = Iterator(dataset=train_data, batch_size=batch_size)
    valid_iterator = Iterator(dataset=validation_data, batch_size=batch_size)
    test_iterator = Iterator(dataset=test_data, batch_size=batch_size)
    print('훈련 데이터의 미니 배치 수 : {}'.format(len(train_iterator)))
    print('검증 데이터의 미니 배치 수 : {}'.format(len(valid_iterator)))

    return train_iterator, valid_iterator, test_iterator, SRC, TRG

