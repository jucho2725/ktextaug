"""
tokenize

build_vocab
"""
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from soynlp.word import WordExtractor
from soynlp.tokenizer import MaxScoreTokenizer
import json
import pickle

import torch
import itertools
from collections import Counter
import re
from sklearn.model_selection import train_test_split


class Params:

    def __init__(self, json_path, data_name):

        self.json_path = json_path
        self.update(json_path)
        self.aug_build_vocab(data_name)
        self.set_cuda()

    def aug_build_vocab(self, data_name):

        data_dir = Path().cwd() / 'data'
        sentence_file = os.path.join(data_dir, data_name)
        test_file = os.path.join(data_dir, 'tok_test_all.csv')

        test_df = pd.read_csv(test_file, encoding='utf-8')
        df = pd.read_csv(sentence_file, encoding='utf-8')

        df = df.append(test_df, ignore_index=True)

        sentence_list = []
        header = df.columns
        df = df.drop(columns=header[0])
        for row in tqdm(df['review']):
            sentence_list.append(row.split('|'))

        sequence = list(map(len, sentence_list))
        df_sequence_length = pd.DataFrame(sequence)
        max_sequence_length = int(df_sequence_length.quantile(0.75))

        token_counter = Counter(itertools.chain.from_iterable(sentence_list))

        vocab = ['<PAD>'] + [word for word in token_counter.keys()]
        vocab_size = len(vocab)
        print('Total Vocab size : ', vocab_size)

        word_to_idx = {idx: word for word, idx in enumerate(vocab)}

        with open('pickles/vocab.pickle', 'wb') as vocabulary:
            pickle.dump(word_to_idx, vocabulary)

        params = {'max_sequence_length': max_sequence_length,
                  'vocab_size': vocab_size, 'pad_idx': word_to_idx['<PAD>']}

        self.__dict__.update(params)
        self.save(json_path=self.json_path)


    def set_cuda(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        params = {'device': device}
        self.__dict__.update(params)


    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)


    def save(self, json_path):
        with open(json_path, mode='w') as io:
            json.dump(self.__dict__, io, indent=4)


    @property
    def dict(self):
        return self.__dict__



# def build_tokenizer():
#
#     data_path = Path().cwd() / 'data'
#     corpus_file = os.path.join(data_path, 'ratings.txt')
#
#     df = pd.read_table(corpus_file, encoding='utf-8')
#     df['document'] = list(map(str, df['document']))
#
#     word_extractor = WordExtractor(min_frequency=10, min_cohesion_forward=0.05)
#     word_extractor.train(df['document'])
#
#     words = word_extractor.extract()
#     cohesion_score = {word: score.cohesion_forward for word, score in words.items()}
#
#     tokenizer = MaxScoreTokenizer(scores=cohesion_score)
#
#     with open('pickles/tokenizer.pickle', 'wb') as pickle_out:
#         pickle.dump(tokenizer, pickle_out)


def padding_sentence(max_sequence_length, mode):

    # pickle_tokenizer = open('pickles/tokenizer.pickle', 'rb')
    file_vocab = open('pickles/vocab.pickle', 'rb')
    vocab = pickle.load(file_vocab)
    # tokenizer =pickle.load(pickle_tokenizer)
    data_dir = Path().cwd() / 'data'

    if mode == 'train':
        corpus_file = os.path.join(data_dir, 'train.csv')
    elif mode == 'valid':
        corpus_file = os.path.join(data_dir, 'valid.csv')
    else:
        corpus_file = os.path.join(data_dir, 'tok_test_all.csv')

    df = pd.read_csv(corpus_file)
    label = torch.LongTensor(df['label'])
    df_length = len(df)

    sentence_list = []
    for row in tqdm(df['review']):
        sentence_list.append(row.split('|'))

    input_sentence = []
    vocab_list = vocab.keys()

    for i, row in enumerate(sentence_list):
        temp_list = []

        for word in row:
            word = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》;]','', word)
            if word in vocab_list:
                temp_list.append(vocab[word])

        row_length = len(temp_list)

        if row_length < max_sequence_length:
            for _ in range(max_sequence_length - row_length):
                temp_list.append(vocab['<PAD>'])
        elif row_length > max_sequence_length:
            temp_list = temp_list[:max_sequence_length]

        input_sentence.append(temp_list)

    input_sentence = np.array(input_sentence).reshape(-1, )
    input_sentence = torch.LongTensor(input_sentence).view(df_length, -1)
    input_package = (input_sentence, label)

    with open(f'pickles/padding_{mode}.pickle', 'wb') as pickle_out:
        pickle.dump(input_package, pickle_out)

    return input_sentence, label


def build_dataset(data_name):
    data_dir = Path().cwd() / 'data'
    train_file = os.path.join(data_dir, data_name)

    df = pd.read_csv(train_file, encoding='utf-8')
    train, valid = train_test_split(df, test_size=0.2, random_state=333)

    train.to_csv(data_dir / 'train.csv', index=False)
    valid.to_csv(data_dir / 'valid.csv', index=False)


def make_iter(mode, params, inputs, labels):

    data = torch.utils.data.TensorDataset(inputs, labels)
    if mode == 'train':
        shuffle = True
    else:
        shuffle = False

    data_loader = torch.utils.data.DataLoader(data, batch_size=params.batch_size, shuffle=shuffle)
    return data_loader