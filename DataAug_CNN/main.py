from utils import Params
from utils import padding_sentence, make_iter, build_dataset
from train import Trainer
from pathlib import Path
import argparse
import pickle
import os
import copy


def main(config, data_name):

    if config.mode == 'train':
        # build_tokenizer()  # tokenizer pickle dump
        build_dataset(data_name)

        params = Params('config/params.json', data_name)

        padding_sentence(params.max_sequence_length, 'train')
        padding_sentence(params.max_sequence_length, 'valid')

        train_inputs = open('pickles/padding_train.pickle', 'rb')
        train_inputs, train_labels = pickle.load(train_inputs)

        valid_inputs = open('pickles/padding_valid.pickle','rb')
        valid_inputs, valid_labels = pickle.load(valid_inputs)

        train_loader = make_iter(config.mode, params, train_inputs, train_labels)
        valid_loader = make_iter(config.mode, params, valid_inputs, valid_labels)

        trainer = Trainer(config.mode, params, train_iter=train_loader, valid_iter=valid_loader)
        print('Start Model Training')
        trainer.train()

    else:
        params = Params('config/params.json', data_name)

        test_inputs, test_labels = padding_sentence(params.max_sequence_length, 'test')
        test_loader = make_iter(config.mode, params, test_inputs, test_labels)

        trainer = Trainer(config.mode, params, test_iter=test_loader)
        print('Test Model')
        trainer.inference()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN for Sentence Classification using NSMC Data set')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    args = parser.parse_args()
    args_test = copy.deepcopy(args)
    args_test.mode = 'test'
    data_dir = Path().cwd() / 'data'
    data_list = os.listdir(data_dir)
    data_list.pop(data_list.index('tok_test_all.csv'))

    if 'train.csv' in data_list:
        data_list.pop(data_list.index('train.csv'))

    if 'valid.csv' in data_list:
        data_list.pop(data_list.index('valid.csv'))

    exp_list = copy.deepcopy(data_list)
    print('data list : ', data_list)
    for data in exp_list:
        print('### Experiment :', data)
        main(args, data)
        main(args_test, data)