from old.test2 import Params
from old.test2 import padding_sentence, make_iter, build_dataset
from old.test2 import Trainer
import argparse
import pickle


def main(config):

    if config.mode == 'train':
        # build_tokenizer()  # tokenizer pickle dump
        build_dataset()

        params = Params('config/params.json')

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
        params = Params('config/params.json')

        test_inputs, test_labels = padding_sentence(params.max_sequence_length, 'test2')
        test_loader = make_iter(config.mode, params, test_inputs, test_labels)

        trainer = Trainer(config.mode, params, test_iter=test_loader)
        print('Test Model')
        trainer.inference()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN for Sentence Classification using NSMC Data set')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test2'])
    args = parser.parse_args()
    main(args)