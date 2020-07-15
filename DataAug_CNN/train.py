import time

import torch
import torch.nn as nn
import torch.optim as optim

from model.cnn import SentenceCNN

class Trainer:
    def __init__(self, mode, params, train_iter=None, valid_iter=None, test_iter=None):
        self.params = params

        # Train mode
        if mode == 'train':
            self.train_iter = train_iter
            self.valid_iter = valid_iter

        # Test mode
        else:
            self.test_iter = test_iter
        self.model = SentenceCNN(self.params.num_classes, self.params.vocab_size, self.params.embedding_dim)
        self.model.to(self.params.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=params.learning_rate)
        # self.loss_fn = nn.BCELoss()
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn.to(self.params.device)

    def train(self):

        best_valid_loss = float('inf')

        for epoch in range(self.params.epoch):
            start = time.time()
            self.model.train()
            loss_sum = 0.0
            acc_sum = 0.0

            # print(len(self.train_iter))

            for data in self.train_iter:

                inputs, labels = data[0].to(self.params.device), data[1].to(self.params.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                # print(outputs.size())             # torch.Size([128, 2])
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                loss_sum += loss.item()

                predicted = torch.max(outputs.data, 1)[1]   # indices = tensor([1, 0, 0, ...])
                predicted.to(self.params.device)
                acc_sum += (predicted == labels).sum().item() # 맞춘 갯수


            train_loss = loss_sum / len(self.train_iter)
            valid_loss, valid_acc = self.evaluate()

            train_acc = acc_sum / (len(self.train_iter)*self.params.batch_size)

            elap = int(time.time() - start)
            elapsed = (elap//3600, (elap % 3600) // 60, str(int((elap % 3600) % 60)))

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), self.params.save_model)

            # print(f'Epoch: {epoch} | Elapsed time: {elapsed[0]:.0f}h {elapsed[1]:.0f}m {elapsed[2]}s')
            # print(f'\tTrain Loss: {train_loss:.3f} | Validation Loss: {valid_loss:.3f} | Train Acc: {train_acc:.3f} | Valid Acc: {valid_acc:.3f}')


    def evaluate(self):
        self.model.eval()
        valid_loss = 0
        valid_acc = 0

        for data in self.valid_iter:
            inputs, labels = data[0].to(self.params.device), data[1].to(self.params.device)

            output = self.model(inputs)

            loss = self.loss_fn(output, labels)
            valid_loss += loss.item()

            predicted = torch.max(output.data, 1)[1]  # indices = tensor([1, 0, 0, ...])
            predicted.to(self.params.device)
            valid_acc += (predicted == labels).sum().item()  # 맞춘 갯수

        return valid_loss / len(self.valid_iter) , valid_acc / (self.params.batch_size*(len(self.valid_iter)))


    def inference(self):
        self.model.load_state_dict(torch.load(self.params.save_model))
        self.model.eval()
        test_loss = 0
        acc_sum = 0

        for data in self.test_iter:
            inputs, labels = data[0].to(self.params.device), data[1].to(self.params.device)
            output = self.model(inputs)

            loss = self.loss_fn(output, labels)

            _, predicted = torch.max(output.data, 1)
            acc_sum += (predicted == labels).sum().item()
            test_loss += loss.item()

            test_acc = acc_sum / (len(self.test_iter)*self.params.batch_size)

        test_loss = test_loss / len(self.test_iter)
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f}')