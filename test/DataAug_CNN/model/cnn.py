import torch
import torch.nn as nn


from model.layers import ConvolutionLayer
from model.layers import MaxPooing


class SentenceCNN(nn.Module):
    def __init__(self, num_classes: int, vocab_size: int, embedd_dim: int) -> None:
        super(SentenceCNN, self).__init__()
        self._vocab_size = vocab_size
        self._embedding_size = embedd_dim
        self._num_class = num_classes

        self._embedding = nn.Embedding(self._vocab_size, self._embedding_size)
        self._convolution = ConvolutionLayer(1, 100, self._embedding_size)
        self._pooling = MaxPooing()
        self._dropout = nn.Dropout()
        self._fc = nn.Linear(3*100, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._embedding(x)  # x : (batch_size, sequence_length, embedding_size)
        x = x.unsqueeze(1)  # x : (batch_size, 1, sequence_length, embedding_size)
        x = self._convolution(x)  # x : Tuple[(batch_size, out_channels, sequence_length-kernel_size+1), (...), (...)]
        x = self._pooling(x)  # x : (batch_size, 3*out_channel)
        x = self._dropout(x)
        x = self._fc(x)

        return x