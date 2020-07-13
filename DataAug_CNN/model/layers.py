import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ConvolutionLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, embedd_dim: int) -> None:
        super(ConvolutionLayer, self).__init__()
        self._in_ch = in_ch
        self._out_ch = out_ch
        self._embedding_dim = embedd_dim

        self._tri_filter = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, self._embedding_dim))
        self._tetra_filter = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(4, self._embedding_dim))
        self._penta_filter = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(5, self._embedding_dim))


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        tri_map = F.relu(self._tri_filter(x))           # torch.Size([128, 100, 14, 1])
        tetra_map = F.relu(self._tetra_filter(x))       # torch.Size([128, 100, 13, 1])
        penta_map = F.relu(self._penta_filter(x))       # torch.Size([128, 100, 12, 1])

        return tri_map, tetra_map, penta_map

class MaxPooing(nn.Module):

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor :
        fmap = torch.cat([f_map.squeeze(3).max(dim=-1)[0] for f_map in x], dim=-1)
        return fmap

