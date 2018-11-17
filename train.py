import torch
import torch.nn as nn
from blocks import SetAttentionBlock, InducedSetAttentionBlock, PoolingMultiheadAttention


class SetTransformer(nn.Module):

    def __init__(self):
        super().__init__()
        self.mab = InducedSetAttentionBlock(d, m, num_heads)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        return self.mab(x, x)


def main():
    model = SetTransformer()
