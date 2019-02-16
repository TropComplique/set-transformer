import torch
import torch.nn as nn
from attention import MultiheadAttention


class MultiheadAttentionBlock(nn.Module):

    def __init__(self, d, h, rff):
        """
        Arguments:
            d: an integer, input dimension.
            h: an integer, number of heads.
            rff: a module, row-wise feedforward layer.
                It takes a float tensor with shape [b, n, d] and
                returns a float tensor with the same shape.
        """
        super().__init__()

        self.multihead = MultiheadAttention(d, h)
        self.layer_norm1 = nn.LayerNorm(d)
        self.layer_norm2 = nn.LayerNorm(d)
        self.rff = rff

    def forward(self, x, y):
        """
        It is equivariant to permutations of the
        second dimension of tensor x (`n`).

        It is invariant to permutations of the
        second dimension of tensor y (`m`).

        Arguments:
            x: float tensors with shape [b, n, d].
            y: float tensors with shape [b, m, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        h = self.layer_norm1(x + self.multihead(x, y, y))
        return self.layer_norm2(h + self.rff(h))


class SetAttentionBlock(nn.Module):

    def __init__(self, d, h, rff):
        super().__init__()
        self.mab = MultiheadAttentionBlock(d, h, rff)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        return self.mab(x, x)


class InducedSetAttentionBlock(nn.Module):

    def __init__(self, d, m, h, first_rff, second_rff):
        super().__init__()
        self.mab1 = MultiheadAttentionBlock(d, h, first_rff)
        self.mab2 = MultiheadAttentionBlock(d, h, second_rff)
        self.inducing_points = nn.Parameter(torch.randn(m, d))

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        i = self.inducing_points.unsqueeze(0).repeat([b, 1, 1])  # shape [b, m, d]
        h = self.mab1(i, x)  # shape [b, m, d]
        return self.mab2(x, h)


class PoolingMultiheadAttention(nn.Module):

    def __init__(self, d, k, h, rff):
        super().__init__()
        self.mab = MultiheadAttentionBlock(d, h, rff)
        self.seed_vectors = nn.Parameter(torch.randn(k, d))

    def forward(self, z):
        """
        Arguments:
            z: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, k, d].
        """
        s = self.seed_vectors.unsqueeze(0).repeat([b, 1, 1])  # shape [b, k, d]
        return self.mab(s, z)
        # note that in the original paper
        # they return mab(s, rff(z))
