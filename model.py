import torch.nn as nn
from blocks import SetAttentionBlock
from blocks import InducedSetAttentionBlock
from blocks import PoolingMultiheadAttention


class SetTransformer(nn.Module):

    def __init__(self, in_dimension, out_dimension):
        """
        Arguments:
            in_dimension: an integer.
            out_dimension: an integer.
        """
        super().__init__()

        d = 128
        m = 16  # number of inducing points
        h = 4  # number of heads
        k = 4  # number of seed vectors

        self.embed = nn.Sequential(
            nn.Linear(in_dimension, d),
            nn.ReLU(inplace=True)
        )
        self.encoder = nn.Sequential(
            InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d)),
            InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d))
        )
        self.decoder = nn.Sequential(
            PoolingMultiheadAttention(d, k, h, RFF(d)),
            SetAttentionBlock(d, h, RFF(d))
        )
        self.predictor = nn.Linear(k * d, out_dimension)

        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.apply(weights_init)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, in_dimension].
        Returns:
            a float tensor with shape [b, out_dimension].
        """

        x = self.embed(x)  # shape [b, n, d]
        x = self.encoder(x)  # shape [b, n, d]
        x = self.decoder(x)  # shape [b, k, d]

        b, k, d = x.shape
        x = x.view(b, k * d)
        return self.predictor(x)


class RFF(nn.Module):
    """
    Row-wise FeedForward layers.
    """
    def __init__(self, d):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(inplace=True),
            nn.Linear(d, d), nn.ReLU(inplace=True),
            nn.Linear(d, d), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        return self.layers(x)
