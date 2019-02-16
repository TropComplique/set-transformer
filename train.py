import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from blocks import SetAttentionBlock, InducedSetAttentionBlock, PoolingMultiheadAttention
from utils import train, evaluate, DEVICE
from input_pipeline import get_loaders


class SetTransformer(nn.Module):

    def __init__(self, m):
        super().__init__()
        self.encoder = nn.Sequantial(
            InducedSetAttentionBlock(
                d=128, m=m, h=4,
                first_rff=RFF(d=128),
                second_rff=RFF(d=128)
            ),
            InducedSetAttentionBlock(
                d=128, m=m, h=4,
                first_rff=RFF(d=128),
                second_rff=RFF(d=128)
            )
        )
        self.decoder = nn.Sequantial(
            PoolingMultiheadAttention(
                d=128, k=4, h=4,
                rff=RFF(d=128)
            ),
            SetAttentionBlock(
                d=128, h=4,
                rff=RFF(d=128)
            )
        )
        self.predictor = nn.Linear(4 * 128, 20)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, 20].
        """
        x = self.encoder(x)  # shape [b, n, d]
        x = self.decoder(x)  # shape [b, k, d]
        b, k, d = x.size
        x = x.view(b, k * d)
        x = self.predictor(x)
        return x


class RFF(nn.Module):

    def __init__(self, d):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(),
            nn.Linear(d, d), nn.ReLU(),
            nn.Linear(d, d), nn.ReLU(),
            nn.Linear(d, d), nn.ReLU()
        )

        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

        self.apply(weights_init)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        return self.layers(x)


class Criterion(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputs, labels):
        """
        Arguments:
            outputs: a float tensor with shape [b, n].
            labels: a float tensor with shape [b, n].
        Returns:
            a float tensor with shape [].
        """
        pass


def train_and_evaluate():

    BATCH_SIZE = 10
    NUM_EPOCHS = 10
    PATH = 'models/run00'

    train_loader, val_loader = get_loaders(BATCH_SIZE)
    model = SetTransformer(m=16).to(DEVICE)
    criterion = Criterion()

    optimizer = optim.Adam(lr=1e-3, params=model.parameters(), weight_decay=1e-4)
    num_steps_per_epoch = len(train_loader.dataset) // BATCH_SIZE
    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps_per_epoch * NUM_EPOCHS, eta_min=1e-6)

    for epoch in range(NUM_EPOCHS):

        train_loss = train(model, optimizer, scheduler, criterion, train_loader)
        val_loss = evaluate(model, criterion, val_loader)
        print('epoch:{0}, train:{1:.3f}, val:{2:.3f}'.format(epoch, train_loss, val_loss))

    torch.save(model.state_dict(), PATH)


train_and_evaluate()
