import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import SetTransformer
from input_pipeline import get_random_datasets


BATCH_SIZE = 32
NUM_STEPS = 50000
EVAL_STEP = 5000
MODELS_DIR = 'models/run00/'
LOGS_DIR = 'summaries/run00/'
DEVICE = torch.device('cuda:0')
USE_FLOAT16 = False

K = 4  # number of components
MIN_SIZE = 100  # minimal number of points
MAX_SIZE = 500  # maximal number of points

if USE_FLOAT16:
    from apex import amp


class LogLikelihood(nn.Module):
    """
    The log-likelihood of a dataset X = {x_1, ..., x_n}
    generated from a Mixture of Gaussians with k components.
    All gaussians are assumed to have diagonal variance matrices.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, means, variances, pis):
        """
        `x` - a batch of datasets of equal size.
        `means, variances, pis` - parameters of distributions.

        Arguments:
            x: a float tensor with shape [b, n, c].
            means: a float tensor with shape [b, k, c].
            variances: a float tensor with shape [b, k, c],
                has positive values only.
            pis: a float tensor with shape [b, k],
                has positive values only.
        Returns:
            a float tensor with shape [b].
        """
        device = x.device
        c = x.size(2)

        EPSILON = torch.tensor(1e-8, device=device)
        PI = torch.tensor(3.141592653589793, device=device)

        variances += EPSILON
        pis += EPSILON

        x = x.unsqueeze(2)  # shape [b, n, 1, c]
        means = means.unsqueeze(1)  # shape [b, 1, k, c]
        variances = variances.unsqueeze(1)  # shape [b, 1, k, c]
        pis = pis.unsqueeze(1)  # shape [b, 1, k]

        x = x - means  # shape [b, n, k, c]
        x = - 0.5 * c * torch.log(2.0 * PI) - 0.5 * variances.log().sum(3) - 0.5 * (x.pow(2) / variances).sum(3)
        # it has shape [b, n, k], it represents log likelihood of multivariate normal distribution

        x = x + pis.log()
        # it has shape [b, n, k]

        return x.logsumexp(2).sum(1).mean(0)


def get_parameters(y):
    b = y.size(0)  # batch size
    y = torch.split(y, [2 * K, 2 * K, K], axis=1)
    means = y[0].view(b, K, 2)
    variances = y[1].exp().view(b, K, 2)
    pis = y[2].exp()  # shape [b, K]
    return means, variances, pis


def train_and_evaluate():

    val_datasets = []
    for _ in range(100):
        x, _ = get_random_datasets(BATCH_SIZE, K, MIN_SIZE, MAX_SIZE)
        val_datasets.append(x)

    num_steps = NUM_EPOCHS * (len(dataset) // BATCH_SIZE)

    writer = SummaryWriter(LOGS_DIR)
    model = SetTransformer(num_outputs=5 * K)
    model = model.train().to(DEVICE)
    criterion = LogLikelihood()

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-3)

    if USE_FLOAT16:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    for iteration in range(1, NUM_STEPS + 1):

        x, _ = get_random_datasets(BATCH_SIZE, K, MIN_SIZE, MAX_SIZE)
        # note that each iteration datasets have different size

        x = x.to(DEVICE)
        # it has shape [b, n, 2]

        start_time = time.perf_counter()
        optimizer.zero_grad()

        y = model(x)  # shape [b, 5 * K]
        means, variances, pis = get_parameters(y)
        loss = criterion(x, means, variances, pis)

        if USE_FLOAT16:
            with amp.scale_loss(loss, optimizer) as loss_scaled:
                loss_scaled.backward()
        else:
            loss.backward()

        optimizer.step()
        scheduler.step()

        step_time = time.perf_counter() - start_time
        step_time = round(1000 * step_time, 1)

        writer.add_scalar('loss', loss.item(), iteration)
        print(f'iteration {iteration}, time {step_time} ms')

        if iteration % EVAL_STEP == 0:
            loss = evaluate(model, criterion, val_datasets)
            writer.add_scalar('val_loss', loss.item(), iteration)
            path = os.path.join(MODELS_DIR, f'iteration_{iteration}.pth')
            torch.save(model.state_dict(), path)


def evaluate(model, criterion, val_datasets):

    model.eval()
    total_loss = 0.0

    for x in val_datasets:

        with torch.no_grad():

            x = x.to(DEVICE)
            y = model(x)

            means, variances, pis = get_parameters(y)
            loss = criterion(x, means, variances, pis)

        total_loss += loss.item()

    model.train()
    num_samples = BATCH_SIZE * len(val_datasets)
    return total_loss / num_samples


train_and_evaluate()
