import torch
import torch.nn as nn


DEVICE = torch.device('cuda:0')


def train(model, optimizer, scheduler, criterion, train_loader):

    model.train()
    total_loss = 0.0
    num_samples = 0

    for batch in train_loader:

        features, labels = batch
        features = features.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        outputs = model(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size

    return total_loss / num_samples


def evaluate(model, criterion, val_loader):

    model.eval()
    total_loss = 0.0
    num_samples = 0

    for batch in val_loader:

        features, labels = batch
        features = features.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.set_grad_enabled(False):
            outputs = model(features)
            loss = criterion(outputs, labels)

        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size

    return total_loss / num_samples
