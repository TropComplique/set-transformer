import torch
import torch.nn as nn


DEVICE = torch.device('cuda:0')


def train_and_evaluate(model):

    for epoch in range(num_epochs):

        train(model, optimizer, criterion, train_loader)
        evaluate(model, criterion, val_loader)

    torch.save(model.state_dict(), PATH)


def train(model, optimizer, criterion, train_loader):

    model.train()
    running_loss = 0.0
    num_samples = 0

    for batch in train_loader:

        features, labels = batch
        features = features.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.set_grad_enabled(True):
            outputs = model(features)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = features.size(0)
        running_loss += loss.item() * batch_size
        num_samples += batch_size

    epoch_loss = running_loss / num_samples
    return epoch_loss


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

    loss = total_loss / num_samples
    return loss
