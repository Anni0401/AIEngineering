import torch



def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """ Trains the model for one epoch on the given dataloader. """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """ Evaluates the model on the given dataloader. """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy


