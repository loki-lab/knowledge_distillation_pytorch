import torch


def accuracy(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        model.eval()
        for data in data_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            correct = predicted.eq(labels).sum().item()
            total = labels.size(0)

        acc = 100 * correct / total
    return acc
