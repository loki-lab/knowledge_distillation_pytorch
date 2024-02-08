from torch import nn
import torch


class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer(model.parameters(), lr=0.001)
        self.device = device

    def train(self, train_loader):
        self.model.train()
        running_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(
                outputs, labels
            )

            running_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss

    def test(self, test_loader):
        with torch.no_grad():
            self.model.eval()
            running_loss = 0
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(
                    outputs, labels
                )

                running_loss += loss.item()

                return loss

    def predict(self, image):
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(image)
            _, predicted = torch.max(outputs.data, 1)
            return predicted

    def save_checkpoint(self, path, epoch, loss, metrics):

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics
        }, path)

    def fit(self, train_loader, test_loader):
        loss = self.train(train_loader)
        val_loss = self.test(test_loader)
