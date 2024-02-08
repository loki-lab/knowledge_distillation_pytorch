import torch


class Trainer:
    def __init__(self, model, criterion, optimizer, device, metrics):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer(model.parameters(), lr=0.001)
        self.device = device
        self.metrics = metrics

    def train(self, train_loader):
        self.model.train()
        running_loss = 0.0
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

        total_loss = running_loss / len(train_loader.dataset)

        metrics = self.metrics(self.model, train_loader, self.device)

        print(f"Train set: Average loss: {total_loss:.4f}, Accuracy: {metrics:.4f}")

    def test(self, test_loader):
        with torch.no_grad():
            self.model.eval()
            running_loss = 0.0
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(
                    outputs, labels
                )

                running_loss += loss.item()

            total_loss = running_loss / len(test_loader.dataset)

            metrics = self.metrics(self.model, test_loader, self.device)

            print(f"Test set: Average loss: {total_loss:.4f}, Accuracy: {metrics:.4f}")

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

    def fit(self, train_loader, test_loader, epochs):
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            self.train(train_loader)
            self.test(test_loader)


class KnowledgeDistillationTrainer(Trainer):
    def __init__(self, teacher_model, student_model, criterion, optimizer, device):
        super().__init__(model=student_model, criterion=criterion, optimizer=optimizer, device=device)