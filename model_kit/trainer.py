import torch
import torch.nn.functional as F
from tqdm import tqdm

class Trainer:
    def __init__(self, model, criterion, optimizer, device, metrics):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer(model.parameters(), lr=0.001)
        self.device = device
        self.metrics = metrics

    def train(self, train_loader):
        self.model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
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

        return metrics, total_loss

    def test(self, test_loader):
        with torch.no_grad():
            self.model.eval()
            running_loss = 0.0
            for inputs, labels in tqdm(test_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(
                    outputs, labels
                )

                running_loss += loss.item()

            total_loss = running_loss / len(test_loader.dataset)

            metrics = self.metrics(self.model, test_loader, self.device)

            print(f"Test set: Average loss: {total_loss:.4f}, Accuracy: {metrics:.4f}")

            return metrics, total_loss

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
        print("Start training")
        best_metrics = 0.0
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            acc, loss = self.train(train_loader)
            val_acc, val_loss = self.test(test_loader)

            if val_acc > best_metrics:
                best_metrics = acc
                self.save_checkpoint("./checkpoints/best_weight", epoch, val_loss, val_acc)

            self.save_checkpoint("./checkpoints/latest_weight", epoch, loss, acc)

        print("Finished training")


class KnowledgeDistillationTrainer(Trainer):
    def __init__(self, teacher_model, student_model, criterion, optimizer, device, metrics, t, alpha):
        super().__init__(model=student_model, criterion=criterion, optimizer=optimizer, device=device, metrics=metrics)
        self.teacher_model = teacher_model
        self.t = t
        self.alpha = alpha

    def train(self, train_loader):
        self.model.train()
        self.teacher_model.eval()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            labels_distill = self.teacher_model(inputs)
            outputs = self.model(inputs)
            distill_loss = (self.criterion(F.log_softmax(outputs / self.t, dim=1),
                                           F.softmax(labels_distill / self.t, dim=1)) * (self.alpha * self.t * self.t) +
                            F.cross_entropy(outputs, labels) * (1. - self.alpha))

            running_loss += distill_loss.item()
            self.optimizer.zero_grad()
            distill_loss.backward()
            self.optimizer.step()

        total_loss = running_loss / len(train_loader.dataset)

        student_metrics = self.metrics(self.model, train_loader, self.device)
        teacher_metrics = self.metrics(self.teacher_model, train_loader, self.device)

        print(f"Distill set: "
              f"Distill loss: {total_loss:.4f}, "
              f"Teacher Accuracy: {teacher_metrics:.4f}, "
              f"Student Accuracy: {student_metrics:.4f}")
