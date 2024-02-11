import torch
import torch.nn.functional as f
from torch import nn
from tqdm import tqdm


class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer(model.parameters(), lr=0.0001)
        self.device = device

    def train(self, train_loader):
        self.model.train()
        running_loss = 0.0
        total_sample = 0.0
        total_correct = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(
                outputs, labels
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            running_loss += loss.item()
            total_sample += labels.size(0)

        total_loss = running_loss / total_sample

        accuracy = total_correct / total_sample

        print(f"Train set: Average loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

        return accuracy, total_loss

    def test(self, test_loader):
        with torch.no_grad():
            self.model.eval()
            running_loss = 0.0
            total_sample = 0.0
            total_correct = 0.0
            for inputs, labels in tqdm(test_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(
                    outputs, labels
                )

                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                running_loss += loss.item()
                total_sample += labels.size(0)

            total_loss = running_loss / total_sample

            accuracy = total_correct / total_sample

            print(f"Test set: Average loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

            return accuracy, total_loss

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
                self.save_checkpoint("./checkpoints/teacher_model/best_weight.pt", epoch, val_loss, val_acc)
                print("Saved best model")
                best_metrics = val_acc

            self.save_checkpoint("./checkpoints/teacher_model/latest_weight.pt", epoch, loss, acc)

        print("Finished training")


class KnowledgeDistillationTrainer(Trainer):
    def __init__(self, teacher_model, student_model, criterion, optimizer, device, t, alpha):
        super().__init__(model=student_model, criterion=criterion, optimizer=optimizer, device=device)
        self.teacher_model = teacher_model.to(device)
        self.t = t
        self.alpha = alpha

    def train(self, train_loader):
        self.model.train()
        self.teacher_model.eval()
        running_loss = 0.0
        total_correct = 0.0
        total_sample = 0.0
        total_correct_distill = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            labels_distill = self.teacher_model(inputs)
            outputs = self.model(inputs)
            distill_loss = (nn.KLDivLoss(reduction="batchmean")(f.log_softmax(outputs / self.t, dim=1),
                            f.softmax(labels_distill / self.t, dim=1)) * (self.alpha * self.t * self.t) +
                            f.cross_entropy(outputs, labels) * (1. - self.alpha))

            self.optimizer.zero_grad()
            distill_loss.backward()
            self.optimizer.step()

            _, predicted = torch.max(outputs, 1)
            _, distill_predicted = torch.max(labels_distill, 1)
            total_correct_distill += (predicted == distill_predicted).sum().item()
            total_correct += (predicted == labels).sum().item()
            running_loss += distill_loss.item()
            total_sample += labels.size(0)

        total_loss = running_loss / total_sample

        accuracy = total_correct / total_sample
        distill_accuracy = total_correct_distill / total_sample

        print(f"Train set: Average loss: {total_loss:.4f},"
              f" Accuracy: {accuracy:.4f}, "
              f" Distillation Accuracy: {distill_accuracy:4f}.")

        return total_loss, accuracy, distill_accuracy

    def fit(self, train_loader, test_loader, epochs):
        print("Start distillation")
        best_metrics = 0.0
        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch + 1, epochs))
            distill_loss, accuracy, distill_accuracy = self.train(train_loader)
            val_acc, val_loss = self.test(test_loader)
            if val_acc > best_metrics:
                self.save_checkpoint("./checkpoints/student_model/best_weight.pt", epoch, val_loss, val_acc)
                print("Saved best model")
                best_metrics = val_acc

            self.save_checkpoint("./checkpoints/student_model/latest_weight.pt", epoch, distill_loss, accuracy)

        print("Finish distillation")
