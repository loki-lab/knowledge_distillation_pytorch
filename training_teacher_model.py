import torch
from torch import nn

from model_kit.models import VGG16
from model_kit.trainer import Trainer
from model_kit.metrics import accuracy
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim import Adam


def check_cuda():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def load_data(data_path, trans=None, train_size=None, val_size=None):
    full_data = datasets.ImageFolder(data_path, transform=trans)
    train_ds, val_ds = random_split(full_data, [train_size, val_size])
    print("dataset train size: ", len(train_ds), "train size: ", len(train_ds))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    return train_loader, val_loader


if __name__ == '__main__':
    dataset_path = "datasets/PetImg"
    transform = {"train": transforms.Compose([transforms.ToTensor(),
                                              transforms.Resize((224, 224), antialias=True),
                                              transforms.Normalize((0.5,), (0.5,)),
                                              transforms.RandomPerspective(distortion_scale=0.5, p=0.1),
                                              transforms.RandomRotation(degrees=(0, 180)),
                                              transforms.RandomHorizontalFlip(p=0.5),
                                              transforms.RandomVerticalFlip(p=0.5)]),
                 "test": transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((224, 224), antialias=True),
                                             transforms.Normalize((0.5,), (0.5,))])
                 }

    train_ld, val_ld = load_data(dataset_path, trans=transform["train"], train_size=20000, val_size=5000)
    model = VGG16(num_classes=2)
    print(model)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, criterion=criterion, optimizer=Adam, device=check_cuda(), metrics=accuracy)

    trainer.fit(train_loader=train_ld, test_loader=val_ld, epochs=25)
