from torch import nn
from model_kit.models import VGG16
from model_kit.trainer import Trainer
from torchvision import transforms
from torch.optim import Adam
from model_kit.utils import check_cuda, load_data


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
    trainer = Trainer(model, criterion=criterion, optimizer=Adam, device=check_cuda())

    trainer.fit(train_loader=train_ld, test_loader=val_ld, epochs=25)
