import torch
from torch import nn
from model_kit.models import VGG11, VGG16
from model_kit.trainer import KnowledgeDistillationTrainer
from torchvision import transforms
from torch.optim import Adam
from model_kit.utils import check_cuda, load_data

if __name__ == '__main__':
    t = 2
    alpha = 0.8

    weight = torch.load("checkpoints/teacher_model/best_weight.pt")

    data_path = "datasets/PetImg"
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

    train_ld, val_ld = load_data(data_path, trans=transform["train"], train_size=20000, val_size=5000, batch_size=16)
    teacher_model = VGG16(num_classes=2)
    teacher_model.load_state_dict(weight["model_state_dict"])
    student_model = VGG11(num_classes=2)
    print(student_model)
    criterion = nn.KLDivLoss(reduction="batchmean")
    trainer = KnowledgeDistillationTrainer(teacher_model=teacher_model,
                                           student_model=student_model,
                                           criterion=criterion,
                                           optimizer=Adam,
                                           device=check_cuda(),
                                           t=t,
                                           alpha=alpha)

    trainer.fit(train_loader=train_ld, test_loader=val_ld, epochs=25)
