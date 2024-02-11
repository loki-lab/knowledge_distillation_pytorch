import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets


def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available")
        return "cuda"
    else:
        print("CUDA is not available")
        return "cpu"


def load_data(data_path, trans=None, train_size=None, val_size=None, batch_size=32):
    full_data = datasets.ImageFolder(data_path, transform=trans)
    train_ds, val_ds = random_split(full_data, [train_size, val_size])
    print("dataset train size: ", len(train_ds), "val size: ", len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader
