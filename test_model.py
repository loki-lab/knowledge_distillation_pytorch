from model_kit.models import VGG16
import torch

if __name__ == '__main__':
    weight = torch.load('./checkpoints/best_weight.pt')
    model = VGG16(num_classes=2).load_state_dict(weight['model_state_dict'])

    print(weight['epoch'])
    print(weight['loss'])
    print(weight['metrics'])
