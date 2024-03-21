from model_kit.models import VGG16, VGG11
import torch

if __name__ == '__main__':
    weight = torch.load('checkpoints/teacher_model/best_weight.pt')
    model = VGG16(num_classes=2)
    model.load_state_dict(weight['model_state_dict'])
    torch.save(model.state_dict(), 'checkpoints/teacher_model/best_model.pt')
    print(weight['epoch'])
    print(weight['loss'])
    print(weight['metrics'])
