import torch
from model_kit.models import VGG11

path = "./checkpoints/.."
weights = torch.load(path)

torch_model = VGG11.load_state_dict(weights)

torch_input = torch.randn(1, 3, 224, 224)
onnx_model = torch.onnx.dynamo_export(torch_model, torch_input)
onnx_model.save("./vgg11_student_model.onnx")