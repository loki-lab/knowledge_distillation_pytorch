import torch
from model_kit.models import VGG11

weight = torch.load("checkpoints/teacher_model/best_weight.pt")

torch_model = VGG11()
torch_model.load_state_dict(weight["model_state_dict"])

torch_input = torch.randn(1, 3, 224, 224)
onnx_model = torch.onnx.dynamo_export(torch_model, torch_input)
onnx_model.save("./vgg11_student_model.onnx")
