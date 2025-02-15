import torch
from models import CrossViTFusion
from utils.export import export_to_onnx

# 加载训练好的模型
model = CrossViTFusion(num_classes=2)
model.load_state_dict(torch.load("models/trained/fusion_model.pth"))

# 创建示例输入（符合各特征的尺寸要求）
sample_input = {
    'dnf': torch.randn(1, 256, 256, 256),
    'dire': torch.randn(1, 256, 256, 256),
    'lgrad': torch.randn(1, 256, 256, 256),
    'ssp': torch.randn(1, 256, 256, 256)
}

# 导出为ONNX
export_to_onnx(model, sample_input, "deploy/fusion_model.onnx")