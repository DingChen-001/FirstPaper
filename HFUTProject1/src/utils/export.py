import torch
import torch.onnx

def export_to_onnx(model, sample_input, output_path="model.onnx"):
    """
    将融合模型导出为ONNX格式（支持动态批量维度）
    输入示例：sample_input = {'dnf': torch.randn(1,256,256,256), ...}
    """
    # 将输入字典转换为元组（ONNX限制）
    input_names = list(sample_input.keys())
    input_tensors = tuple(sample_input.values())
    
    # 动态轴配置（批量维度可变化）
    dynamic_axes = {name: {0: 'batch_size'} for name in input_names}
    
    torch.onnx.export(
        model,
        input_tensors,
        output_path,
        input_names=input_names,
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=13
    )