import torch
import torch.nn as nn

class LearnableGradient(nn.Module):
    """
    LGrad特征提取器：基于深度梯度学习的伪影检测
    论文核心：生成图像的梯度模式与真实图像存在差异
    """
    def __init__(self, pretrained_path='models/pre-trained/resnet50.pth'):
        super().__init__()
        # 加载预训练ResNet并提取中间梯度
        self.backbone = ResNet50(pretrained_path)
        self.grad_layers = ['layer3', 'layer4']
        
        # 梯度特征处理（参考lgrad.txt的GradientStream模块）
        self.grad_processor = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1)
        )
        
        # 注册梯度钩子
        self.gradients = {}
        for name, module in self.backbone.named_modules():
            if name in self.grad_layers:
                module.register_forward_hook(self.save_gradient)
    
    def save_gradient(self, module, input, output):
        """保存中间层的梯度"""
        def grad_hook(grad):
            self.gradients[module] = grad
        output.register_hook(grad_hook)
    
    def forward(self, x):
        # 前向传播获取梯度
        _ = self.backbone(x)
        
        # 提取并处理梯度特征
        grad_feats = []
        for layer in self.grad_layers:
            grad = self.gradients[layer]
            processed = self.grad_processor(grad)
            grad_feats.append(F.adaptive_avg_pool2d(processed, (256, 256)))
        
        return torch.cat(grad_feats, dim=1)  # [B, 512, 256, 256]