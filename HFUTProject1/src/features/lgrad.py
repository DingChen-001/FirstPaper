# src/features/lgrad.py
class GradientOperator(nn.Module):
    """自定义梯度特征计算层"""
    def __init__(self):
        super().__init__()
        # Sobel算子参数
        self.sobel_x = nn.Parameter(
            torch.tensor([[[1, 0, -1], [2, 0, -2], [1, 0, -1]]], dtype=torch.float32),
            requires_grad=False
        )
        self.sobel_y = nn.Parameter(
            torch.tensor([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]], dtype=torch.float32),
            requires_grad=False
        )

    def forward(self, x):
        # 计算梯度幅值和方向
        g_x = F.conv2d(x, self.sobel_x.repeat(x.shape[1],1,1,1))
        g_y = F.conv2d(x, self.sobel_y.repeat(x.shape[1],1,1,1))
        magnitude = torch.sqrt(g_x**2 + g_y**2)
        orientation = torch.atan2(g_y, g_x)
        return torch.cat([magnitude, orientation], dim=1)

class LGradFeature(nn.Module):
    def __init__(self, pretrained_path="models/pre-trained/resnet50.pth"):
        super().__init__()
        self.grad_op = GradientOperator()
        self.backbone = ResNet50(pretrained=True)
        
        # 注册梯度钩子
        self.gradients = {}
        def save_grad(name):
            def hook(grad):
                self.gradients[name] = grad
            return hook
        
        self.backbone.layer3.register_forward_hook(save_grad('layer3'))
        self.backbone.layer4.register_forward_hook(save_grad('layer4'))

    def forward(self, x):
        # 原始图像梯度特征
        raw_grad = self.grad_op(x)
        
        # 深度特征梯度
        _ = self.backbone(x)
        layer3_grad = self.gradients['layer3']
        layer4_grad = self.gradients['layer4']
        
        # 特征融合
        return torch.cat([
            raw_grad,
            F.interpolate(layer3_grad, scale_factor=2),
            F.interpolate(layer4_grad, scale_factor=4)
        ], dim=1)