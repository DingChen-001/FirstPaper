import torch
import torchvision.transforms as T

class FeaturePreprocessor:
    """
    统一不同特征的空间和通道维度
    关键功能：
    - 空间重采样（对齐DIRE高分辨率残差与SSP块）
    - 通道压缩（统一DNF/LGrad维度）
    """
    def __init__(self, target_size=256, feature_dims=256):
        # 空间对齐变换
        self.resize = T.Resize(target_size, antialias=True)
        
        # 通道压缩层（1x1卷积）
        self.dnf_adaptor = nn.Conv2d(1, feature_dims, 1)
        self.dire_adaptor = nn.Conv2d(3, feature_dims, 1)
        self.lgrad_adaptor = nn.Conv2d(64, feature_dims, 1)
        self.ssp_adaptor = nn.Conv2d(128, feature_dims, 1)

    def __call__(self, features: dict) -> dict:
        """
        输入: 各特征的原始输出字典
            {
                "dnf": [B, 1, 128, 128],
                "dire": [B, 3, 512, 512],
                "lgrad": [B, 64, 256, 256],
                "ssp": [B, 128, 64, 64]
            }
        输出: 对齐后的特征字典（空间和通道统一）
        """
        processed = {}
        # 处理DNF特征
        processed['dnf'] = self.dnf_adaptor(features['dnf'])
        
        # 处理DIRE特征：降采样到256x256
        dire_resized = self.resize(features['dire'])
        processed['dire'] = self.dire_adaptor(dire_resized)
        
        # 处理LGrad特征：无需调整大小，只压缩通道
        processed['lgrad'] = self.lgrad_adaptor(features['lgrad'])
        
        # 处理SSP特征：上采样到256x256
        ssp_up = F.interpolate(features['ssp'], size=256, mode='bilinear')
        processed['ssp'] = self.ssp_adaptor(ssp_up)
        
        return processed  # 所有特征变为[B, 256, 256, 256]
        def fgsm_attack(image, epsilon, data_grad):
            """
            FGSM对抗样本生成（用于增强训练鲁棒性）
            参考论文：Adversarial Examples for Generative Models
            """
            sign_data_grad = data_grad.sign()
            perturbed_image = image + epsilon * sign_data_grad
            return torch.clamp(perturbed_image, 0, 1)

class AdversarialAugment:
    def __init__(self, model, epsilon=0.03):
        self.model = model
        self.epsilon = epsilon
    
    def __call__(self, x):
        x.requires_grad = True
        outputs = self.model(x)
        loss = F.cross_entropy(outputs, torch.zeros_like(outputs))  # 欺骗模型
        loss.backward()
        perturbed = fgsm_attack(x, self.epsilon, x.grad.data)
        return perturbed.detach()