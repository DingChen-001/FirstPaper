import random
import torchvision.transforms as T

class GenerativeAwareAugment:
    """
    面向生成图像检测的特化数据增强组合
    包含：
    - 频域滤波（模拟生成伪影）
    - 局部噪声注入
    - 色彩偏移
    """
    def __init__(self):
        # 空间增强
        self.spatial_aug = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.RandomResizedCrop(256, scale=(0.8, 1.0))
        ])
        
        # 频域增强
        self.freq_aug = T.RandomApply([
            T.GaussianBlur(kernel_size=5),
            T.GaussianBlur(kernel_size=3)
        ], p=0.5)
        
        # 噪声增强
        self.noise_levels = [0.01, 0.03, 0.05]

    def __call__(self, img_tensor):
        # 空间变换
        img = self.spatial_aug(img_tensor)
        
        # 频域滤波
        img = self.freq_aug(img)
        
        # 注入脉冲噪声
        if random.random() > 0.7:
            noise_mask = torch.rand_like(img) < random.choice(self.noise_levels)
            img = torch.where(noise_mask, torch.rand_like(img), img)
        
        return img