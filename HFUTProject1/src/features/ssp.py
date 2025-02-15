import torch
import torch.nn as nn

class SingleSimplePatch(nn.Module):
    """
    SSP特征提取器：基于局部图像块的统计特征
    论文核心：生成图像在局部块内具有统计异常
    """
    def __init__(self, patch_size=64, num_patches=16):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        # 特征编码器（参考源码ssp.txt的LightweightStatsNet）
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1)
        )
        
    def random_patch(self, x):
        """动态选择高方差区域块（源码改进版）"""
        # 计算局部方差图
        unfolded = F.unfold(x, kernel_size=16, stride=8)
        variances = torch.var(unfolded, dim=1)  # [B, H*W]
        
        # 选择高方差区域索引
        _, top_indices = torch.topk(variances, self.num_patches, dim=1)
        return top_indices

    def forward(self, x):
        """
        输入: x [B, 3, H, W]
        输出: SSP特征 [B, 128]
        """
        # 动态选择图像块
        indices = self.random_patch(x)
        
        # 提取块特征
        batch_features = []
        for b in range(x.size(0)):
            patches = []
            for idx in indices[b]:
                # 计算块位置
                h = (idx // (x.size(3)//8)) * 8
                w = (idx % (x.size(3)//8)) * 8
                patch = x[b, :, h:h+self.patch_size, w:w+self.patch_size]
                patches.append(patch)
            
            # 编码块特征
            patches = torch.stack(patches)  # [num_patches, C, H, W]
            features = self.encoder(patches).squeeze(-1).squeeze(-1)
            batch_features.append(features.mean(dim=0))  # 平均所有块
        
        return torch.stack(batch_features)