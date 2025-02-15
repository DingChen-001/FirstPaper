import torch
import torch.nn as nn
from sklearn.decomposition import PCA

class HybridReducer(nn.Module):
    """
    混合降维策略：
    - 训练阶段：使用可学习的1x1卷积降维
    - 推理阶段：可选PCA降维（需提前拟合）
    - 支持多种归一化方法
    """
    def __init__(self, in_dim, out_dim=256, norm_type='bn', use_pca=False):
        super().__init__()
        # 可学习降维
        self.conv_reducer = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1),
            nn.GELU()
        )
        
        # PCA降维器
        self.pca = PCA(n_components=out_dim) if use_pca else None
        self.use_pca = use_pca
        
        # 归一化层
        self.norm = {
            'bn': nn.BatchNorm2d(out_dim),
            'ln': nn.LayerNorm([out_dim, 1, 1]),
            'in': nn.InstanceNorm2d(out_dim)
        }[norm_type]

    def fit_pca(self, features):
        """离线拟合PCA"""
        flattened = features.view(-1, features.size(1)).cpu().numpy()
        self.pca.fit(flattened)
        
    def forward(self, x):
        if self.training or not self.use_pca:
            # 训练模式使用卷积降维
            reduced = self.conv_reducer(x)
        else:
            # 推理模式可选PCA
            x_flat = x.permute(0,2,3,1).view(-1, x.size(1))
            reduced = torch.tensor(self.pca.transform(x_flat.cpu())).to(x.device)
            reduced = reduced.view(x.size(0), x.size(2), x.size(3), -1).permute(0,3,1,2)
            
        return self.norm(reduced)