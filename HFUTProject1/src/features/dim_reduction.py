import torch.nn as nn

class FeatureReducer(nn.Module):
    """
    统一特征降维与归一化处理（适用于所有输入特征）
    支持动态维度适配与多种归一化策略选择
    """
    def __init__(self, in_dim, out_dim=256, norm_type='layer'):
        super().__init__()
        # 降维层
        self.reducer = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1),
            nn.GELU()
        )
        
        # 归一化选择
        self.norm = {
            'layer': nn.LayerNorm(out_dim),
            'batch': nn.BatchNorm2d(out_dim),
            'instance': nn.InstanceNorm2d(out_dim)
        }[norm_type]

    def forward(self, x):
        reduced = self.reducer(x)
        # 归一化（保持维度兼容性）
        if isinstance(self.norm, nn.LayerNorm):
            B, C, H, W = reduced.shape
            return self.norm(reduced.view(B, C, -1).transpose(1,2)).transpose(1,2).view(B, C, H, W)
        else:
            return self.norm(reduced)