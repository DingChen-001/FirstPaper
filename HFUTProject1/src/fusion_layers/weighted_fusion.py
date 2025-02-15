import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicFeatureFusion(nn.Module):
    """
    动态加权多特征融合模块
    功能：为DNF、DIRE、LGrad、SSP特征分配可学习权重
    论文参考：DNF论文中的动态特征选择策略
    """
    def __init__(self, num_features=4, hidden_dim=128):
        super().__init__()
        # 权重生成网络（输入为各特征的均值）
        self.weight_net = nn.Sequential(
            nn.Linear(num_features * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features)
        )
        
    def forward(self, features: dict) -> torch.Tensor:
        """
        输入: features字典包含各特征张量（已对齐为相同尺寸）
        输出: 加权融合后的特征 [B, C, H, W]
        """
        # 拼接各特征的均值统计量
        stats = [torch.mean(feat, dim=[1,2,3]) for feat in features.values()]
        stats = torch.cat(stats, dim=1)  # [B, num_features*C]
        
        # 生成归一化权重
        weights = F.softmax(self.weight_net(stats), dim=1)  # [B, 4]
        
        # 加权融合
        weighted_feats = []
        for i, (name, feat) in enumerate(features.items()):
            weighted_feats.append(feat * weights[:, i].view(-1,1,1,1))
        
        return torch.sum(torch.stack(weighted_feats), dim=0)  # 求和融合