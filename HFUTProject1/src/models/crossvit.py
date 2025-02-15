import torch
from torch import nn
from einops import rearrange

class CrossAttentionBlock(nn.Module):
    """
    CrossViT的交叉注意力模块（参考crossvit.txt核心组件）
    输入：两个不同尺度的特征序列
    输出：交叉注意力增强后的融合特征
    """
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        
    def forward(self, x, context):
        # 交换序列维度与通道维度
        x = rearrange(x, "b c h w -> (h w) b c")
        context = rearrange(context, "b c h w -> (h w) b c")
        
        # 交叉注意力（以x为query，context为key/value）
        attn_out, _ = self.attn(
            query=self.norm1(x),
            key=self.norm1(context),
            value=self.norm1(context)
        )
        x = x + attn_out  # 残差连接
        return rearrange(x, "(h w) b c -> b c h w", h=int(x.shape[0]**0.5))

class CrossViTFusion(nn.Module):
    """
    多尺度CrossViT融合模型（支持两阶段层次化融合）
    """
    def __init__(self, in_dim=256, num_classes=2):
        super().__init__()
        # 局部分支（处理高频特征：SSP和LGrad）
        self.local_branch = CrossAttentionBlock(in_dim)
        # 全局分支（处理语义特征：DNF和DIRE）
        self.global_branch = CrossAttentionBlock(in_dim)
        
        # 分类头
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_dim, num_classes)
        
    def forward(self, features: dict) -> torch.Tensor:
        # 提取各特征
        dnf = features['dnf']
        dire = features['dire']
        lgrad = features['lgrad']
        ssp = features['ssp']
        
        # 第一阶段：局部特征融合（SSP + LGrad）
        local_fused = self.local_branch(ssp, lgrad)
        
        # 第二阶段：全局特征融合（DNF + DIRE）
        global_fused = self.global_branch(dnf, dire)
        
        # 拼接融合结果
        final_feature = torch.cat([local_fused, global_fused], dim=1)
        return self.head(final_feature)