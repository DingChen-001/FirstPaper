import torch
import torch.nn as nn
from einops import rearrange

class CrossScaleAttention(nn.Module):
    """
    改进的跨尺度注意力机制（参考CrossViT论文第3.2节）
    核心改进：引入位置编码与通道注意力
    """
    def __init__(self, dim, num_heads=8, scale_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.scale_ratio = scale_ratio
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, dim, 1, 1))
        
        # 线性变换
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 16, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 16, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, local_feat, global_feat):
        """
        输入:
            local_feat: 局部特征 [B, C, H, W]
            global_feat: 全局特征 [B, C, H*s, W*s]
        输出: 融合特征 [B, C, H, W]
        """
        B, C, H, W = local_feat.shape
        
        # 缩放全局特征到局部尺寸
        global_down = F.interpolate(global_feat, scale_factor=self.scale_ratio, mode='bilinear')
        
        # 合并特征并添加位置编码
        fused = local_feat + global_down + self.pos_embed
        
        # 通道注意力加权
        channel_weights = self.channel_attn(fused)
        fused = fused * channel_weights
        
        # 生成QKV
        qkv = self.to_qkv(fused).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=self.num_heads), qkv)
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 特征聚合
        out = (attn @ v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=H, y=W)
        return out