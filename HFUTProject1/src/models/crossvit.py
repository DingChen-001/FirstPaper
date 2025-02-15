# src/models/crossvit.py
class CrossScaleAttention(nn.Module):
    def __init__(self, dim, num_heads=8, scales=[16, 8]):
        super().__init__()
        self.scale_embeddings = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=s, stride=s) for s in scales
        ])
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.norm = nn.LayerNorm(dim)

    def forward(self, local_feat, global_feat):
        # 多尺度嵌入
        scale_feats = []
        for emb in self.scale_embeddings:
            scale_feats.append(emb(global_feat).flatten(2).transpose(1,2))
        
        # 注意力计算
        query = local_feat.flatten(2).transpose(1,2)
        key = torch.cat(scale_feats, dim=1)
        value = key
        
        attn_out, _ = self.attention(query, key, value)
        attn_out = self.norm(attn_out + query)
        
        # 恢复空间维度
        return attn_out.transpose(1,2).view_as(local_feat)

class CrossViT(nn.Module):
    def __init__(self, in_dims={'dnf':256, 'dire':256, 'lgrad':512, 'ssp':128}):
        super().__init__()
        # 输入适配层
        self.adaptors = nn.ModuleDict({
            name: nn.Conv2d(dim, 256, 1) for name, dim in in_dims.items()
        })
        
        # 交叉注意力模块
        self.cross_attn1 = CrossScaleAttention(256)
        self.cross_attn2 = CrossScaleAttention(256)
        
        # 分类头
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 2)
        )

    def forward(self, features):
        # 统一特征维度
        aligned = {}
        for name, feat in features.items():
            aligned[name] = self.adaptors[name](feat)
        
        # 层次化融合
        local_fused = self.cross_attn1(aligned['lgrad'], aligned['ssp'])
        global_fused = self.cross_attn2(aligned['dnf'], aligned['dire'])
        
        # 最终分类
        return self.head(local_fused + global_fused)