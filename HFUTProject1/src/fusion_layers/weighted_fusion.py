# src/fusion_layers/weighted_fusion.py
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes//ratio, 1),
            nn.ReLU(),
            nn.Conv2d(in_planes//ratio, in_planes, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class FeatureFusion(nn.Module):
    def __init__(self, feature_dims):
        super().__init__()
        self.attentions = nn.ModuleDict({
            name: ChannelAttention(dim) for name, dim in feature_dims.items()
        })
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(sum(feature_dims.values()), 512, 1),
            nn.BatchNorm2d(512),
            nn.GELU()
        )

    def forward(self, features):
        # 通道注意力加权
        weighted = {}
        for name, feat in features.items():
            attn = self.attentions[name](feat)
            weighted[name] = feat * attn
        
        # 拼接并降维
        fused = torch.cat(list(weighted.values()), dim=1)
        return self.fusion_conv(fused)
    
class EnhancedFusion(nn.Module):
    def __init__(self, feat_dims):
        super().__init__()
        # 初始化各特征的降维器
        self.reducers = nn.ModuleDict({
            name: FeatureReducer(dim, norm_type='layer') 
            for name, dim in feat_dims.items()
        })
        
        # 注意力融合层
        self.fusion = FeatureFusion(feat_dims={k:256 for k in feat_dims})

    def forward(self, features):
        # 降维+归一化
        reduced_feats = {name: self.reducers[name](feat) for name, feat in features.items()}
        return self.fusion(reduced_feats)