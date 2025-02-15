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