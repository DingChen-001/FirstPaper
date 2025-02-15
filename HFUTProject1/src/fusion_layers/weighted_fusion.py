class SmartFusion(nn.Module):
    def __init__(self, feat_dims, cfg):
        super().__init__()
        # 动态配置降维器
        self.reducers = nn.ModuleDict({
            name: HybridReducer(
                dim, 
                out_dim=cfg.fusion.reduced_dim,
                norm_type=cfg.norm_type,
                use_pca=cfg.use_pca
            ) for name, dim in feat_dims.items()
        })
        
        # 自适应权重融合
        self.weights = nn.ParameterDict({
            name: nn.Parameter(torch.ones(1)) for name in feat_dims.keys()
        })

    def forward(self, features):
        # 降维与归一化
        reduced = {name: reducer(feat) for name, (feat, reducer) in zip(features, self.reducers)}
        
        # 加权融合
        total_weight = sum([self.weights[name] for name in features.keys()])
        fused = sum([reduced[name] * (self.weights[name]/total_weight) for name in features.keys()])
        return fused