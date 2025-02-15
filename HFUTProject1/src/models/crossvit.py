class CrossViT(nn.Module):
    """
    完整CrossViT架构，包含：
    - 双分支多尺度处理
    - 交叉注意力融合
    - 分类头适配
    """
    def __init__(self, cfg):
        super().__init__()
        # 分支1：小尺度处理
        self.branch_small = self._build_branch(
            img_size=cfg.img_size,
            patch_size=cfg.patch_sizes[0],
            embed_dim=cfg.embed_dim,
            depth=cfg.depth
        )
        
        # 分支2：大尺度处理
        self.branch_large = self._build_branch(
            img_size=cfg.img_size,
            patch_size=cfg.patch_sizes[1],
            embed_dim=cfg.embed_dim,
            depth=cfg.depth
        )
        
        # 交叉注意力模块
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads
        )
        
        # 分类头
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim * 2),
            nn.Linear(cfg.embed_dim * 2, cfg.num_classes)
        )

    def _build_branch(self, img_size, patch_size, embed_dim, depth):
        return nn.Sequential(
            PatchEmbed(img_size, patch_size, 3, embed_dim),
            *[TransformerBlock(embed_dim, num_heads=8) for _ in range(depth)]
        )

    def forward(self, x):
        # 双分支处理
        feat_s = self.branch_small(x)  # [B, N, D]
        feat_l = self.branch_large(x)  # [B, M, D]
        
        # 交叉注意力
        feat_s, _ = self.cross_attn(feat_s, feat_l, feat_l)
        feat_l, _ = self.cross_attn(feat_l, feat_s, feat_s)
        
        # 分类特征聚合
        cls_feat = torch.cat([feat_s[:, 0], feat_l[:, 0]], dim=1)
        return self.head(cls_feat)
    @classmethod
    def from_pretrained(cls, config_path, model_path=None):
        """智能加载模型"""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        
        model = cls(cfg['model'])
        if model_path:
            state_dict = torch.load(model_path, map_location='cpu')
            
            # 处理可能的键不匹配
            new_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace("module.", "")  # 处理DP/DDP保存的权重
                new_dict[new_k] = v
            model.load_state_dict(new_dict)
        
        return model

    def serve(self, img_tensor):
        """生产环境服务接口"""
        with torch.no_grad():
            features = self.extract_features(img_tensor)
            return {
                'prob_fake': torch.softmax(self(features), dim=-1)[:, 1].cpu().numpy(),
                'features': features.cpu().numpy()
            }