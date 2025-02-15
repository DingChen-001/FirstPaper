from timm.models.vision_transformer import Block

class CrossViTClassifier(nn.Module):
    """
    完整CrossViT实现,支持：
    - 多尺度patch嵌入
    - 交叉注意力机制
    - 分类头微调
    """
    def __init__(self, config):
        super().__init__()
        # 多尺度嵌入
        self.patch_embeds = nn.ModuleList([
            PatchEmbed(img_size=config.img_size, 
                      patch_size=ps, 
                      in_chans=config.in_chans,
                      embed_dim=config.embed_dim)
            for ps in config.patch_sizes
        ])
        
        # 交叉注意力块
        self.blocks = nn.ModuleList([
            Block(dim=config.embed_dim, 
                  num_heads=config.num_heads,
                  mlp_ratio=config.mlp_ratio)
            for _ in range(config.depth)
        ])
        
        # 分类头
        self.head = nn.Linear(config.embed_dim * len(config.patch_sizes), config.num_classes)

    def forward(self, x):
        multi_scale = []
        for embed in self.patch_embeds:
            # 多尺度特征提取
            feat = embed(x)
            cls_token = self.cls_token.expand(feat.shape[0], -1, -1)
            feat = torch.cat((cls_token, feat), dim=1)
            
            # 交叉注意力处理
            for blk in self.blocks:
                feat = blk(feat)
            multi_scale.append(feat[:, 0])  # 取CLS token
        
        # 多尺度特征融合
        fused = torch.cat(multi_scale, dim=1)
        return self.head(fused)
    
    @classmethod
    def from_pretrained(cls, config_path, ckpt_path):
        """加载预训练模型"""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        model = cls(config)
        state_dict = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict)
        return model.eval()

    def predict(self, features):
        with torch.no_grad():
            logits = self(features)
            return torch.softmax(logits, dim=-1)