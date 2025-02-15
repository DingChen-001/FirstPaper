from torch.optim import Adam, AdamW, SGD
from .scheduler import WarmupCosineDecay

def build_optimizer(model, config):
    """
    根据配置返回优化器与学习率调度器
    支持不同模块设置差异化的学习率
    """
    # 分离特征提取参数与融合参数
    feat_params = []
    fusion_params = []
    for name, param in model.named_parameters():
        if 'backbone' in name:
            feat_params.append(param)
        else:
            fusion_params.append(param)
    
    # 多参数组优化
    optimizer = AdamW([
        {'params': feat_params, 'lr': config.feat_lr},
        {'params': fusion_params, 'lr': config.fusion_lr}
    ], weight_decay=config.weight_decay)
    
    # 学习率调度
    scheduler = WarmupCosineDecay(
        optimizer, 
        warmup_epochs=config.warmup_epochs,
        total_epochs=config.total_epochs
    )
    return optimizer, scheduler