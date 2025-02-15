import torch
from models.crossvit import CrossViTFusion
from features import DNF, DIRE, LGrad, SSP
from datasets import FeatureDataset
from torch.utils.data import DataLoader

def train_stage1(config):
    """阶段1：单特征预训练"""
    # 初始化特征提取器
    dnf_extractor = DNF(pretrained_path=config.dnf.model_path)
    dire_extractor = DIRE(config.dire.vae_path)
    
    # 数据集与加载器
    dataset = FeatureDataset(root=config.data.path, 
                            preprocessors={'dnf': dnf_extractor, 'dire': dire_extractor})
    loader = DataLoader(dataset, batch_size=config.train.batch_size)
    
    # 训练循环（示例：DNF训练）
    optimizer = torch.optim.Adam(dnf_extractor.parameters(), lr=config.dnf.lr)
    for epoch in range(config.train.epochs):
        for batch in loader:
            images, labels = batch
            # 提取特征并计算损失
            features = dnf_extractor(images)
            loss = F.cross_entropy(features, labels)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def train_stage2(config):
    """阶段2:融合模型训练"""
    # 加载预训练特征提取器（冻结参数）
    extractors = {
        'dnf': DNF(pretrained_path=config.dnf.model_path).eval(),
        'dire': DIRE(config.dire.vae_path).eval(),
        'lgrad': LGrad(config.lgrad.weights_path).eval(),
        'ssp': SSP(config.ssp.patch_size).eval()
    }
    
    # 初始化融合模型
    fusion_model = CrossViTFusion(num_classes=2)
    
    # 数据加载（使用预处理对齐）
    dataset = FeatureDataset(root=config.data.path, 
                            preprocessors=extractors,
                            transform=AdversarialAugment(fusion_model))  # 对抗增强
    loader = DataLoader(dataset, batch_size=config.train.batch_size)
    
    # 优化器仅训练融合部分
    optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=config.fusion.lr)
    
    # 训练循环
    for epoch in range(config.fusion.epochs):
        for batch in loader:
            features, labels = batch
            outputs = fusion_model(features)
            loss = F.cross_entropy(outputs, labels)
            # 梯度回传
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()