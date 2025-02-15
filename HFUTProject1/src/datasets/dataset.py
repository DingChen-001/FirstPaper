import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from .preprocess import FeaturePreprocessor

class MultiFeatureDataset(Dataset):
    """
    支持多特征提取的数据集类
    功能：
    - 自动加载图像并应用不同特征提取器
    - 支持对抗样本动态生成
    """
    def __init__(self, root_dir, preprocessors, transform=None, is_train=True):
        """
        Args:
            root_dir: 数据集根目录（需包含real_images/和generated_images/）
            preprocessors: 字典，包含各特征提取器实例
            transform: 数据增强函数
        """
        self.real_dir = os.path.join(root_dir, 'real_images')
        self.fake_dir = os.path.join(root_dir, 'generated_images')
        self.real_images = [os.path.join(self.real_dir, f) for f in os.listdir(self.real_dir)]
        self.fake_images = [os.path.join(self.fake_dir, f) for f in os.listdir(self.fake_dir)]
        self.all_images = self.real_images + self.fake_images
        self.labels = [0]*len(self.real_images) + [1]*len(self.fake_images)
        
        self.preprocessors = preprocessors  # {'dnf': DNF实例, 'dire': DIRE实例...}
        self.transform = transform
        self.feature_align = FeaturePreprocessor()
        self.is_train = is_train

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        label = self.labels[idx]
        
        # 加载图像并转换为Tensor
        img = Image.open(img_path).convert('RGB')
        img_tensor = torchvision.transforms.ToTensor()(img)
        
        # 数据增强（训练时启用）
        if self.transform and self.is_train:
            img_tensor = self.transform(img_tensor)
        
        # 并行提取各特征
        features = {}
        for name, processor in self.preprocessors.items():
            with torch.set_grad_enabled(name == 'dnf'):  # 仅DNF需要梯度
                features[name] = processor(img_tensor.unsqueeze(0)).squeeze(0)
        
        # 特征对齐
        aligned_features = self.feature_align(features)
        return aligned_features, torch.tensor(label, dtype=torch.long)