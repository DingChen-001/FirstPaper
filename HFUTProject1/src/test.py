import torch
from models import CrossViTFusion
from utils.metrics import DetectionEvaluator
from datasets import MultiFeatureDataset
from torch.utils.data import DataLoader

def evaluate_model(config):
    # 初始化模型
    model = CrossViTFusion(num_classes=2)
    model.load_state_dict(torch.load(config.model.checkpoint_path))
    model.eval()
    
    # 加载测试集
    test_dataset = MultiFeatureDataset(
        root_dir=config.data.test_path,
        preprocessors=config.preprocessors,
        is_train=False
    )
    test_loader = DataLoader(test_dataset, batch_size=config.test.batch_size)
    
    # 评估器
    evaluator = DetectionEvaluator()
    
    with torch.no_grad():
        for batch in test_loader:
            features, labels = batch
            start_time = time.time()
            outputs = model(features)
            infer_time = time.time() - start_time
            
            evaluator.update(outputs, labels, infer_time)
    
    metrics = evaluator.compute()
    print(f"测试结果：{metrics}")
    return metrics

if __name__ == "__main__":
    from config import load_config  # 假设有配置文件加载函数
    config = load_config("config/fusion_config.yaml")
    evaluate_model(config)