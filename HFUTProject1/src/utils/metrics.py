import torch
from sklearn.metrics import accuracy_score, f1_score
import time

class DetectionEvaluator:
    """综合评估指标计算器"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.preds = []
        self.labels = []
        self.inference_times = []
    
    def update(self, logits: torch.Tensor, labels: torch.Tensor, infer_time: float):
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        self.preds.extend(preds)
        self.labels.extend(labels.cpu().numpy())
        self.inference_times.append(infer_time)
    
    def compute(self):
        """返回关键指标字典"""
        acc = accuracy_score(self.labels, self.preds)
        f1 = f1_score(self.labels, self.preds)
        avg_time = sum(self.inference_times) / len(self.inference_times)
        return {
            'Accuracy': round(acc, 4),
            'F1-Score': round(f1, 4),
            'Inference Time (ms)': round(avg_time * 1000, 2),
            'FPS': round(1 / avg_time, 1) if avg_time > 0 else 0
        }