def validate_model(model, val_loader, criterion):
    """
    在验证集上评估模型性能，返回关键指标
    包含早停(early stopping)逻辑判断
    """
    model.eval()
    evaluator = DetectionEvaluator()
    
    with torch.no_grad():
        for batch in val_loader:
            features, labels = batch
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            evaluator.update(outputs, labels, 0)  # 时间统计关闭
    
    metrics = evaluator.compute()
    metrics['val_loss'] = loss.item()
    return metrics

class EarlyStopper:
    """早停机制控制器"""
    def __init__(self, patience=5, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def should_stop(self, current_loss):
        if current_loss < self.best_loss - self.delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience