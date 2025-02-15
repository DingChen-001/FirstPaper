from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineDecay(_LRScheduler):
    """
    自定义学习率调度：线性热身+余弦衰减
    论文验证对GAN检测任务有效
    """
    def __init__(self, optimizer, warmup_epochs=5, total_epochs=100):
        self.warmup = warmup_epochs
        self.total = total_epochs
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup:
            # 线性热身
            return [base_lr * (self.last_epoch+1)/self.warmup 
                    for base_lr in self.base_lrs]
        else:
            # 余弦衰减
            progress = (self.last_epoch - self.warmup) / (self.total - self.warmup)
            return [base_lr * 0.5 * (1 + math.cos(math.pi * progress)) 
                    for base_lr in self.base_lrs]