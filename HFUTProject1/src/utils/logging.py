import time
from torch.utils.tensorboard import SummaryWriter

class TrainingLogger:
    """训练过程记录与可视化"""
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.start_time = time.time()
        self.step = 0
    
    def log_metrics(self, metrics_dict, phase='train'):
        """记录指标到TensorBoard"""
        for name, value in metrics_dict.items():
            self.writer.add_scalar(f'{phase}/{name}', value, self.step)
        
        # 计算累计时间
        elapsed = time.time() - self.start_time
        self.writer.add_scalar(f'{phase}/elapsed_time', elapsed, self.step)
        self.step += 1
    
    def log_model_graph(self, model, input_sample):
        """记录模型计算图"""
        self.writer.add_graph(model, input_sample)
    
    def close(self):
        self.writer.close()