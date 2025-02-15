import yaml
from pathlib import Path

def load_config(config_path):
    """加载YAML配置文件并转换为字典"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 路径自动补全处理
    base_dir = Path(__file__).parent.parent
    for key in ['data', 'models']:
        if key in config:
            config[key] = {k: str(base_dir / v) for k, v in config[key].items()}
    
    return config