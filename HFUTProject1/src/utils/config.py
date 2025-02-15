import yaml
from addict import Dict

def load_config(path):
    with open(path, 'r') as f:
        return Dict(yaml.safe_load(f))

def merge_config(base, override):
    """合并多层配置"""
    for k, v in override.items():
        if isinstance(v, dict) and k in base:
            merge_config(base[k], v)
        else:
            base[k] = v