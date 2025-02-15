import argparse
from train import train_stage1, train_stage2
from test import evaluate_model
from config import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, required=True, 
                       choices=['pretrain', 'fusion', 'test'])
    parser.add_argument('--config', type=str, default="config/fusion_config.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.stage == "pretrain":
        train_stage1(config)
    elif args.stage == "fusion":
        train_stage2(config)
    elif args.stage == "test":
        evaluate_model(config)