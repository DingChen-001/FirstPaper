# 指令

## 运行流程

- 数据准备：将原始数据按目录结构放入data/raw/

- 单特征预训练：运行 python main.py --stage pretrain --config config/dnf_config.yaml

- 融合模型训练：运行 python main.py --stage fusion

- 性能评估：运行 python main.py --stage test

- 模型导出：执行 python deploy/convert_onnx.py

---

## 运行指令

### 分阶段训练

python main.py --stage pretrain --config config/dnf.yaml
python main.py --stage fusion --config config/crossvit.yaml

### 启用多GPU训练

torchrun --nproc_per_node=4 main.py --stage fusion --config config/crossvit.yaml
