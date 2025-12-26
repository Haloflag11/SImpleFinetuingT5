# FineTuningT5: T5模型微调项目

这是一个基于T5（Text-to-Text Transfer Transformer）模型的微调框架，专注于中文问答任务，使用LoRA（Low-Rank Adaptation）技术实现参数高效微调。

## 🌟 功能特性

- **参数高效微调**：集成LoRA技术，仅训练少量参数即可获得良好效果
- **灵活的数据集支持**：支持自定义数据集格式，易于扩展到其他任务
- **完整的训练流程**：包含数据加载、模型训练、评估和推理
- **多种优化策略**：实现了AdamW优化器、学习率调度和早停机制
- **可视化支持**：训练过程中自动生成损失曲线
- **模型导出**：支持保存最佳模型和合并LoRA权重

## 📁 项目结构

```
FineTuningT5/
├── Dataset/                    # 数据集相关代码
│   ├── __init__.py
│   └── data.py                 # 数据集加载和处理
├── Model/                      # 模型相关代码
│   ├── __init__.py
│   └── model_loader.py         # 模型加载和配置
├── Utils/                      # 工具函数
│   ├── __init__.py
│   ├── sft.py                  # 微调训练器
│   └── tools/
│       ├── __init__.py
│       ├── evaluate_bleu.py    # BLEU评估指标
│       └── leakage.py          # 数据泄露检查
├── train.py                    # 训练主脚本
├── inference.py                # 推理主脚本
├── testbleu.py                 # BLEU测试脚本
└── README.md                   # 项目说明文档
```

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 主要依赖项

- transformers
- torch
- datasets
- peft
- accelerate
- numpy
- matplotlib
- tqdm

## 📚 使用方法

### 1. 模型训练

运行训练脚本开始微调T5模型：

```bash
python train.py
```

#### 训练配置说明

在`train.py`中可以调整以下参数：

- `model_path`：预训练模型路径（默认使用mengzi-t5-base）
- `use_lora`：是否使用LoRA微调（默认启用）
- `batch_size`：训练批次大小
- `lr`：学习率
- `epochs`：训练轮数
- `patience`：早停耐心值

### 2. 模型推理

使用训练好的模型进行推理：

```bash
python inference.py
```

推理脚本会自动加载最佳模型检查点并在验证集上进行评估。

## 🎯 评估指标

项目使用BLEU（Bilingual Evaluation Understudy）指标评估生成结果的质量：

- BLEU-1：单词语义匹配度
- BLEU-2：双词语义匹配度
- BLEU-3：三词语义匹配度
- BLEU-4：四词语义匹配度

## 🔧 自定义配置

### 数据集格式

数据集采用JSON格式，每条数据包含以下字段：

```json
{
  "question": "问题文本",
  "context": "上下文文本",
  "answer": "答案文本"
}
```

### LoRA配置

在`train.py`的`load_model`函数中可以调整LoRA参数：

- `r`：LoRA的秩，控制参数更新规模
- `lora_alpha`：LoRA的缩放因子
- `target_modules`：需要应用LoRA的目标模块
- `lora_dropout`：LoRA的dropout率

## 📊 训练监控

训练过程中会自动生成损失曲线，保存在`/tmp/T5/Result/plot_loss.png`路径下。

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证，详情请查看 LICENSE 文件。

## 📞 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。

---

**FineTuningT5** - 让T5模型微调变得简单高效！