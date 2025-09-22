# 🚀 RPT - 大语言模型训练与评测项目

基于 Qwen2.5-7B 的大语言模型微调训练与评测框架，支持 LoRA 微调和多种评测指标。

## 📋 目录

- [项目简介](#-项目简介)
- [环境要求](#-环境要求)
- [安装配置](#-安装配置)
- [使用方法](#-使用方法)
- [项目结构](#-项目结构)
- [评测结果](#-评测结果)

## 🎯 项目简介

本项目提供了一个完整的大语言模型训练与评测流水线，包括：
- 基于 Qwen2.5-7B 的模型微调
- LoRA 参数高效微调
- 多维度模型评测
- 可视化结果展示

## 💻 环境要求

- Python 3.8+
- CUDA 支持的 GPU
- LLaMA Factory 框架

## 🔧 安装配置

### 1. 下载模型文件

📁 **Qwen2.5-7B 基础模型**
```bash
# 从 Hugging Face 下载 Qwen2.5-7B instruction 模型
# 放置到: models/ 目录下
```

📁 **LoRA 模型文件**
```bash
# 从 Google Drive 下载 LoRA 模型文件
# 放置到: outputs/train/qwen25-7b/lora/ 目录下
```

📁 **数据文件**
```bash
# 下载训练和评测数据文件
# 放置到: data/ 目录下
```

### 2. 安装依赖框架

```bash
# 下载并安装 LLaMA Factory 框架
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```

### 3. 配置文件设置

编辑 `configs/global.yaml` 文件，配置上述文件夹的存放地址：

```yaml
# 参考现有的 configs/global.yaml 文件进行配置
model_path: "path/to/models"
lora_path: "path/to/outputs/train/qwen25-7b/lora"
data_path: "path/to/data"
```

## 🚀 使用方法

### 训练模型

```bash
# 运行训练脚本
bash scripts/train.sh
```

### 模型推理

```bash
# 运行推理脚本
bash scripts/infer.sh
```

### 模型评测

```bash
# 运行评测脚本
bash scripts/eval.sh
```

> 💡 **提示**: 运行前请修改脚本中的相关路径配置

## 📁 项目结构

```
RPT/
├── 📂 configs/          # 配置文件
│   └── global.yaml      # 全局配置
├── 📂 data/             # 数据文件
├── 📂 models/           # 模型文件
├── 📂 outputs/          # 输出结果
│   ├── evaluation/      # 评测结果
│   └── train/          # 训练输出
├── 📂 scripts/          # 执行脚本
│   ├── train.sh        # 训练脚本
│   ├── infer.sh        # 推理脚本
│   └── eval.sh         # 评测脚本
├── 📂 src/              # 源代码
└── README.md           # 项目说明
```

## 📊 评测结果

评测结果保存在 `outputs/evaluation/` 目录下，包含两种评测方式：

### 📈 PA 评测
- 提供柱状图可视化


### 📈 K_W 评测
- 提供柱状图可视化
- 提供雷达图多维度分析

---

## 📝 注意事项

1. 确保所有文件路径配置正确
2. 检查 GPU 内存是否足够进行模型训练
3. 根据实际情况调整训练参数
