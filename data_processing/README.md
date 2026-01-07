# Data Processing - 数据处理模块

本文件夹包含数据加载和预处理相关的代码。

## 文件说明

| 文件名 | 功能描述 |
|--------|----------|
| `data_loader.py` | 数据加载器，负责读取CSV光谱文件、数据标准化、数据集划分 |

## 主要功能

### 1. 光谱数据读取
- 支持多种编码格式（UTF-8、GBK）
- 自动解析CSV文件中的光谱数据

### 2. 数据预处理
- Z-score标准化
- 数据增强（可选）

### 3. 数据集划分
- 训练集/验证集/测试集划分
- 支持按比例或固定数量划分

## 数据集结构

```
dataset/
├── PP+淀粉/                    # PP三分类数据集
│   ├── 无污染/
│   ├── 低浓度/
│   └── 高浓度/
├── PE+淀粉/                    # PE三分类数据集
│   ├── 无污染/
│   ├── 低浓度/
│   └── 高浓度/
└── PP+PE+淀粉/                 # PP+PE混合数据集
    ├── 无污染/
    ├── 低PP+低PE+淀粉/
    ├── 高PP+低PE+淀粉/
    ├── 低PP+高PE+淀粉/
    └── 高PP+高PE+淀粉/

dataset_baseline_comparison/     # 传统模型对比实验数据集
└── ...
```

## 数据集使用说明

### FCN模型使用的数据集

| 任务 | 数据集路径 | 类别 |
|------|-----------|------|
| PP三分类 | `dataset/PP+淀粉/` | 无污染、低浓度、高浓度 |
| PE三分类 | `dataset/PE+淀粉/` | 无污染、低浓度、高浓度 |
| PP+PE五分类 | `dataset/PP+PE+淀粉/` | 无污染、低PP+低PE、高PP+低PE、低PP+高PE、高PP+高PE |
| PP二分类 | `dataset/PP+PE+淀粉/` | 低PP、高PP |
| PE二分类 | `dataset/PP+PE+淀粉/` | 低PE、高PE |

### 传统模型（对比实验）使用的数据集

| 模型 | 数据集路径 |
|------|-----------|
| KNN | `dataset_baseline_comparison/` |
| SVM (RBF) | `dataset_baseline_comparison/` |
| Random Forest | `dataset_baseline_comparison/` |

## 使用方法

```python
from data_processing.data_loader import load_dataset, standardize

# 加载数据
spectra, labels = load_dataset("dataset/PP+淀粉/")

# 标准化
spectra_normalized = standardize(spectra)
```
