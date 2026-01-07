# Model - 模型设计与训练模块

本文件夹包含FCN模型定义和训练脚本。

## 文件说明

| 文件名 | 功能描述 |
|--------|----------|
| `FCN.py` | FCN模型定义，包含单任务和双任务架构 |
| `train_all.py` | 训练脚本，支持单任务和双任务训练 |
| `train_baselines_comparison.py` | 传统机器学习模型（KNN、SVM、Random Forest）对比实验 |
| `MODEL_PARAMS.md` | 模型参数详细说明文档 |

## 分类任务

| 任务名称 | 类型 | 数据集 | 输出 |
|----------|------|--------|------|
| PP+Starch | 单任务 | `dataset/PP+淀粉/` | PP浓度(3类) |
| PE+Starch | 单任务 | `dataset/PE+淀粉/` | PE浓度(3类) |
| PP+PE+Starch | 双任务 | `dataset/PP+PE+淀粉/` | PP浓度 + PE浓度 |

## 模型架构

### FaultClassificationNetwork (单任务)
```
输入 -> FeatureEncoder -> Classifier -> 输出(3类)
```

### FaultMultiHeadNetwork (双任务)
```
输入 -> FeatureEncoder -> Classifier_PP -> PP输出(3类)
                       -> Classifier_PE -> PE输出(3类)
```

## 评估指标

| 指标 | 说明 |
|------|------|
| Accuracy | 准确率 |
| F1-score (macro) | 宏平均F1分数 |
| Recall (macro) | 宏平均召回率 |

## 模型输出位置

训练好的模型保存在 `output/model_data/` 目录下：

| 任务 | 路径 | 权重文件 |
|------|------|----------|
| PP+Starch | `PP_Starch/checkpoints/best/` | encoder.pth, classifier.pth |
| PE+Starch | `PE_Starch/checkpoints/best/` | encoder.pth, classifier.pth |
| PP+PE+Starch | `PP_PE_Starch/checkpoints/best/` | encoder.pth, classifier_pp.pth, classifier_pe.pth |

## 使用方法

```bash
# 训练所有模型
python model/train_all.py --task all --root dataset --output-root output/model_data

# 训练单个任务
python model/train_all.py --task pp --root dataset --output-root output/model_data

# 训练双任务模型
python model/train_all.py --task pp_pe --root dataset --output-root output/model_data

# 运行传统模型对比实验
python model/train_baselines_comparison.py
```

## 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 100 | 最大训练轮数 |
| `--batch-size` | 32 | 批次大小 |
| `--lr` | 3e-4 | 学习率 |
| `--weight-decay` | 1e-4 | 权重衰减 |
| `--patience` | 20 | 早停耐心值 |
| `--seed` | 42 | 随机种子 |
| `--task` | all | 训练任务（all/pp/pe/pp_pe） |

## 性能对比

| 模型 | PP+Starch | PE+Starch | PP+PE+Starch |
|------|-----------|-----------|--------------|
| **FCN (Ours)** | **~97%** | **~97%** | **~98.68%** |
| KNN | 73.33% | 73.33% | 86.67% |
| SVM (RBF) | 63.33% | 76.67% | 83.33% |
| Random Forest | 66.67% | 76.67% | 86.67% |

详细模型参数请参考 [MODEL_PARAMS.md](MODEL_PARAMS.md)
