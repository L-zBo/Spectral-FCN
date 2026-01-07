# FCN模型参数说明

## 模型架构

### FaultClassificationNetwork (FCN)

```
FaultClassificationNetwork
├── FeatureEncoder (特征编码器)
│   ├── ConvWide (宽卷积层)
│   │   ├── Conv1d: in=1, out=64, kernel=8, stride=4
│   │   ├── BatchNorm1d
│   │   ├── LeakyReLU
│   │   └── ChannelAttention (通道注意力)
│   ├── ConvMultiScale (多尺度卷积层) x2
│   │   ├── Conv1d: kernel=1,3,5,7 (多尺度特征提取)
│   │   ├── BatchNorm1d
│   │   ├── ReLU
│   │   └── ChannelAttention
│   └── AdaptiveAvgPool1d → 输出维度: 128
│
└── Classifier (分类器)
    ├── Linear: 128 → 128
    ├── ReLU
    └── Linear: 128 → num_classes
```

### 通道注意力机制 (ChannelAttention)

```
ChannelAttention
├── AdaptiveAvgPool1d + AdaptiveMaxPool1d
├── SE Block (Squeeze-and-Excitation)
│   ├── Conv1d: in_channels → in_channels/16
│   ├── ReLU
│   └── Conv1d: in_channels/16 → in_channels
└── Sigmoid
```

## 训练超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `epochs` | 100 | 最大训练轮数 |
| `batch_size` | 32 | 批次大小 |
| `lr` | 3e-4 | 学习率 |
| `weight_decay` | 1e-4 | 权重衰减（L2正则化） |
| `patience` | 20 | 早停耐心值（连续N轮无提升则停止） |
| `seed` | 42 | 随机种子 |
| `standardize` | zscore | 数据标准化方法 |

## 优化器与调度器

| 组件 | 类型 | 参数 |
|------|------|------|
| 优化器 | AdamW | lr=3e-4, weight_decay=1e-4 |
| 学习率调度器 | CosineAnnealingLR | T_max=epochs, eta_min=1e-6 |
| 损失函数 | CrossEntropyLoss | - |

## 数据划分

### PP+淀粉 / PE+淀粉 (三分类)

| 类别 | 总样本数 | 训练集 | 验证集 |
|------|----------|--------|--------|
| 无污染 | 30 | 22 | 8 |
| 轻微浓度 | 30 | 22 | 8 |
| 严重浓度 | 30 | 22 | 8 |
| **合计** | **90** | **66** | **24** |

### PP+PE+淀粉 (三分类)

| 类别 | 总样本数 | 训练集 | 验证集 |
|------|----------|--------|--------|
| 无污染 | 50 | 40 | 10 |
| 轻微PP+轻微PE | 50 | 40 | 10 |
| 严重PP+严重PE | 50 | 40 | 10 |
| **合计** | **150** | **120** | **30** |

## 评估指标

| 指标 | 说明 |
|------|------|
| Accuracy | 准确率 |
| F1-score (macro) | 宏平均F1分数 |
| Recall (macro) | 宏平均召回率 |
| Precision (macro) | 宏平均精确率 |

## 输入输出

| 项目 | 规格 |
|------|------|
| 输入形状 | `[batch_size, 1, seq_len]` |
| 输入类型 | 拉曼光谱强度值（Z-score标准化后） |
| 输出形状 | `[batch_size, num_classes]` |
| 输出类型 | 各类别的logits |

## 模型保存

训练完成后，模型权重保存在：
```
output/model_data/{task_name}/checkpoints/best/
├── encoder.pth      # 特征编码器权重
└── classifier.pth   # 分类器权重
```

训练历史保存在：
```
output/model_data/{task_name}/
├── history.json     # 训练历史（每epoch的loss、acc、f1、recall、precision）
└── meta.json        # 数据集元信息
```

## 使用示例

```python
from FCN import FaultClassificationNetwork

# 创建模型
model = FaultClassificationNetwork(num_classes=3)

# 加载权重
model.load_weights("output/model_data/PP_Starch/checkpoints/best")

# 推理
import torch
x = torch.randn(1, 1, 273)  # 单个样本
logits = model(x)
pred = logits.argmax(dim=1)
```
