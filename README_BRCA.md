# BRCA 多模态癌症亚型分类

本项目基于 MMANet 框架，将原有的图片真伪检测任务修改为癌症亚型分类任务，使用 BRCA 数据集进行多模态学习。

## 数据集说明

- **数据集**: BRCA (Breast Cancer) 多模态数据集
- **模态数量**: 3个模态
- **分类任务**: 5分类癌症亚型分类
- **数据格式**: CSV 文件，参考 MOGONET-main/BRCA 数据集
- **数据路径**: `C:\Users\陶雪峰\Desktop\GAN\MOGONET-main\BRCA`

## 文件结构

```
edANet/classification/
├── datasets/
│   ├── brca_dataset.py          # BRCA 数据集类
│   └── ...
├── models/
│   ├── brca_baseline.py         # BRCA 模型定义
│   └── ...
├── src/
│   ├── brca_multi_dataloader.py # BRCA 数据加载器
│   ├── brca_multi_main.py       # BRCA 主训练文件
│   └── ...
├── configuration/
│   ├── config_brca_multi.py     # BRCA 配置参数
│   └── ...
├── lib/
│   ├── model_develop_brca.py    # BRCA 训练函数
│   └── ...
├── test_brca.py                 # 系统集成测试
└── simple_test.py               # 简单功能测试
```

## 主要修改

### 1. 数据集类 (`brca_dataset.py`)
- 继承 `torch.utils.data.Dataset`
- 支持多模态 CSV 数据加载
- 支持模态缺失模拟
- 自动处理训练/测试数据分割

### 2. 模型架构 (`brca_baseline.py`)
- 全连接神经网络架构
- 三个独立的模态编码器
- 共享的融合层
- 支持模态 dropout
- 输出 5 分类结果

### 3. 配置参数 (`config_brca_multi.py`)
- 修改分类数量为 5
- 设置 BRCA 数据集路径
- 调整训练参数

### 4. 训练函数 (`model_develop_brca.py`)
- 适配新的数据格式
- 支持 F1 分数的计算
- 详细的分类指标报告

## 使用方法

### 1. 测试系统集成
```bash
cd edANet/classification
python simple_test.py
```

### 2. 运行完整训练
```bash
cd edANet/classification
python src/brca_multi_main.py
```

### 3. 运行系统集成测试
```bash
cd edANet/classification
python test_brca.py
```

## 数据格式

BRCA 数据集包含以下文件：
- `1_tr.csv`, `2_tr.csv`, `3_tr.csv`: 三个模态的训练数据
- `1_te.csv`, `2_te.csv`, `3_te.csv`: 三个模态的测试数据
- `labels_tr.csv`: 训练集标签 (0-4)
- `labels_te.csv`: 测试集标签 (0-4)

每行数据表示一个样本的特征向量。

## 模型特性

- **多模态融合**: 支持三个模态的联合学习
- **模态缺失处理**: 可以处理任意模态缺失的情况
- **端到端训练**: 完整的训练和测试流程
- **详细评估**: 提供准确率、F1 分数等多种评估指标

## 输出结果

训练过程中会生成：
- 模型权重文件 (`.pth`)
- 训练日志文件 (`.csv`)
- 模型参数文件 (`.pt`)

## 注意事项

1. 数据集路径在配置文件中已设置
2. 确保所有依赖库已安装
3. 训练过程会自动创建输出目录
4. 支持 CPU 运行，无需 GPU

## 扩展功能

系统支持以下扩展：
- 调整模态数量
- 修改分类数量
- 自定义模型架构
- 调整训练参数
