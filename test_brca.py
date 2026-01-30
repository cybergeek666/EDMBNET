#!/usr/bin/env python3
"""
测试 BRCA 数据集加载和模型运行的脚本
"""

import sys
import os
sys.path.append('.')

from datasets.brca_dataset import BRCA_Dataset
from models.brca_baseline import BRCA_Baseline
from configuration.config_brca_multi import args
import torch
import torch.nn as nn
import numpy as np


def test_data_loading():
    """测试数据加载"""
    print("测试数据加载...")

    # 创建数据集实例
    dataset = BRCA_Dataset(
        data_path=args.data_root,
        view_list=[1, 2, 3],
        mode='train',
        miss_modal=None
    )

    print(f"训练集大小: {len(dataset)}")
    print(f"标签范围: {dataset.labels.min()} - {dataset.labels.max()}")

    # 获取一个样本
    sample = dataset[0]
    print(f"样本键: {sample.keys()}")

    for key in ['modal_1', 'modal_2', 'modal_3']:
        if key in sample:
            print(f"{key} 形状: {sample[key].shape}")

    print(f"标签: {sample['label']}")

    return dataset


def test_model():
    """测试模型"""
    print("\n测试模型...")

    # 创建模型
    model = BRCA_Baseline(args, input_dim=2000, hidden_dim=512, num_classes=args.class_num)

    # 创建虚拟输入数据
    batch_size = 2
    modal_1 = torch.randn(batch_size, 2000)
    modal_2 = torch.randn(batch_size, 2000)
    modal_3 = torch.randn(batch_size, 2000)

    # 前向传播
    output, x1, x2, x3 = model(modal_1, modal_2, modal_3)

    print(f"模型输出形状: {output.shape}")
    print(f"编码器输出形状: x1={x1.shape}, x2={x2.shape}, x3={x3.shape}")

    return model


def test_training_setup():
    """测试训练设置"""
    print("\n测试训练设置...")

    # 创建模型
    model = BRCA_Baseline(args, input_dim=2000, hidden_dim=512, num_classes=args.class_num)

    # 创建损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # 创建虚拟目标
    batch_size = 4
    target = torch.randint(0, args.class_num, (batch_size,))

    # 前向传播
    modal_1 = torch.randn(batch_size, 2000)
    modal_2 = torch.randn(batch_size, 2000)
    modal_3 = torch.randn(batch_size, 2000)

    output, _, _, _ = model(modal_1, modal_2, modal_3)

    # 计算损失
    loss = criterion(output, target)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"损失值: {loss.item():.4f}")
    print("训练设置测试完成")


def main():
    print("开始测试 BRCA 系统集成...")

    try:
        # 测试数据加载
        dataset = test_data_loading()

        # 测试模型
        model = test_model()

        # 测试训练设置
        test_training_setup()

        print("\n所有测试通过！系统集成成功。")
        return True

    except Exception as e:
        print(f"\n测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
