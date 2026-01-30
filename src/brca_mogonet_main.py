import sys
import os
sys.path.append('..')

from models.brca_baseline import BRCA_Baseline_Simple
from src.brca_multi_dataloader import brca_multi_dataloader
from configuration.config_brca_mogonet import args  # 使用MOGONET风格配置
import torch
import torch.nn as nn
from lib.model_develop_brca import train_base_multi_brca
from lib.processing_utils import get_file_list
import torch.optim as optim
import numpy as np
import random


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)  # Disabled for CPU usage


def brca_mogonet_main(args):
    print("=" * 60)
    print("MOGONET风格的edANet训练")
    print("=" * 60)
    print("核心改进:")
    print("1. 简化的图卷积网络（参考MOGONET的GCN）")
    print("2. 注意力机制的多模态融合")
    print("3. 邻接矩阵建模样本关系")
    print("4. 分阶段特征提取")
    print("=" * 60)

    # 创建数据加载器
    train_loader = brca_multi_dataloader(train=True, args=args)
    test_loader = brca_multi_dataloader(train=False, args=args)

    # 设置日志名称
    args.log_name = args.name + '.csv'
    args.model_name = args.name

    # 初始化模型 - 使用简化的MOGONET风格模型
    model = BRCA_Baseline_Simple(args, input_dims=[1000, 1000, 503], hidden_dim=256, num_classes=args.class_num)

    # 强制使用CPU
    device = torch.device('cpu')
    model.to(device)
    print("CPU is using")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    # 使用Adam优化器（参考MOGONET）
    optimizer = optim.Adam(filter(lambda param: param.requires_grad, model.parameters()),
                          lr=args.lr, weight_decay=args.weight_decay)

    # 添加学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # 设置训练参数
    args.retrain = False

    # 开始训练
    print("\n开始训练MOGONET风格的edANet...")
    train_base_multi_brca(model=model, cost=criterion, optimizer=optimizer,
                          train_loader=train_loader, test_loader=test_loader,
                          scheduler=scheduler, args=args)


if __name__ == '__main__':
    seed_torch(42)  # 设置随机种子确保结果可重复
    brca_mogonet_main(args=args)
