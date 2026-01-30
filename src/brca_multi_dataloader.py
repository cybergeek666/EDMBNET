import torch
import numpy as np
from datasets.brca_dataset import create_brca_dataloader


def brca_multi_dataloader(train, args):
    """BRCA 多模态数据加载器"""

    # 设置模态列表
    view_list = [1, 2, 3]  # 三个模态

    if train:
        mode = 'train'
        shuffle = True
    else:
        mode = 'test'
        shuffle = False

    # 创建数据加载器
    data_loader = create_brca_dataloader(
        data_path=args.data_root,
        view_list=view_list,
        batch_size=args.batch_size,
        mode=mode,
        miss_modal=args.miss_modal if args.miss_modal > 0 else None,
        shuffle=shuffle,
        num_workers=0  # 设置为0避免多进程问题
    )

    return data_loader


def custom_collate_fn(batch):
    """自定义 collate 函数，将字典格式转换为元组格式"""
    modal_1_batch = []
    modal_2_batch = []
    modal_3_batch = []
    labels_batch = []

    for sample in batch:
        modal_1_batch.append(sample['modal_1'])
        modal_2_batch.append(sample['modal_2'])
        modal_3_batch.append(sample['modal_3'])
        labels_batch.append(sample['label'])

    # 转换为numpy数组
    modal_1_batch = np.array(modal_1_batch, dtype=np.float32)
    modal_2_batch = np.array(modal_2_batch, dtype=np.float32)
    modal_3_batch = np.array(modal_3_batch, dtype=np.float32)
    labels_batch = np.array(labels_batch, dtype=np.int64)

    # 返回 (data_list, labels) 格式
    return [modal_1_batch, modal_2_batch, modal_3_batch], labels_batch


def collate_fn(batch):
    """自定义 collate 函数处理不同长度的序列数据"""
    # 找到最长的序列长度
    max_length = max([item['modal_1'].shape[0] for item in batch])

    # 填充序列
    padded_batch = []
    labels = []

    for item in batch:
        padded_item = {}

        # 填充每个模态的数据
        for key in ['modal_1', 'modal_2', 'modal_3']:
            data = item[key]
            if len(data.shape) == 1:  # 如果是一维特征
                # 转换为二维
                data = data.reshape(-1, 1)
            padded_item[key] = torch.FloatTensor(data)

        labels.append(item['label'])
        padded_batch.append(padded_item)

    # 转换为张量
    labels = torch.LongTensor(labels)

    return padded_batch, labels


