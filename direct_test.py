#!/usr/bin/env python3
"""
直接测试 BRCA 数据集加载 - 使用与 MOGONET 完全相同的方法
"""

import sys
sys.path.append('.')

import numpy as np
import os

def test_direct_loading():
    """直接测试数据加载"""
    data_path = r'C:\Users\陶雪峰\Desktop\GAN\MOGONET-main\BRCA'

    print("直接测试 BRCA 数据集加载...")
    print("=" * 50)

    try:
        # 加载标签 - 参考 MOGONET 的方式
        label_file = os.path.join(data_path, 'labels_tr.csv')
        labels = np.loadtxt(label_file, delimiter=',')
        labels = labels.astype(int)

        print(f"✓ 成功读取 {len(labels)} 个标签")
        print(f"标签范围: {labels.min()} - {labels.max()}")
        print(f"前5个标签: {labels[:5]}")

        # 加载数据文件 1
        data_file = os.path.join(data_path, '1_tr.csv')
        data = np.loadtxt(data_file, delimiter=',')

        print(f"✓ 成功读取数据文件 1: {data.shape}")
        print(f"数据类型: {data.dtype}")
        print(f"数据范围: {data.min()".6f"} - {data.max()".6f"}")

        # 加载数据文件 2
        data_file = os.path.join(data_path, '2_tr.csv')
        data = np.loadtxt(data_file, delimiter=',')

        print(f"✓ 成功读取数据文件 2: {data.shape}")

        # 加载数据文件 3
        data_file = os.path.join(data_path, '3_tr.csv')
        data = np.loadtxt(data_file, delimiter=',')

        print(f"✓ 成功读取数据文件 3: {data.shape}")

        print("\n🎉 所有数据文件加载成功！")
        print("问题已解决 - 使用与 MOGONET 相同的方法即可。")

        return True

    except Exception as e:
        print(f"✗ 加载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_direct_loading()
