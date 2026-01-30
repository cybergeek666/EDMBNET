#!/usr/bin/env python3
"""
测试 BRCA 数据集编码问题的解决方案
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import os

def detect_file_encoding():
    """检测文件编码"""
    data_path = r'C:\Users\陶雪峰\Desktop\GAN\MOGONET-main\BRCA'

    print("检测 BRCA 数据集编码...")

    # 测试标签文件
    label_file = os.path.join(data_path, 'labels_tr.csv')

    # 读取文件的前几个字节来检测编码
    with open(label_file, 'rb') as f:
        raw_data = f.read(10)  # 读取前10个字节
        print(f"文件前10字节 (十六进制): {[hex(b) for b in raw_data]}")

    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'utf-16', 'ascii']

    for encoding in encodings_to_try:
        try:
            print(f"\n尝试编码: {encoding}")
            labels_df = pd.read_csv(label_file, header=None, encoding=encoding)
            labels = labels_df.values.flatten().astype(int)
            print(f"  ✓ 成功读取 {len(labels)} 个标签")
            print(f"  标签范围: {labels.min()} - {labels.max()}")
            print(f"  前5个标签: {labels[:5]}")
            return encoding  # 返回成功的编码
        except Exception as e:
            print(f"  ✗ 编码 {encoding} 失败: {str(e)}")

    return None

def test_with_encoding(encoding):
    """使用指定编码测试完整的数据加载"""
    data_path = r'C:\Users\陶雪峰\Desktop\GAN\MOGONET-main\BRCA'

    print(f"\n使用编码 {encoding} 测试完整数据加载...")

    try:
        # 读取标签
        label_file = os.path.join(data_path, 'labels_tr.csv')
        labels_df = pd.read_csv(label_file, header=None, encoding=encoding)
        labels = labels_df.values.flatten().astype(int)
        print(f"✓ 成功读取 {len(labels)} 个标签")

        # 读取数据
        data_file = os.path.join(data_path, '1_tr.csv')
        data_df = pd.read_csv(data_file, header=None, encoding=encoding)
        data = data_df.values.astype(np.float32)
        print(f"✓ 成功读取数据: {data.shape}")

        return True

    except Exception as e:
        print(f"✗ 读取失败: {str(e)}")
        return False

if __name__ == "__main__":
    # 检测编码
    encoding = detect_file_encoding()

    if encoding:
        print(f"\n🎉 找到正确的编码: {encoding}")
        success = test_with_encoding(encoding)
        if success:
            print("✅ 数据加载测试成功！")
    else:
        print("\n❌ 无法确定正确的编码格式")
