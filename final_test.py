#!/usr/bin/env python3
"""
最终测试 BRCA 数据集加载
"""

import sys
sys.path.append('.')

import chardet
import pandas as pd
import numpy as np
import os

def test_chardet_solution():
    """测试使用 chardet 自动检测编码的解决方案"""
    data_path = r'C:\Users\陶雪峰\Desktop\GAN\MOGONET-main\BRCA'

    print("使用 chardet 自动检测编码测试...")

    # 测试标签文件
    label_file = os.path.join(data_path, 'labels_tr.csv')

    try:
        # 检测编码
        with open(label_file, 'rb') as f:
            raw_data = f.read()
            detected_encoding = chardet.detect(raw_data)
            encoding = detected_encoding.get('encoding', 'utf-8')

        print(f"检测到的编码: {encoding}")
        print(f"检测置信度: {detected_encoding.get('confidence', 0)".2f"}")

        # 使用检测到的编码读取文件
        labels_df = pd.read_csv(label_file, header=None, encoding=encoding)
        labels = labels_df.values.flatten().astype(int)

        print(f"✓ 成功读取 {len(labels)} 个标签")
        print(f"标签范围: {labels.min()} - {labels.max()}")

        # 测试数据文件
        data_file = os.path.join(data_path, '1_tr.csv')

        with open(data_file, 'rb') as f:
            raw_data = f.read()
            detected_encoding = chardet.detect(raw_data)
            data_encoding = detected_encoding.get('encoding', 'utf-8')

        print(f"数据文件编码: {data_encoding}")

        data_df = pd.read_csv(data_file, header=None, encoding=data_encoding)
        data = data_df.values.astype(np.float32)

        print(f"✓ 成功读取数据: {data.shape}")

        print("\n🎉 chardet 解决方案成功！")

        # 返回找到的编码
        return encoding, data_encoding

    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    encoding, data_encoding = test_chardet_solution()
