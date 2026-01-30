#!/usr/bin/env python3
"""
快速测试 BRCA 数据集是否能正常加载
"""

import sys
sys.path.append('.')

try:
    from datasets.brca_dataset import BRCA_Dataset
    print("✓ 成功导入 BRCA_Dataset")

    # 测试创建数据集
    dataset = BRCA_Dataset(
        data_path=r'C:\Users\陶雪峰\Desktop\GAN\MOGONET-main\BRCA',
        view_list=[1, 2, 3],
        mode='train',
        miss_modal=None
    )
    print("✓ 成功创建训练数据集")
    print(f"训练集大小: {len(dataset)}")

    # 测试一个样本
    sample = dataset[0]
    print("✓ 成功读取样本")
    print(f"样本键: {list(sample.keys())}")

    print("\n🎉 所有测试通过！编码问题已修复。")

except Exception as e:
    print(f"✗ 错误: {str(e)}")
    import traceback
    traceback.print_exc()
