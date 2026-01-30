#!/usr/bin/env python3
"""
BRCA 癌症亚型分类训练启动脚本
"""

import sys
import os
sys.path.append('.')

def run_brca_training():
    """启动 BRCA 训练"""
    print("=" * 60)
    print("BRCA 多模态癌症亚型分类训练")
    print("=" * 60)
    print()

    print("检查数据集...")
    data_path = r'/home/amax/4t/amax/CGLIU/CODE/3/data/BRCA'

    if not os.path.exists(data_path):
        print(f"错误: 数据集路径不存在: {data_path}")
        return False

    # 检查必需的文件
    required_files = [
        '1_tr.csv', '2_tr.csv', '3_tr.csv',
        '1_te.csv', '2_te.csv', '3_te.csv',
        'labels_tr.csv', 'labels_te.csv'
    ]

    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(data_path, file)):
            missing_files.append(file)

    if missing_files:
        print(f"错误: 缺少以下文件: {missing_files}")
        return False

    print("✓ 数据集检查通过")
    print()

    print("启动训练...")
    try:
        from src.brca_multi_main import brca_main
        from configuration.config_brca_multi import args

        print("训练配置:")
        print(f"  - 分类数量: {args.class_num}")
        print(f"  - 批次大小: {args.batch_size}")
        print(f"  - 学习率: {args.lr}")
        print(f"  - 训练轮数: {args.train_epoch}")
        print()

        # 启动训练
        brca_main(args)

        return True

    except Exception as e:
        print(f"训练启动失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_system():
    """测试系统集成"""
    print("运行系统集成测试...")
    print()

    try:
        from simple_test import main as test_main
        success = test_main()
        return success
    except Exception as e:
        print(f"系统测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("欢迎使用 BRCA 癌症亚型分类系统")
    print()

    # 询问用户操作
    print("请选择操作:")
    print("1. 运行系统测试")
    print("2. 启动完整训练")
    print("3. 退出")

    choice = input("请输入选择 (1-3): ").strip()

    if choice == '1':
        print()
        success = test_system()
        if success:
            print("\n✓ 系统测试通过，可以开始训练")
        else:
            print("\n✗ 系统测试失败，请检查代码和数据")

    elif choice == '2':
        print()
        success = run_brca_training()
        if success:
            print("\n✓ 训练完成")
        else:
            print("\n✗ 训练启动失败")

    elif choice == '3':
        print("退出系统")
        return

    else:
        print("无效选择，请重新运行脚本")


if __name__ == '__main__':
    main()
