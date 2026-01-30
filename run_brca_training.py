#!/usr/bin/env python3
"""
BRCA Cancer subtype classification
"""

import sys
import os
sys.path.append('.')

def run_brca_training():
    """START EDMBNET training"""
    print("=" * 60)
    print("BRCA classification training")
    print("=" * 60)
    print()

    print("check datasets...")
    data_path = r'/home/amax/4t/amax/CGLIU/CODE/3/data/BRCA'

    if not os.path.exists(data_path):
        print(f"error:path does not exist {data_path}")
        return False

    
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
        print(f"error: missing files: {missing_files}")
        return False

    print("✓files checked")
    print()

    print("START training...")
    try:
        from src.brca_multi_main import brca_main
        from configuration.config_brca_multi import args

        print("config:")
        print(f"  - class_num: {args.class_num}")
        print(f"  - batch_size: {args.batch_size}")
        print(f"  - lr: {args.lr}")
        print(f"  - train_epoch: {args.train_epoch}")
        print()

       
        brca_main(args)

        return True

    except Exception as e:
        print(f"START failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_system():
    ""testing"""
    print("run test...")
    print()

    try:
        from simple_test import main as test_main
        success = test_main()
        return success
    except Exception as e:
        print(f"TEST failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    "
    print("Cancer subtype classification")
    print()

    

    choice = input("Please type (1-3): ").strip()

    if choice == '1':
        print()
        success = test_system()
        if success:
            print("\n✓ system checked")
        else:
            print("\n✗ failed")

    elif choice == '2':
        print()
        success = run_brca_training()
        if success:
            print("\n✓ training complete")
        else:
            print("\n✗ failure")

    elif choice == '3':
        print("Exist")
        return

    else:
        print("invalid selection")


if __name__ == '__main__':
    main()

