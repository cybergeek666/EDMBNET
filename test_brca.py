```python
#!/usr/bin/env python3
"""
Script for testing BRCA dataset loading and model execution
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
    """Test data loading"""
    print("Testing data loading...")

    # Create dataset instance
    dataset = BRCA_Dataset(
        data_path=args.data_root,
        view_list=[1, 2, 3],
        mode='train',
        miss_modal=None
    )

    print(f"Training set size: {len(dataset)}")
    print(f"Label range: {dataset.labels.min()} - {dataset.labels.max()}")

    # Get a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")

    for key in ['modal_1', 'modal_2', 'modal_3']:
        if key in sample:
            print(f"{key} shape: {sample[key].shape}")

    print(f"Label: {sample['label']}")

    return dataset


def test_model():
    """Test model"""
    print("\nTesting model...")

    # Create model
    model = BRCA_Baseline(args, input_dim=2000, hidden_dim=512, num_classes=args.class_num)

    # Create dummy input data
    batch_size = 2
    modal_1 = torch.randn(batch_size, 2000)
    modal_2 = torch.randn(batch_size, 2000)
    modal_3 = torch.randn(batch_size, 2000)

    # Forward pass
    output, x1, x2, x3 = model(modal_1, modal_2, modal_3)

    print(f"Model output shape: {output.shape}")
    print(f"Encoder output shapes: x1={x1.shape}, x2={x2.shape}, x3={x3.shape}")

    return model


def test_training_setup():
    """Test training setup"""
    print("\nTesting training setup...")

    # Create model
    model = BRCA_Baseline(args, input_dim=2000, hidden_dim=512, num_classes=args.class_num)

    # Create loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Create dummy target
    batch_size = 4
    target = torch.randint(0, args.class_num, (batch_size,))

    # Forward pass
    modal_1 = torch.randn(batch_size, 2000)
    modal_2 = torch.randn(batch_size, 2000)
    modal_3 = torch.randn(batch_size, 2000)

    output, _, _, _ = model(modal_1, modal_2, modal_3)

    # Calculate loss
    loss = criterion(output, target)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Loss value: {loss.item():.4f}")
    print("Training setup test completed")


def main():
    print("Starting BRCA system integration test...")

    try:
        # Test data loading
        dataset = test_data_loading()

        # Test model
        model = test_model()

        # Test training setup
        test_training_setup()

        print("\nAll tests passed! System integration successful.")
        return True

    except Exception as e:
        print(f"\nTest failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
