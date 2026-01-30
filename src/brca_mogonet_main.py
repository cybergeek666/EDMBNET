import sys
import os
sys.path.append('..')

from models.brca_baseline import BRCA_Baseline_Simple
from src.brca_multi_dataloader import brca_multi_dataloader
from configuration.config_brca_mogonet import args  # Using MOGONET-style configuration
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
    print("EDMBNET Training")
    print("=" * 60)
    print("Core Improvements:")  
    print("4. Phased Feature Extraction")
    print("=" * 60)

    # Create data loaders
    train_loader = brca_multi_dataloader(train=True, args=args)
    test_loader = brca_multi_dataloader(train=False, args=args)

    # Set log names
    args.log_name = args.name + '.csv'
    args.model_name = args.name

    # Initialize model - using simplified MOGONET-style model
    model = BRCA_Baseline_Simple(args, input_dims=[1000, 1000, 503], hidden_dim=256, num_classes=args.class_num)

    # Force CPU usage
    device = torch.device('cpu')
    model.to(device)
    print("CPU is being used")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # Use Adam optimizer (referencing MOGONET)
    optimizer = optim.Adam(filter(lambda param: param.requires_grad, model.parameters()),
                          lr=args.lr, weight_decay=args.weight_decay)

    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # Set training parameters
    args.retrain = False

    # Start training
    print("\nStarting EDMBNET training...")
    train_base_multi_brca(model=model, cost=criterion, optimizer=optimizer,
                          train_loader=train_loader, test_loader=test_loader,
                          scheduler=scheduler, args=args)


if __name__ == '__main__':
    seed_torch(42)  # Set random seed to ensure reproducible results
    brca_mogonet_main(args=args)
