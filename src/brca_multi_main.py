import sys
import os
sys.path.append('..')

from models.brca_baseline import BRCA_Baseline
from src.brca_multi_dataloader import brca_multi_dataloader
from configuration.config_brca_multi import args
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


def brca_main(args):
    
    train_loader = brca_multi_dataloader(train=True, args=args)
    test_loader = brca_multi_dataloader(train=False, args=args)

    
    args.log_name = args.name + '.csv'
    args.model_name = args.name

    
    model = BRCA_Baseline(args, input_dim=[1000, 1000, 503], hidden_dim=256, num_classes=args.class_num)

    
    device = torch.device('cpu')
    model.to(device)
    print("CPU is using")

   
    criterion = nn.CrossEntropyLoss()

   
    optimizer = optim.Adam(filter(lambda param: param.requires_grad, model.parameters()),
                          lr=args.lr, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

   
    args.retrain = False

  
    train_base_multi_brca(model=model, cost=criterion, optimizer=optimizer,
                          train_loader=train_loader, test_loader=test_loader,
                          scheduler=scheduler, args=args)


if __name__ == '__main__':
    brca_main(args=args)

