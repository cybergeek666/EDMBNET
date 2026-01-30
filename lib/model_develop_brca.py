"""针对 BRCA 数据集的模型训练函数"""

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import time
import csv
import os
from sklearn.metrics import accuracy_score, f1_score, classification_report
from lib.model_develop_utils import GradualWarmupScheduler


def calc_accuracy_brca(model, loader, verbose=False):
    """
    计算 BRCA 模型的准确率
    """
    mode_saved = model.training
    model.train(False)
    device = torch.device('cpu')
    model.to(device)

    outputs_full = []
    labels_full = []

    for batch_data, batch_labels in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):
        # 解包数据
        modal_data = batch_data  # 这是包含三个模态数据的列表

        with torch.no_grad():
            # 将每个模态的数据转换为张量
            modal_1 = torch.FloatTensor(modal_data[0]).to(device)
            modal_2 = torch.FloatTensor(modal_data[1]).to(device)
            modal_3 = torch.FloatTensor(modal_data[2]).to(device)

            outputs_batch = model(modal_1, modal_2, modal_3)
            if isinstance(outputs_batch, tuple):
                outputs_batch = outputs_batch[0]
            elif not isinstance(outputs_batch, torch.Tensor):
                outputs_batch = torch.tensor(outputs_batch, device=device)

        outputs_full.append(outputs_batch)
        labels_full.append(torch.LongTensor(batch_labels).to(device))

    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    accuracy = float("%.6f" % accuracy)

    return accuracy, labels_full.cpu().numpy(), labels_predicted.cpu().numpy()


def train_base_multi_brca(model, cost, optimizer, train_loader, test_loader, scheduler=None, args=None):
    """
    适用于 BRCA 多模态分类的基础训练函数
    """
    print(args)

    # 初始化计时器
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # 保存参数
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    # 学习率衰减
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int32(args.train_epoch * 1 / 6),
                                                                              np.int32(args.train_epoch * 2 / 6),
                                                                              np.int32(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    else:
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1)

    # 训练初始化
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    log_list = []

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # 训练循环
    while epoch < epoch_num:
        model.train()
        # 临时调试代码
        # for i, batch in enumerate(train_loader):
        #     print(f"Batch type: {type(batch)}")
        #     print(f"Batch length: {len(batch)}")
        #     print(batch)
        #     if i >= 1:  # 只看第一个batch
        #         break

        # 然后根据实际情况调整解包
        for batch_idx, (batch_data, target) in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1

            # 解包数据
            modal_data = batch_data

            # 清零梯度
            for p in model.parameters():
                p.grad = None

            model.args.epoch = epoch

            # 将数据转换为张量并确保在正确的设备上
            device = next(model.parameters()).device
            modal_1 = torch.FloatTensor(modal_data[0]).to(device)
            modal_2 = torch.FloatTensor(modal_data[1]).to(device)
            modal_3 = torch.FloatTensor(modal_data[2]).to(device)

            # 前向传播
            output = model(modal_1, modal_2, modal_3)
            if isinstance(output, tuple):
                output = output[0]

            # 计算损失
            target_tensor = torch.LongTensor(target).to(device)
            loss = cost(output, target_tensor)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        # 测试
        model.eval()
        accuracy_test, true_labels, pred_labels = calc_accuracy_brca(model, loader=test_loader, verbose=True)

        # 计算详细的分类指标
        f1_macro = f1_score(true_labels, pred_labels, average='macro')
        f1_weighted = f1_score(true_labels, pred_labels, average='weighted')

        if accuracy_test > accuracy_best and epoch > 5:  # 跳过前几个epoch
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path)

        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(f1_macro)
        log_list.append(f1_weighted)
        log_list.append(accuracy_best)

        print("Epoch {}, loss={:.5f}, accuracy_test={:.5f}, f1_macro={:.5f}, f1_weighted={:.5f}, accuracy_best={:.5f}".format(
            epoch, train_loss / len(train_loader), accuracy_test, f1_macro, f1_weighted, accuracy_best))

        train_loss = 0

        # 学习率调度
        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 10:
            print(epoch, optimizer.param_groups[0]['lr'])

        # 保存模型和参数
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        # 保存日志
        with open(log_dir, 'a+', newline='') as f:
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []

        # 学习率调度
        if scheduler is not None:
            scheduler.step()

        epoch = epoch + 1

    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)
