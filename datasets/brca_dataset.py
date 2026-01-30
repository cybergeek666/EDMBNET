import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import random
from configuration.config_brca_mogonet import args

class BRCA_Dataset(Dataset):
    def __init__(self, data_path, view_list, mode='train', miss_modal=None, fill=0, transform=None):
        """
        BRCA 多模态癌症亚型分类数据集

        Args:
            data_path: 数据根目录路径
            view_list: 模态列表，如 [1, 2, 3]
            mode: 'train' 或 'test'
            miss_modal: 缺失模态的索引，如果为 None 则使用所有模态
            fill: 缺失模态填充值
            transform: 数据变换
        """
        self.data_path = data_path
        self.view_list = view_list
        self.mode = mode
        self.miss_modal = miss_modal
        self.fill = fill
        self.transform = transform

        # 加载数据
        self.data_list, self.labels = self._load_data()

        # 数据预处理 - 标准化
        self._preprocess_data()

        # # 模拟模态缺失
        # if self.miss_modal is not None:
        #     self._apply_modality_dropout()

    def _load_data(self):
        """加载数据和标签"""
        data_list = []
        view_list = [1, 2, 3]  # 三个模态
        if args.miss_modal == 0:
            # 使用所有三个模态
            view_list = [1, 2, 3]

        elif args.miss_modal == int(1):
            # 缺失模态1，使用模态2和3
            view_list = [-1, 2, 3]

        elif args.miss_modal == 2:
            # 缺失模态2，使用模态1和3
            view_list = [1,-1, 3]

        elif args.miss_modal == 3:
            # 缺失模态3，使用模态1和2
            view_list = [1, 2, -1]

        elif args.miss_modal == 4:
            # 缺失模态3，使用模态1和2
            view_list = [1, -1, -1]

        elif args.miss_modal == 5:
            # 缺失模态3，使用模态1和2
            view_list = [-1, 2, -1]

        elif args.miss_modal == 6:
            # 缺失模态3，使用模态1和2
            view_list = [-1, -1, 3]
        else:
            # 默认使用所有模态
            used_modals = [1, 2, 3]


        # 加载标签 - 参考 MOGONET 的方式
        if self.mode == 'train':
            label_file = os.path.join(self.data_path, 'labels_tr.csv')
        else:
            label_file = os.path.join(self.data_path, 'labels_te.csv')

        # 使用更健壮的方法加载标签
        labels = self._load_csv_file(label_file)
        labels = labels.astype(int)

        # 加载各模态数据 - 参考 MOGONET 的方式
        for view in view_list:
            if view == -1:
                data = torch.zeros([612,1000])
                data_list.append(data)
                continue
            if self.mode == 'train':
                data_file = os.path.join(self.data_path, f'{view}_tr.csv')
            else:
                data_file = os.path.join(self.data_path, f'{view}_te.csv')

            data = self._load_csv_file(data_file)

            data_list.append(data)

        return data_list, labels

    def _preprocess_data(self):
        """数据预处理 - 标准化"""
        from sklearn.preprocessing import StandardScaler

        print("进行数据预处理...")
        self.scalers = []

        for i, data in enumerate(self.data_list):
            scaler = StandardScaler()
            # 重塑数据为2D进行标准化
            original_shape = data.shape
            data_2d = data.reshape(-1, original_shape[-1])

            # 标准化
            data_normalized = scaler.fit_transform(data_2d)

            # 恢复原始形状
            self.data_list[i] = data_normalized.reshape(original_shape)

            self.scalers.append(scaler)
            print(f"  模态{i+1}标准化完成，形状: {self.data_list[i].shape}")

    def _load_csv_file(self, file_path):
        """简化的CSV文件加载方法，参考原始MOGONET代码"""
        print(f"正在加载文件: {file_path}")
        
        # 首先尝试直接加载（原始MOGONET方法）
        try:
            data = np.loadtxt(file_path, delimiter=',')
            print(f"直接加载成功，数据形状: {data.shape}")
            return data
        except UnicodeDecodeError as e:
            print(f"UTF-8解码失败: {e}")
            print("尝试使用latin-1编码...")
            
            try:
                data = np.loadtxt(file_path, delimiter=',', encoding='latin-1')
                print(f"latin-1编码成功，数据形状: {data.shape}")
                return data
            except Exception as e2:
                print(f"latin-1编码失败: {e2}")
                raise ValueError(f"无法加载文件 {file_path}: {e2}")
        except Exception as e:
            print(f"其他错误: {e}")
            raise ValueError(f"无法加载文件 {file_path}: {e}")

    def _apply_modality_dropout(self):
        """应用模态缺失"""
        if self.miss_modal is None:
            return

        # 将缺失模态的数据替换为填充值
        for idx in self.miss_modal:
            if idx < len(self.data_list):
                self.data_list[idx] = np.full_like(self.data_list[idx], self.fill)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """获取单个样本"""
        # 获取各模态数据
        modal_data = {}
        for i, data in enumerate(self.data_list):
            modal_data[f'modal_{i+1}'] = data[idx]

        # 获取标签
        label = self.labels[idx]

        sample = {
            'modal_1': modal_data['modal_1'],
            'modal_2': modal_data['modal_2'],
            'modal_3': modal_data['modal_3'],
            'label': label
        }

        # 应用变换
        if self.transform:
            sample = self.transform(sample)

        return sample


def create_brca_dataloader(data_path, view_list, batch_size=64, mode='train',
                          miss_modal=None, shuffle=True, num_workers=0):
    """创建 BRCA 数据加载器"""

    dataset = BRCA_Dataset(
        data_path=data_path,
        view_list=view_list,
        mode=mode,
        miss_modal=miss_modal
    )

    from src.brca_multi_dataloader import custom_collate_fn

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=custom_collate_fn
    )
