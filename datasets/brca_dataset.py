```python
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
        BRCA Multimodal Cancer Subtype Classification Dataset

        Args:
            data_path: Root directory path for data
            view_list: List of modalities, e.g., [1, 2, 3]
            mode: 'train' or 'test'
            miss_modal: Indices of missing modalities, if None then use all modalities
            fill: Fill value for missing modalities
            transform: Data transformation
        """
        self.data_path = data_path
        self.view_list = view_list
        self.mode = mode
        self.miss_modal = miss_modal
        self.fill = fill
        self.transform = transform

        # Load data
        self.data_list, self.labels = self._load_data()

        # Data preprocessing - Standardization
        self._preprocess_data()

        # # Simulate modality missing
        # if self.miss_modal is not None:
        #     self._apply_modality_dropout()

    def _load_data(self):
        """Load data and labels"""
        data_list = []
        view_list = [1, 2, 3]  # Three modalities
        if args.miss_modal == 0:
            # Use all three modalities
            view_list = [1, 2, 3]

        elif args.miss_modal == int(1):
            # Missing modality 1, use modalities 2 and 3
            view_list = [-1, 2, 3]

        elif args.miss_modal == 2:
            # Missing modality 2, use modalities 1 and 3
            view_list = [1, -1, 3]

        elif args.miss_modal == 3:
            # Missing modality 3, use modalities 1 and 2
            view_list = [1, 2, -1]

        elif args.miss_modal == 4:
            # Missing modality 3, use modalities 1 and 2
            view_list = [1, -1, -1]

        elif args.miss_modal == 5:
            # Missing modality 3, use modalities 1 and 2
            view_list = [-1, 2, -1]

        elif args.miss_modal == 6:
            # Missing modality 3, use modalities 1 and 2
            view_list = [-1, -1, 3]
        else:
            # Default: use all modalities
            used_modals = [1, 2, 3]

        # Load labels - referencing MOGONET's method
        if self.mode == 'train':
            label_file = os.path.join(self.data_path, 'labels_tr.csv')
        else:
            label_file = os.path.join(self.data_path, 'labels_te.csv')

        # Use a more robust method to load labels
        labels = self._load_csv_file(label_file)
        labels = labels.astype(int)

        # Load each modality's data - referencing MOGONET's method
        for view in view_list:
            if view == -1:
                data = torch.zeros([612, 1000])
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
        """Data preprocessing - Standardization"""
        from sklearn.preprocessing import StandardScaler

        print("Performing data preprocessing...")
        self.scalers = []

        for i, data in enumerate(self.data_list):
            scaler = StandardScaler()
            # Reshape data to 2D for standardization
            original_shape = data.shape
            data_2d = data.reshape(-1, original_shape[-1])

            # Standardize
            data_normalized = scaler.fit_transform(data_2d)

            # Restore original shape
            self.data_list[i] = data_normalized.reshape(original_shape)

            self.scalers.append(scaler)
            print(f"  Modality {i+1} standardization completed, shape: {self.data_list[i].shape}")

    def _load_csv_file(self, file_path):
        """Simplified CSV file loading method, referencing original MOGONET code"""
        print(f"Loading file: {file_path}")
        
        # First try direct loading (original MOGONET method)
        try:
            data = np.loadtxt(file_path, delimiter=',')
            print(f"Direct loading successful, data shape: {data.shape}")
            return data
        except UnicodeDecodeError as e:
            print(f"UTF-8 decoding failed: {e}")
            print("Attempting latin-1 encoding...")
            
            try:
                data = np.loadtxt(file_path, delimiter=',', encoding='latin-1')
                print(f"latin-1 encoding successful, data shape: {data.shape}")
                return data
            except Exception as e2:
                print(f"latin-1 encoding failed: {e2}")
                raise ValueError(f"Cannot load file {file_path}: {e2}")
        except Exception as e:
            print(f"Other error: {e}")
            raise ValueError(f"Cannot load file {file_path}: {e}")

    def _apply_modality_dropout(self):
        """Apply modality missing"""
        if self.miss_modal is None:
            return

        # Replace missing modality data with fill value
        for idx in self.miss_modal:
            if idx < len(self.data_list):
                self.data_list[idx] = np.full_like(self.data_list[idx], self.fill)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """Get a single sample"""
        # Get each modality's data
        modal_data = {}
        for i, data in enumerate(self.data_list):
            modal_data[f'modal_{i+1}'] = data[idx]

        # Get label
        label = self.labels[idx]

        sample = {
            'modal_1': modal_data['modal_1'],
            'modal_2': modal_data['modal_2'],
            'modal_3': modal_data['modal_3'],
            'label': label
        }

        # Apply transformation
        if self.transform:
            sample = self.transform(sample)

        return sample


def create_brca_dataloader(data_path, view_list, batch_size=64, mode='train',
                          miss_modal=None, shuffle=True, num_workers=0):
    """Create BRCA data loader"""

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
```
