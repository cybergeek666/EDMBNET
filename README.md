# EDMBNET

BRCA Multiomics Cancer Subtype Classification


Dataset Description
Dataset: BRCA (Breast Cancer) multimodal dataset
Number of modalities: 3 modalities
Classification task: 5-class cancer subtype classification
Data format: CSV files, referencing the MOGONET-main/BRCA dataset
Data path: C:\Users\Tao Xuefeng\Desktop\GAN\MOGONET-main\BRCA

File Structure
edANet/classification/
├── datasets/
│ ├── brca_dataset.py # BRCA dataset class
│ └── ...
├── models/
│ ├── brca_baseline.py # BRCA model definition
│ └── ...
├── src/
│ ├── brca_multi_dataloader.py # BRCA data loader
│ ├── brca_multi_main.py # BRCA main training file
│ └── ...
├── configuration/
│ ├── config_brca_multi.py # BRCA configuration parameters
│ └── ...
├── lib/
│ ├── model_develop_brca.py # BRCA training function
│ └── ...
├── test_brca.py # System integration test
└── simple_test.py # Simple functional test

Main Modifications

Dataset Class (brca_dataset.py)
Inherits from torch.utils.data.Dataset
Supports multimodal CSV data loading
Supports simulation of missing modalities
Automatically handles train/test data splitting

Model Architecture (brca_baseline.py)
Fully connected neural network architecture
Three independent modality encoders
Shared fusion layer
Supports modality dropout
Outputs 5-class classification results

Configuration Parameters (config_brca_multi.py)
Modified classification number to 5
Set BRCA dataset path
Adjusted training parameters

Training Function (model_develop_brca.py)
Adapted to new data format
Supports F1-score calculation
Detailed classification metric reports

Usage

Test system integration
cd ../classification
python simple_test.py

Run full training
cd ../classification
python src/brca_multi_main.py

Run system integration test
cd ../classification
python test_brca.py

Data Format
The BRCA dataset includes the following files:

1_tr.csv, 2_tr.csv, 3_tr.csv: Training data for three modalities
1_te.csv, 2_te.csv, 3_te.csv: Test data for three modalities
labels_tr.csv: Training set labels (0–4)
labels_te.csv: Test set labels (0–4)

Each row of data represents a feature vector for a sample.

