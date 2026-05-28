import os
import csv
import urllib.request
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def download_pima_dataset(dest_path="data/pima-indians-diabetes.csv"):
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(dest_path):
        print(f"[SILOSPAN MODEL] Downloading Pima Indians Diabetes dataset to {dest_path}...")
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        urllib.request.urlretrieve(url, dest_path)

def load_tabular_data(
    file_path: str,
    target_col: int = -1,
    partition_id: int = 0,
    num_partitions: int = 2,
    batch_size: int = 32,
    impute_cols: list = None,
    scale: bool = True
):
    """
    Generic dataset loader for any tabular CSV. Handles:
    - Loading features and labels from a specified CSV file.
    - Median imputation of missing values for specified columns.
    - Feature scaling to range [0, 1].
    - Deterministic 80/20 train/validation split.
    - Partitioning train data among federated silos (Non-IID sort by last feature column).
    """
    features = []
    labels = []
    
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            row_floats = [float(x) for x in row]
            
            # Extract target and features based on index position
            if target_col == -1 or target_col == len(row_floats) - 1:
                features.append(row_floats[:-1])
                labels.append(int(row_floats[-1]))
            else:
                feat = row_floats.copy()
                lbl = int(feat.pop(target_col))
                features.append(feat)
                labels.append(lbl)
                
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    
    # Impute missing values (0s) with column medians
    if impute_cols:
        for col in impute_cols:
            col_vals = features[:, col]
            non_zero = col_vals[col_vals > 0]
            median_val = np.median(non_zero) if len(non_zero) > 0 else 0.0
            features[col_vals == 0, col] = median_val
            
    # Scale features to range [0, 1]
    if scale:
        min_vals = np.min(features, axis=0)
        max_vals = np.max(features, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        features = (features - min_vals) / range_vals
        
    # Split train and validation
    num_samples = len(features)
    indices = np.arange(num_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    val_size = int(0.2 * num_samples)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    
    # Create age/feature sorted Non-IID partitions (sort by the last feature index)
    sort_feature_idx = train_features.shape[1] - 1
    sorted_indices = np.argsort(train_features[:, sort_feature_idx])
    
    partition_size = len(sorted_indices) // num_partitions
    start_idx = partition_id * partition_size
    end_idx = (partition_id + 1) * partition_size if partition_id < num_partitions - 1 else len(sorted_indices)
    
    local_indices = sorted_indices[start_idx:end_idx]
    local_features = train_features[local_indices]
    local_labels = train_labels[local_indices]
    
    # Build datasets
    train_dataset = TensorDataset(torch.tensor(local_features), torch.tensor(local_labels))
    val_features = features[val_indices]
    val_labels = labels[val_indices]
    val_dataset = TensorDataset(torch.tensor(val_features), torch.tensor(val_labels))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def load_diabetes_data(partition_id: int, num_partitions: int, batch_size: int = 32):
    """
    Backward compatibility wrapper to load Pima Indians dataset.
    """
    download_pima_dataset()
    return load_tabular_data(
        file_path="data/pima-indians-diabetes.csv",
        target_col=-1,
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
        impute_cols=[1, 2, 3, 4, 5],
        scale=True
    )
