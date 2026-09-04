import os
import copy
import numpy as np
import torch
import mlconfig
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import  transforms
    
# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import util
from util import Transform3D

# MedMNIST
import medmnist
from medmnist import INFO, Evaluator

# Datasets
transform_options = {
    'MedMNIST': {
        "train_transform": [transforms.ToTensor()],
        "test_transform": [transforms.ToTensor()]
    },
    'MedMNIST3D': {
        "train_transform": [Transform3D()],
        "test_transform": [Transform3D()]
    },
}

for name in [
    'PoisonPathMNIST','PoisonDermaMNIST','PoisonOCTMNIST','PoisonPneumoniaMNIST',
    'PoisonRetinaMNIST','PoisonBreastMNIST','PoisonBloodMNIST','PoisonTissueMNIST',
    'PoisonOrganAMNIST','PoisonOrganCMNIST','PoisonOrganSMNIST'
]:
    transform_options[name] = transform_options['MedMNIST']

for name in [
    'PoisonOrganMNIST3D','PoisonNoduleMNIST3D','PoisonFractureMNIST3D',
    'PoisonAdrenalMNIST3D','PoisonVesselMNIST3D','PoisonSynapseMNIST3D'
]:
    transform_options[name] = transform_options['MedMNIST3D']

MED2D_RGB = {'PathMNIST','DermaMNIST','RetinaMNIST','BloodMNIST'}
MED2D_GRAY  = {'OrganAMNIST','OrganCMNIST','OrganSMNIST','OCTMNIST','PneumoniaMNIST','BreastMNIST','TissueMNIST'}
MED3D      = {'OrganMNIST3D','NoduleMNIST3D','FractureMNIST3D','AdrenalMNIST3D','VesselMNIST3D','SynapseMNIST3D'}

IMG_EXTS = {'.jpg','.jpeg','.png','.ppm','.bmp','.tiff','.npz'}
# =========================
# Helper functions
# =========================
def is_image_file(filename: str) -> bool:
    """Check if a file name has a valid image extension."""
    return any(filename.lower().endswith(extension) for extension in IMG_EXTS)

def _is_poison(dtype: str) -> bool:
    """Return True if dataset name starts with 'Poison'."""
    return dtype.startswith('Poison')

def _base(dtype: str) -> str:
    """Return dataset base name without 'Poison' prefix."""
    return dtype.replace('Poison','',1) if _is_poison(dtype) else dtype

def _is_gray2d(dtype: str) -> bool:
    """Return True if dataset is grayscale 2D MedMNIST."""
    return dtype in MED2D_GRAY

def _is_rgb2d(dtype: str) -> bool:
    """Return True if dataset is RGB 2D MedMNIST."""
    return dtype in MED2D_RGB

def _is_3d(dtype: str) -> bool:
    """Return True if dataset is 3D MedMNIST."""
    return dtype in MED3D

def _family(dtype: str) -> str:
    """Return family key used in transform_options."""
    if dtype in MED2D_GRAY or dtype in MED2D_RGB:
        return 'MedMNIST'
    if dtype in MED3D:
        return 'MedMNIST3D'
    if dtype.startswith('Poison'):
        return _family(_base(dtype))
    return dtype

def _compose(p):
    """Ensure the transform pipeline is a Compose object."""
    return p if isinstance(p, transforms.Compose) else transforms.Compose(p)
    
# =========================
# Dataset builders
# =========================
def _build_official(dtype: str, split: str, tfm, high_resolution: bool):
    """
    Build an official MedMNIST dataset.
    Handles RGB/Grayscale and optional high-resolution resizing.
    """
    info = INFO[dtype.lower()]
    DataClass = getattr(medmnist, info['python_class'])

    if _is_gray2d(dtype):
        as_rgb = True
    elif _is_rgb2d(dtype):
        as_rgb = False
    elif _is_3d(dtype):
        as_rgb = False
    else:
        as_rgb = True  # fallback

    if high_resolution and (_is_gray2d(dtype) or _is_rgb2d(dtype)):
        return DataClass(split=split, transform=tfm, download=True, as_rgb=as_rgb, size=224)
    else:
        return DataClass(split=split, transform=tfm, download=True, as_rgb=as_rgb)

def _build_poison(base_dtype: str, path: str, split: str, tfm, high_resolution: bool):
    """
    Build a Poison dataset.
    Prefers a custom class named Poison{BaseType}, 
    falls back to folder/npz loaders if not found.
    """
    if _is_3d(base_dtype) or high_resolution:
        return Data3DFolderWithLabel(path, None, transform=tfm)
    else:
        return DataFolderWithLabel(path, None, transform=tfm)

def _make_dataset(dtype: str, split: str, path: str, tfm, high_resolution: bool):
    """Factory method to create either official or Poison datasets."""
    if _is_poison(dtype):
        return _build_poison(_base(dtype), path, split, tfm, high_resolution)
    base = _base(dtype)
    if _is_gray2d(base) or _is_rgb2d(base) or _is_3d(base):
        return _build_official(base, split, tfm, high_resolution)
    raise ValueError(f"Dataset type {dtype} not implemented.")

# =========================
# Main dataset generator
# =========================
@mlconfig.register
class DatasetGenerator():
    def __init__(self, 
                 train_batch_size=128, eval_batch_size=128, num_of_workers=4, 
                 train_data_path='../datasets/', train_data_type='PathMNIST', seed=0,
                 test_data_path='../datasets/', test_data_type='PathMNIST', 
                 recover_test_data_path='../datasets/', recover_test_data_type='PathMNIST',
                 no_train_augments=False, high_resolution=False):
        np.random.seed(seed)

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_of_workers = num_of_workers
        self.seed = seed

        self.train_data_type = train_data_type
        self.test_data_type = test_data_type
        self.recover_test_data_type = recover_test_data_type

        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.recover_test_data_path = recover_test_data_path

        # --- Build transforms ---
        train_key = _family(train_data_type)
        test_key  = _family(test_data_type)

        try:
            train_transform = _compose(transform_options[train_key]['train_transform'])
            test_transform  = _compose(transform_options[test_key]['test_transform'])
        except KeyError as e:
            raise ValueError(f"Missing transform config: {e}")
        
        recover_test_transform = test_transform
        if no_train_augments:
            train_transform = test_transform

        # --- Build datasets ---
        self.datasets = {
            'train_dataset': _make_dataset(train_data_type, 'train', self.train_data_path, train_transform, high_resolution),
            'test_dataset': _make_dataset(test_data_type, 'test', self.test_data_path, test_transform, high_resolution),
            'recover_test_dataset': _make_dataset(recover_test_data_type, 'test', self.recover_test_data_path, recover_test_transform, high_resolution),
        }

    def getDataLoader(self, train_shuffle=True, train_drop_last=True):
        data_loaders = {}

        data_loaders['train_dataset'] = DataLoader(dataset=self.datasets['train_dataset'],
                                                   batch_size=self.train_batch_size,
                                                   shuffle=train_shuffle, pin_memory=True,
                                                   drop_last=train_drop_last, num_workers=self.num_of_workers)

        data_loaders['test_dataset'] = DataLoader(dataset=self.datasets['test_dataset'],
                                                  batch_size=self.eval_batch_size,
                                                  shuffle=False, pin_memory=True,
                                                  drop_last=False, num_workers=self.num_of_workers)

        data_loaders['recover_test_dataset'] = DataLoader(dataset=self.datasets['recover_test_dataset'],
                                                  batch_size=self.eval_batch_size,
                                                  shuffle=False, pin_memory=True,
                                                  drop_last=False, num_workers=self.num_of_workers)

        return data_loaders

# =========================
# Custom dataset classes
# =========================
class DataFolderWithLabel(Dataset):
    """
    Generic 2D folder dataset: expects structure root/class_x/img.png
    Assumes class folder names are integer labels.
    """
    def __init__(self, root, pred_idx=None, transform=None):
        self.labels = []
        self.images = []
        self.transform = transform

        for class_name in sorted(os.listdir(root)):
            label = int(class_name)
            for file_name in sorted(os.listdir(os.path.join(root, class_name))):
                if not is_image_file(file_name):
                    continue
                self.images.append(os.path.join(root, class_name, file_name))
                self.labels.append(label)

        self.pred_idx = self.labels if pred_idx is None else pred_idx

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, self.pred_idx[idx]

class Data3DFolderWithLabel(Dataset):
    """
    3D or high-resolution dataset loader (current version: loads from a single .npz file).
    The .npz file must contain:
        - perturb_images: shape (N, D, H, W) or (N, C, D, H, W)
        - perturb_labels: shape (N,)
    """
    def __init__(self, npz_file, pred_idx=None, transform=None):
        data = np.load(npz_file)
        self.images = data['perturb_images']
        self.labels = data['perturb_labels']
        self.transform = transform
        self.pred_idx = self.labels if pred_idx is None else pred_idx

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, self.pred_idx[idx]

