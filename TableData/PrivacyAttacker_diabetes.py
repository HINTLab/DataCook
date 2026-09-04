# 2025-5-13 Sunday 
# Author: Sihan(Max) Shang
# Description: This script attacks the mlp model of diabetes.csv.
#  diabetes.csv 

import os 
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tqdm
import sys
from typing import Tuple, Dict
import copy

def fix_everything(seed=42):
    """Make results reproducible (CPU/CUDA, cuDNN)."""
    import random, os, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
fix_everything()


class PrivacyAttacker:
    """
    Adversarial/Anti-adversarial attacker for tabular data.

    - adversarial: pushes predictions toward (wrong) target
    - anti-adversarial: pushes predictions away (invert direction)
    """
    def __init__(self, device, model: nn.Module, x_min, x_max, epsilon: float = 5, alpha: float = 0.05, num_steps: int = 300, invert_direction: bool = False):

        self.model = model
        self.model.eval() 
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        self.x_min = x_min  # shape: [num_features]
        self.x_max = x_max
        
        self.attackable_indices = [5, 6]  # e.g., [3, 4, 6]
        self.invert_direction = invert_direction

    def generate_adversarial_data(self, data: torch.Tensor, pseudo_labels: torch.Tensor) -> torch.Tensor:
        """
        Generate adversarial (or anti-adversarial) examples against given target labels.
        The 'invert_direction' flag controls whether gradients are negated.
        """
        x_orig = data.clone().detach()
        x_adv = x_orig.clone().detach().requires_grad_(True)

        mask = torch.zeros_like(x_orig)
        if self.attackable_indices is not None:
            mask[:, self.attackable_indices] = 1.0   

        for step in range(self.num_steps):
            self.model.zero_grad()
            outputs = self.model(x_adv)
            loss = self.criterion(outputs, pseudo_labels).mean()
            loss.backward()

            with torch.no_grad():
                grad = x_adv.grad.sign()

                # Direction: -1 for anti-adv/datacook (anti-adversarial), +1 for adv/datacook-adv
                direction = -1.0 if self.invert_direction else 1.0
                perturbed = self.alpha * grad * mask * direction
                x_adv = x_adv + perturbed
                
                # Clamp to epsilon L_inf ball around original input
                delta = torch.clamp(x_adv - x_orig, -self.epsilon, self.epsilon)
                x_adv = x_orig + delta

                # Clamp to original feature-wise valid range
                x_adv = torch.max(torch.min(x_adv, self.x_max), self.x_min)

            x_adv = x_adv.detach().requires_grad_(True)

        return x_adv.detach()

class PrivacyProtector:
    """
    Protection pipeline:
    1) Generate pseudo labels from the original model (optional).
    2) Create adversarial/anti-adversarial datasets.
    3) Train a protected model on adversarial data.
    """
    def __init__(self, original_model: nn.Module,raw_model: nn.Module, device: torch.device):
        self.original_model = original_model
        self.raw_model = raw_model
        self.device = device
        self.protected_model = None
        
    def generate_pseudo_labels(self, data_loader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Produce pseudo labels by running the original model and thresholding probabilities.
        Returns concatenated tensors: (data, pseudo_labels, ground_truth_labels)
        """
        self.original_model.eval()
        all_data = []
        all_labels = []
        all_pseudo_labels = []

        with torch.no_grad():
            for data, label in data_loader:
                outputs = self.original_model(data)           # shape [B, 2]
                pseudo_labels = outputs.argmax(dim=1)         # shape [B]
                all_data.append(data.cpu())
                all_labels.append(label.cpu())
                all_pseudo_labels.append(pseudo_labels.cpu())

        return torch.cat(all_data), torch.cat(all_pseudo_labels), torch.cat(all_labels)

    
    def create_adversarial_dataset(self, data_loader: torch.utils.data.DataLoader, attacker: PrivacyAttacker, use_pseudo_labels: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create (adversarial) dataset given an attacker and whether to use pseudo labels or original labels.
        Returns adv_data, pseudo_labels, original_labels (for later evaluation).
        """
        original_data, pseudo_labels, original_label = self.generate_pseudo_labels(data_loader)
        # Choose supervision for the attacker:
        # - use_pseudo_labels=True: datacook / datacook-adv
        # - use_pseudo_labels=False: adv / anti-adv
        supervise_labels = pseudo_labels if use_pseudo_labels else  original_label.squeeze()
        adv_data = attacker.generate_adversarial_data(original_data, supervise_labels)

        return adv_data, pseudo_labels, original_label
    
    def train_protected_model(self, 
                            train_loader: torch.utils.data.DataLoader,
                            num_epochs: int = 100,
                            learning_rate: float = 0.001) -> nn.Module:
        """
        Train a protected model on adversarial (or anti-adversarial) data.
        """
        self.protected_model = copy.deepcopy(self.raw_model)
        optimizer = optim.Adam(self.protected_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            self.protected_model.train()
            running_loss = 0.0
            
            for adv_data, pseudo_labels in train_loader:
            
                optimizer.zero_grad()
                outputs = self.protected_model(adv_data)
                pseudo_labels = pseudo_labels.squeeze()  # shape: [B]
                loss = criterion(outputs, pseudo_labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
        
        return self.protected_model

# Define the MLP model  
class TabularNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TabularNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def report_feature_changes(original_data: torch.Tensor, adv_data: torch.Tensor, scaler: StandardScaler, feature_names: list):
    """
    Report per-feature min/max range before vs. after attack (inverse-transformed to original scale).
    """
    original_np = scaler.inverse_transform(original_data.numpy())
    adv_np = scaler.inverse_transform(adv_data.numpy())
    
    report = []
    for i, name in enumerate(feature_names):
        orig_min = original_np[:, i].min()
        orig_max = original_np[:, i].max()
        adv_min = adv_np[:, i].min()
        adv_max = adv_np[:, i].max()
        delta = np.max(np.abs(adv_np[:, i] - original_np[:, i]))

        report.append({
            "Feature": name,
            "Orig Min": round(orig_min, 4),
            "Orig Max": round(orig_max, 4),
            "Adv Min": round(adv_min, 4),
            "Adv Max": round(adv_max, 4),
            "Max Change": round(delta, 4)
        })

    df = pd.DataFrame(report)
    print("\n Feature Range Changes (original scale):")
    print(df.to_string(index=False))


def evaluate_models(original_model: nn.Module,
                   protected_model: nn.Module,
                   clean_test_loader: torch.utils.data.DataLoader,
                   adv_test_loader: torch.utils.data.DataLoader,
                   device: torch.device) -> Dict[str, float]:
    """
    Evaluate models on clean and adversarial test sets (subset accuracy for multi-label).
    """
    def calculate_accuracy(model, data_loader):
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, labels in data_loader:

                logits = model(data)  # [B, 2]
                predicted = torch.argmax(logits, dim=1)  # [B]
                
                all_preds.append(predicted.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        return accuracy_score(all_labels, all_preds)

    
    results = {
        'original_clean_acc': calculate_accuracy(original_model, clean_test_loader),
        'protected_clean_acc': calculate_accuracy(protected_model, clean_test_loader),
        'protected_adv_acc': calculate_accuracy(protected_model, adv_test_loader)
    }
    
    return results

def parse_args():
    parser = argparse.ArgumentParser(description="Tabular protection")
    parser.add_argument(
        "--attack_type",
        type=str,
        default="datacook",
        choices=["adv", "anti-adv", "datacook-adv", "datacook"],
        help="Choose attack mode: adv/anti-adv vs datacook-adv/datacook (pseudo labels) and gradient direction."
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs for protected model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    return parser.parse_args()
    
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # CUDA Options

    torch.cuda.manual_seed(42)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]

    data = pd.read_csv('dataset/diabetes.csv')

    # Feature selection
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    labels = ['Outcome']

    # Fill missing values with median
    X = data[features].fillna(data[features].median())
    Y = data[labels].astype(int)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.long)   # shape: [B]
    Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.long)

    BATCH_SIZE = args.batch_size
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    x_min = X_tensor.min(dim=0)[0]
    x_max = X_tensor.max(dim=0)[0]

    model = TabularNet(input_dim=X.shape[1], output_dim=2)
    original_model = TabularNet(input_dim=X.shape[1], output_dim=2)
    model.load_state_dict(torch.load("best_model_diabete.pt"))
    best_model = model
    
    # Attack mode switches
    # pseudo labels: datacook/datacook-adv use pseudo; adv/anti-adv use original labels
    use_pseudo_labels = args.attack_type in ["datacook-adv", "datacook"]
    # invert gradient: anti-adv/datacook use -1; adv/datacook-adv use +1
    invert_direction = args.attack_type in ["anti-adv", "datacook"]

    protector = PrivacyProtector(best_model, original_model, device)
    attacker = PrivacyAttacker(device, best_model, x_min, x_max, invert_direction=invert_direction)

    adv_train_data, train_pseudo_labels, orig_train_labels = protector.create_adversarial_dataset(train_loader, attacker, use_pseudo_labels=use_pseudo_labels)
    adv_test_data, test_pseudo_labels, orig_test_labels = protector.create_adversarial_dataset(test_loader, attacker, use_pseudo_labels=use_pseudo_labels)

    adv_train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(adv_train_data, orig_train_labels),
        batch_size=32, shuffle=False
    )
    
    adv_test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(adv_test_data, orig_test_labels),
        batch_size=32, shuffle=False
    )
    
    protected_model = protector.train_protected_model(adv_train_loader)
    
    results = evaluate_models(
        best_model,
        protected_model,
        test_loader,      # clean
        adv_test_loader,  # adversarial
        device
    )
    print("\nEvaluation Results:")
    print(f"Original model on clean test:    {results['original_clean_acc']:.4f}")
    print(f"Protected model on clean test:   {results['protected_clean_acc']:.4f}")
    print(f"Protected model on adversarial:  {results['protected_adv_acc']:.4f}")

    report_feature_changes(
        original_data=X_train_tensor,       
        adv_data=adv_train_data,            
        scaler=scaler,                     
        feature_names=features            
    )
    
if __name__ == '__main__':
    main()
