# 2025-5-13 Tuesday 
# Author: Sihan(Max) Shang
# Description: This script implements a multi-output classifier using mlp for systemic diseases.csv.
# systemic diseases.csv 

# Import necessary libraries
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('systemic diseases.csv')

# Feature selection
# Cholesterol - 总胆固醇水平; HDL - 高密度脂蛋白胆固醇; Hyperlipidemia - 高脂血症; BMI - 体重指数; DBP - 舒张压; SBP - 收缩压; Smoking - 吸烟; Hypertension - 高血压; Diabetes - 糖尿病; ASCVD - 动脉粥样硬化性心血管疾病; MI - 心肌梗死; Stroke - 中风; dem - 是否发生认知障碍/痴呆; PD - 帕金森病
features = ['Age', 'Gender', 'Ethnic', 'Cholesterol', 'HDL', 'Hyperlipidemia', 'BMI', 'DBP', 'SBP', 'Smoking', 'Hypertension', 'Diabetes', 'Smoking', 'ASCVD', 'MI', 'Stroke', 'dem', 'PD']
# inciASCVD - 是否在随访期间新发ASCVD事件; inciMI - 是否在随访期间新发心肌梗死; inciStroke - 中风; incidem - 是否在随访期间新发痴呆; inciPD - 帕金森病
labels = ['inciASCVD', 'inciMI', 'inciStroke', 'incidem', 'inciPD']

# Convert categorical variables to numerical
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
data['Ethnic'] = data['Ethnic'].map({'Non-Whites': 0, 'Whites': 1})
data['Hyperlipidemia'] = data['Hyperlipidemia'].map({'yes': 0, 'no': 1})
data['Hypertension'] = data['Hypertension'].map({'yes': 0, 'no': 1})
data['Diabetes'] = data['Diabetes'].map({'yes': 0, 'no': 1})
data['Smoking'] = data['Smoking'].map({'yes': 0, 'no': 1})

# Fill missing values with median
X = data[features].fillna(data[features].median())
Y = data[labels].astype(int)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float32)

BATCH_SIZE = 32  
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

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

# Initialize the model, loss function and optimizer   
model = TabularNet(input_dim=X.shape[1], output_dim=5)
criterion = nn.BCEWithLogitsLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_accuracy = 0.0  

for epoch in range(100):
    model.train()
    epoch_loss = 0
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()


    model.eval()
    with torch.no_grad():
        y_pred = torch.sigmoid(model(X_test_tensor)).numpy()
        y_pred_bin = (y_pred > 0.5).astype(int)
        subset_accuracy = accuracy_score(Y_test.values, y_pred_bin)

        if subset_accuracy > best_accuracy:
            best_accuracy = subset_accuracy
            torch.save(model.state_dict(), "best_model.pt")  
            print(f"[Epoch {epoch}]  New best model saved with subset accuracy: {best_accuracy:.4f}")

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {epoch_loss / len(train_loader):.4f}")

from sklearn.metrics import classification_report

model.eval()
with torch.no_grad():
    y_pred = torch.sigmoid(model(X_test_tensor)).numpy()
    y_pred_bin = (y_pred > 0.5).astype(int)
    print(classification_report(Y_test, y_pred_bin, target_names=Y.columns))

# Calculate accuracy
accuracy = np.mean(y_pred_bin == Y_test.values)
print("\nAccuracy:", accuracy)

subset_accuracy = accuracy_score(Y_test.values, y_pred_bin)
print(f"Subset accuracy (exact match): {subset_accuracy:.4f}")