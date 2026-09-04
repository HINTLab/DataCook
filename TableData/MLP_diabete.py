# 2025-5-13 Tuesday 
# Author: Sihan(Max) Shang
# Description: This script implements a binary classification MLP for diabetes.csv.


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
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
# Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.long)   # shape: [B]
Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.long)



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
model = TabularNet(input_dim=X.shape[1], output_dim=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_accuracy = 0.0  

for epoch in range(100):
    model.train()
    epoch_loss = 0
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y.squeeze())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)                    # [B, 2]
        preds = logits.argmax(dim=1)                     # [B]
        acc = accuracy_score(Y_test_tensor.numpy(), preds.numpy())

        if acc > best_accuracy:
            best_accuracy = acc
            torch.save(model.state_dict(), "best_model_diabete.pt")
            print(f"[Epoch {epoch}]  New best model saved with accuracy: {best_accuracy:.4f}")


    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {epoch_loss / len(train_loader):.4f}")

from sklearn.metrics import classification_report

model.eval()
with torch.no_grad():
    logits = model(X_test_tensor)
    preds = logits.argmax(dim=1)        
    
print("\nClassification report:")
print(classification_report(Y_test_tensor.numpy(), preds.numpy(), target_names=['No Diabetes', 'Diabetes']))


subset_accuracy = accuracy_score(Y_test_tensor.numpy(), preds.numpy())

print(f"Subset accuracy (exact match): {subset_accuracy:.4f}")