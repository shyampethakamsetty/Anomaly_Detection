
#import data from csv file

import pandas as pd

file_path = "/content/manufacturing_test_data.csv"
df = pd.read_csv(file_path)

print(df)

#========================================================================================================

#Data Preprocessing

df['M1_Status'] = df['M1_Status'].map({'Running': 1, 'Stopped': 0})
df['M2_Status'] = df['M2_Status'].map({'Running': 1, 'Stopped': 0})
df['M3_Status'] = df['M3_Status'].map({'Running': 1, 'Stopped': 0})


worker_cols = ['M1_Worker_Count', 'M2_Worker_Count', 'M3_Worker_Count']
df[worker_cols] = df[worker_cols].apply(pd.to_numeric, errors='coerce')

print(df)

#========================================================================================================

#filling missing values

df.fillna(0, inplace=True)

#========================================================================================================

#splitting data into training and testing data


split_idx = int(0.8 * len(df))
train_df = df.iloc[:split_idx] 
test_df = df.iloc[split_idx:] 

#========================================================================================================

#converting data into tensor

import torch
from torch_geometric.data import Data


train_node_features = torch.tensor(train_df[['M1_Status', 'M2_Status', 'M3_Status',
                                             'M1_Worker_Count', 'M2_Worker_Count', 'M3_Worker_Count']].values,
                                   dtype=torch.float)


test_node_features = torch.tensor(test_df[['M1_Status', 'M2_Status', 'M3_Status',
                                           'M1_Worker_Count', 'M2_Worker_Count', 'M3_Worker_Count']].values,
                                 dtype=torch.float)


#========================================================================================================

#creating edge index

train_edge_index = torch.tensor([[i, i + 1] for i in range(len(train_df) - 1)], dtype=torch.long).t().contiguous()
test_edge_index = torch.tensor([[i, i + 1] for i in range(len(test_df) - 1)], dtype=torch.long).t().contiguous()

#========================================================================================================

#creating graph data

train_graph = Data(x=train_node_features, edge_index=train_edge_index)
test_graph = Data(x=test_node_features, edge_index=test_edge_index)

print(train_graph)
print(test_graph) 

#========================================================================================================

#creating model

import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


#========================================================================================================

#training model

input_dim = 6 
hidden_dim = 16  
output_dim = 6    
model = GCN(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()


epochs = 50  
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(train_graph.x, train_graph.edge_index)
    loss = loss_fn(output, train_graph.x)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


#========================================================================================================

# 🔹 Testing (Anomaly Detection)
#Gaussian distribution assumptions.

model.eval()
with torch.no_grad():
    reconstructed_test = model(test_graph.x, test_graph.edge_index)
    errors = torch.mean((reconstructed_test - test_graph.x) ** 2, dim=1)

threshold = errors.mean() + 2 * errors.std()
anomalies = (errors > threshold).nonzero(as_tuple=True)[0]

print("Detected Anomalous Timestamps:", anomalies.tolist())

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.plot(errors.numpy(), label="Reconstruction Error", color="blue")
plt.scatter(anomalies.numpy(), errors[anomalies].numpy(), color='red', label="Anomalies", zorder=3)
plt.axhline(y=threshold.item(), color='r', linestyle='--', label="Threshold")
plt.xlabel("Timestamp")
plt.ylabel("Reconstruction Error")
plt.title("Anomaly Detection using GAE")
plt.legend()
plt.show()


#========================================================================================================

'''Alternative Approachs:

import matplotlib.pyplot as plt
import scipy.stats as stats

model.eval()
with torch.no_grad():
    reconstructed_test = model(test_graph.x, test_graph.edge_index)
    errors = torch.mean((reconstructed_test - test_graph.x) ** 2, dim=1)

# 🔹 Compute Z-Scores
z_scores = stats.zscore(errors.numpy())

# 🔹 Set a threshold (e.g., Z-score > 2.5)
anomalies = (abs(z_scores) > 2.5).nonzero()[0]

# 🔹 Visualization
plt.figure(figsize=(12, 5))
plt.plot(errors.numpy(), label="Reconstruction Error", color="blue")
plt.scatter(anomalies, errors[anomalies].numpy(), color='red', label="Anomalies", zorder=3)
plt.axhline(y=errors.mean().item(), color='g', linestyle='--', label="Mean Error")
plt.xlabel("Timestamp")
plt.ylabel("Reconstruction Error")
plt.title("Anomaly Detection using Z-Score")
plt.legend()
plt.show()






from sklearn.ensemble import IsolationForest

model.eval()
with torch.no_grad():
    reconstructed_test = model(test_graph.x, test_graph.edge_index)
    errors = torch.mean((reconstructed_test - test_graph.x) ** 2, dim=1).numpy().reshape(-1, 1)

# 🔹 Train an Isolation Forest model
iso_forest = IsolationForest(contamination=0.05)  # 5% anomalies
iso_forest.fit(errors)

# 🔹 Predict anomalies (-1 = anomaly, 1 = normal)
predictions = iso_forest.predict(errors)
anomalies = (predictions == -1).nonzero()[0]

# 🔹 Visualization
plt.figure(figsize=(12, 5))
plt.plot(errors, label="Reconstruction Error", color="blue")
plt.scatter(anomalies, errors[anomalies], color='red', label="Anomalies", zorder=3)
plt.axhline(y=errors.mean(), color='g', linestyle='--', label="Mean Error")
plt.xlabel("Timestamp")
plt.ylabel("Reconstruction Error")
plt.title("Anomaly Detection using Isolation Forest")
plt.legend()
plt.show()






from sklearn.svm import OneClassSVM

model.eval()
with torch.no_grad():
    reconstructed_test = model(test_graph.x, test_graph.edge_index)
    errors = torch.mean((reconstructed_test - test_graph.x) ** 2, dim=1).numpy().reshape(-1, 1)

# 🔹 Train One-Class SVM
svm = OneClassSVM(nu=0.05, kernel="rbf")  # 5% anomalies
svm.fit(errors)

# 🔹 Predict anomalies (-1 = anomaly, 1 = normal)
predictions = svm.predict(errors)
anomalies = (predictions == -1).nonzero()[0]

# 🔹 Visualization
plt.figure(figsize=(12, 5))
plt.plot(errors, label="Reconstruction Error", color="blue")
plt.scatter(anomalies, errors[anomalies], color='red', label="Anomalies", zorder=3)
plt.axhline(y=errors.mean(), color='g', linestyle='--', label="Mean Error")
plt.xlabel("Timestamp")
plt.ylabel("Reconstruction Error")
plt.title("Anomaly Detection using One-Class SVM")
plt.legend()
plt.show()

'''
