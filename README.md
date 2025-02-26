# Manufacturing Process Anomaly Detection using Graph Convolutional Networks (GCN)

## Overview
This project aims to detect anomalies in a manufacturing process using Graph Convolutional Networks (GCN). The dataset contains machine statuses and worker counts, which are transformed into a graph-based structure for analysis. The model is trained to reconstruct normal operations and identify deviations as anomalies.

## Features
- Data preprocessing: Cleaning, encoding, and handling missing values.
- Graph representation: Converting sequential manufacturing data into a graph structure.
- GCN-based anomaly detection: Identifying abnormal machine behavior.
- Alternative detection methods: Z-Score, Isolation Forest, and One-Class SVM.
- Visualization: Reconstruction error plots for anomaly detection insights.

## Installation
### Prerequisites
Ensure you have Python installed along with the required libraries:
```bash
pip install pandas torch torch-geometric matplotlib scipy scikit-learn
```

## Usage
### 1. Load and Preprocess Data
```python
import pandas as pd
file_path = "/content/manufacturing_test_data.csv"
df = pd.read_csv(file_path)
```
- Encode machine status (`Running` -> 1, `Stopped` -> 0).
- Convert worker counts to numeric format.
- Fill missing values with `0`.

### 2. Split Data
```python
split_idx = int(0.8 * len(df))
train_df = df.iloc[:split_idx] 
test_df = df.iloc[split_idx:]
```

### 3. Convert Data to Graph Structure
```python
from torch_geometric.data import Data
train_graph = Data(x=train_node_features, edge_index=train_edge_index)
test_graph = Data(x=test_node_features, edge_index=test_edge_index)
```

### 4. Train GCN Model
```python
import torch
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x
```

### 5. Detect Anomalies
```python
model.eval()
with torch.no_grad():
    reconstructed_test = model(test_graph.x, test_graph.edge_index)
    errors = torch.mean((reconstructed_test - test_graph.x) ** 2, dim=1)
thresh = errors.mean() + 2 * errors.std()
anomalies = (errors > thresh).nonzero(as_tuple=True)[0]
```

### 6. Visualization
```python
import matplotlib.pyplot as plt
plt.plot(errors.numpy(), label="Reconstruction Error", color="blue")
plt.scatter(anomalies.numpy(), errors[anomalies].numpy(), color='red', label="Anomalies")
plt.axhline(y=thresh.item(), color='r', linestyle='--', label="Threshold")
plt.legend()
plt.show()
```

## Alternative Approaches
- **Z-Score Method**: Uses statistical deviation to detect anomalies.
- **Isolation Forest**: Machine learning-based anomaly detection.
- **One-Class SVM**: Identifies outliers using support vector machines.

## Conclusion
This project demonstrates the power of GCNs for anomaly detection in manufacturing data. By leveraging graph structures, we model dependencies between timestamps and detect irregular machine behavior efficiently.

## License
This project is open-source and available for educational and research purposes.
