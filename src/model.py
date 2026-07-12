import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class QChemGNN(torch.nn.Module):
    """
    Quantum Chemistry Graph Neural Network (QChem-GNN)
    Graph Attention Network for predicting molecular thermodynamic properties.
    """
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim):
        super().__init__()
        self.node_embed = torch.nn.Linear(node_in_dim, hidden_dim)

        self.conv1 = GATConv(hidden_dim, hidden_dim, edge_dim=edge_in_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim, edge_dim=edge_in_dim)

        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch, return_attention=False):
        x = self.node_embed(x)
        
        if return_attention:
            x, (ei1, alpha1) = self.conv1(x, edge_index, edge_attr, return_attention_weights=True)
            x = F.relu(x)
            x, (ei2, alpha2) = self.conv2(x, edge_index, edge_attr, return_attention_weights=True)
            x = F.relu(x)
        else:
            x = F.relu(self.conv1(x, edge_index, edge_attr))
            x = F.relu(self.conv2(x, edge_index, edge_attr))
            
        pooled = global_mean_pool(x, batch)
        out = self.predictor(pooled)
        
        if return_attention:
            return out, (ei1, alpha1), (ei2, alpha2)
        return out