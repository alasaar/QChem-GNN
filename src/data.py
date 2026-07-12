import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform
from torch_geometric.nn import global_add_pool

class AddRBFDistance(BaseTransform):
    """
    Calculates Euclidean distance between atoms and expands it into 
    Radial Basis Function (RBF) bins for spatial embeddings.
    """
    def __init__(self, num_rbf=16, cutoff=2.5):
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.centers = torch.linspace(0, cutoff, num_rbf)
        self.width = cutoff / num_rbf

    def forward(self, data):
        row, col = data.edge_index
        dist = (data.pos[row] - data.pos[col]).norm(dim=-1, keepdim=True)
        rbf = torch.exp(-((dist - self.centers) ** 2) / (2 * self.width ** 2))
        data.edge_attr = torch.cat([data.edge_attr, rbf], dim=-1)
        return data

def reference_energy(data, ref_tensor):
    """
    Calculates the baseline thermodynamic energy of isolated atoms in the graph.
    """
    target_device = data.batch.device
    if ref_tensor is None:
        return torch.zeros(data.num_graphs, device=target_device)
    z_cpu = data.z.cpu()
    batch_cpu = data.batch.cpu()
    ref_cpu = ref_tensor[z_cpu].squeeze(-1)
    ref_per_graph = global_add_pool(ref_cpu, batch_cpu)
    return ref_per_graph.to(target_device)

def prepare_dataloaders(data_dir='./data/QM9', batch_size=128, target_idx=7):
    """
    Downloads QM9, applies RBF transforms, calculates residual stats, and splits the data.
    """
    print("Fetching and processing Quantum Chemistry Dataset (QM9)...")
    dataset = QM9(root=data_dir)
    dataset.transform = AddRBFDistance(num_rbf=16, cutoff=2.5)

    torch.manual_seed(42)
    perm = torch.randperm(len(dataset))
    dataset = dataset[perm]

    n_train = int(0.8 * len(dataset))
    n_val = int(0.1 * len(dataset))

    train_dataset = dataset[:n_train]
    val_dataset = dataset[n_train:n_train + n_val]
    test_dataset = dataset[n_train + n_val:]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Calculate Residual Statistics (Mean and Std) for normalization
    atomref = dataset.atomref(target_idx)
    residuals = []
    
    # Fast pass to calculate target standardization stats
    temp_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
    for data in temp_loader:
        ref = reference_energy(data, atomref)
        residuals.append(data.y[:, target_idx] - ref)
    
    residuals = torch.cat(residuals)
    target_mean = residuals.mean().item()
    target_std = residuals.std().item()

    return train_loader, val_loader, test_loader, dataset, atomref, target_mean, target_std