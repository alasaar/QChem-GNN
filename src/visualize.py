import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def setup_plots_dir(dir_name='./plots'):
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def plot_training_curve(train_loss, val_loss, epochs, save_dir):
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_loss, label='Train MAE', marker='o', markersize=3)
    plt.plot(epochs_range, val_loss, label='Val MAE', marker='o', markersize=3)
    plt.axhline(0.043, color='gray', linestyle='--', linewidth=1, label='Chemical accuracy (0.043 eV)')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (eV)')
    plt.title('QChem-GNN Training Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_parity(all_targets, all_preds, test_mae, test_r2, save_dir):
    plt.figure(figsize=(6, 6))
    hb = plt.hexbin(all_targets.numpy(), all_preds.numpy(), gridsize=60, cmap='viridis', mincnt=1)
    lims = [min(all_targets.min(), all_preds.min()).item(), max(all_targets.max(), all_preds.max()).item()]
    plt.plot(lims, lims, 'r--', linewidth=1, label='Perfect prediction')
    plt.xlabel('True U0 (eV)')
    plt.ylabel('Predicted U0 (eV)')
    plt.title(f'Parity Plot -- Test Set (MAE={test_mae:.3f} eV, R2={test_r2:.4f})')
    plt.colorbar(hb, label='Point density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'parity_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_residuals(errors, save_dir):
    plt.figure(figsize=(7, 5))
    plt.hist(errors.numpy(), bins=80, color='steelblue', edgecolor='none')
    plt.axvline(0, color='black', linewidth=1)
    plt.xlabel('Prediction Error (Predicted - True, eV)')
    plt.ylabel('Count')
    plt.title('Test Set Residual Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'residual_histogram.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_error_vs_size(all_num_atoms, errors, save_dir):
    sizes = all_num_atoms.numpy()
    abs_err = errors.abs().numpy()

    unique_sizes = np.unique(sizes)
    mae_by_size = [abs_err[sizes == s].mean() for s in unique_sizes]
    count_by_size = [np.sum(sizes == s) for s in unique_sizes]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(unique_sizes, mae_by_size, color='indianred', alpha=0.8)
    ax1.set_xlabel('Number of Atoms in Molecule')
    ax1.set_ylabel('MAE (eV)', color='indianred')
    ax1.set_title('Error vs. Molecule Size')

    ax2 = ax1.twinx()
    ax2.plot(unique_sizes, count_by_size, color='steelblue', marker='o', markersize=3)
    ax2.set_ylabel('Number of Test Molecules', color='steelblue')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mae_by_molecule_size.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_attention_graph(model, dataset, device, save_dir):
    # Find a nicely sized molecule for visualization
    sample_idx = next(i for i in range(len(dataset)) if dataset[i].num_nodes >= 15)
    sample = dataset[sample_idx].to(device)
    sample.batch = torch.zeros(sample.num_nodes, dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        out, (ei1, alpha1), (ei2, alpha2) = model(
            sample.x, sample.edge_index, sample.edge_attr, sample.batch, return_attention=True
        )

    alpha = alpha2.mean(dim=-1).cpu().numpy()
    edge_index_np = ei2.cpu().numpy()

    atom_types = ['H', 'C', 'N', 'O', 'F']
    type_idx = sample.x[:, :5].argmax(dim=1).cpu().numpy()
    symbols = [atom_types[t] for t in type_idx]
    colors = {'H': '#dddddd', 'C': '#333333', 'N': '#3050f8', 'O': '#ff0d0d', 'F': '#90e050'}
    node_colors = [colors[s] for s in symbols]

    pos_2d = sample.pos[:, :2].cpu().numpy()

    plt.figure(figsize=(7, 7))
    alpha_norm = (alpha - alpha.min()) / (alpha.max() - alpha.min() + 1e-8)
    for k in range(edge_index_np.shape[1]):
        src, dst = edge_index_np[:, k]
        plt.plot([pos_2d[src, 0], pos_2d[dst, 0]], [pos_2d[src, 1], pos_2d[dst, 1]],
                 color='orange', linewidth=1 + 4 * alpha_norm[k], alpha=0.6, zorder=1)

    plt.scatter(pos_2d[:, 0], pos_2d[:, 1], c=node_colors, s=300, edgecolors='black', zorder=2)
    for i, sym in enumerate(symbols):
        plt.text(pos_2d[i, 0], pos_2d[i, 1], sym, ha='center', va='center', fontsize=9,
                 color='white' if sym == 'C' else 'black', zorder=3)

    plt.title(f'QChem-GNN Layer-2 Attention Weights\n(molecule {sample_idx}, {sample.num_nodes} atoms, edge thickness = attention)')
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'attention_visualization.png'), dpi=150, bbox_inches='tight')
    plt.close()