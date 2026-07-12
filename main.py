import torch
from src.data import prepare_dataloaders, reference_energy
from src.model import QChemGNN
from src.train import train_model
from src.visualize import setup_plots_dir, plot_training_curve, plot_parity, plot_residuals, plot_error_vs_size, plot_attention_graph

def evaluate_and_plot(model, test_loader, dataset, atomref, target_mean, target_std, checkpoint_path, device, target_idx=7):
    print("\n--- Running Evaluation & Visualization ---")
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    all_preds, all_targets, all_baseline, all_num_atoms = [], [], [], []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            ref = reference_energy(data, atomref)

            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            pred_norm = out.squeeze()
            pred_ev = pred_norm * target_std + target_mean + ref
            target_ev = data.y[:, target_idx]
            baseline_ev = ref + target_mean

            all_preds.append(pred_ev.cpu())
            all_targets.append(target_ev.cpu())
            all_baseline.append(baseline_ev.cpu())
            all_num_atoms.append(torch.bincount(data.batch.cpu()))

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_baseline = torch.cat(all_baseline)
    all_num_atoms = torch.cat(all_num_atoms)

    errors = all_preds - all_targets
    test_mae = errors.abs().mean().item()
    test_rmse = errors.pow(2).mean().sqrt().item()

    residual_true = all_targets - all_baseline + target_mean
    ss_res = errors.pow(2).sum()
    ss_tot = (residual_true - residual_true.mean()).pow(2).sum()
    test_r2 = (1 - ss_res / ss_tot).item()

    baseline_mae = (all_baseline - all_targets).abs().mean().item()

    print(f"Test MAE:                   {test_mae:.4f} eV")
    print(f"Test RMSE:                  {test_rmse:.4f} eV")
    print(f"Test R^2 (residual scale):  {test_r2:.5f}")
    print(f"Baseline (train-mean) MAE:  {baseline_mae:.4f} eV")

    # Generate Plots
    plots_dir = setup_plots_dir()
    plot_parity(all_targets, all_preds, test_mae, test_r2, plots_dir)
    plot_residuals(errors, plots_dir)
    plot_error_vs_size(all_num_atoms, errors, plots_dir)
    plot_attention_graph(model, dataset, device, plots_dir)
    print(f"All visualizations saved to {plots_dir}/")

def main():
    # 1. Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware initialized. Running on: {device}")
    target_idx = 7 # U0 (Internal Energy)
    epochs = 25

    # 2. Data Pipeline
    train_loader, val_loader, test_loader, dataset, atomref, target_mean, target_std = prepare_dataloaders(target_idx=target_idx)
    
    # 3. Model Initialization
    model = QChemGNN(
        node_in_dim=dataset.num_features,
        edge_in_dim=dataset.num_edge_features,
        hidden_dim=64
    ).to(device)

    # 4. Training
    train_history, val_history, checkpoint_path = train_model(
        model, train_loader, val_loader, atomref, target_mean, target_std, device, target_idx, epochs
    )

    # Plot Training Curve
    plots_dir = setup_plots_dir()
    plot_training_curve(train_history, val_history, epochs, plots_dir)

    # 5. Evaluation and XAI Visualization
    evaluate_and_plot(model, test_loader, dataset, atomref, target_mean, target_std, checkpoint_path, device, target_idx)

if __name__ == "__main__":
    main()