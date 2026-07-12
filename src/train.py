import torch
import os
from tqdm.auto import tqdm
from src.data import reference_energy

def train_model(model, train_loader, val_loader, atomref, target_mean, target_std, device, target_idx=7, epochs=25, save_dir='./checkpoints'):
    """
    Production-grade training loop with learning rate scheduling and early stopping checkpoints.
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, 'qchem_gnn_best.pth')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.L1Loss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=epochs
    )

    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history_ev = []

    print("\n--- Initializing QChem-GNN Training Sequence ---")

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
        
        for data in train_pbar:
            data = data.to(device)
            ref = reference_energy(data, atomref)
            target = ((data.y[:, target_idx] - ref) - target_mean) / target_std

            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out.squeeze(), target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item() * data.num_graphs
            train_pbar.set_postfix({'MAE (norm)': loss.item()})

        avg_train_loss_norm = total_train_loss / len(train_loader.dataset)
        avg_train_loss_ev = avg_train_loss_norm * target_std

        # Validation Phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                ref = reference_energy(data, atomref)
                target = ((data.y[:, target_idx] - ref) - target_mean) / target_std
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                loss = criterion(out.squeeze(), target)
                total_val_loss += loss.item() * data.num_graphs

        avg_val_loss_norm = total_val_loss / len(val_loader.dataset)
        avg_val_loss_ev = avg_val_loss_norm * target_std

        train_loss_history.append(avg_train_loss_ev)
        val_loss_history_ev.append(avg_val_loss_ev)

        # Checkpoint logic
        if avg_val_loss_norm < best_val_loss:
            best_val_loss = avg_val_loss_norm
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Epoch {epoch:02d} | Train MAE: {avg_train_loss_ev:.4f} eV | Val MAE: {avg_val_loss_ev:.4f} eV -> [NEW BEST SAVED]")
        else:
            print(f"Epoch {epoch:02d} | Train MAE: {avg_train_loss_ev:.4f} eV | Val MAE: {avg_val_loss_ev:.4f} eV")

    return train_loss_history, val_loss_history_ev, checkpoint_path