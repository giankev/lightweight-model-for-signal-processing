"""Dataset split and supervised training from notebook 09."""
import random
from dataclasses import replace
import numpy as np
import tensorflow as tf
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from .config import SimConfig, training_seeds
from .simulation import generate_dataset_offline
from .models import HybridNeuralReceiverOFDM, build_model

def set_seed_all(seed: int = 46):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(int(seed))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_pytorch_ofdm_loaders(base_cfg: SimConfig, batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    """Generate the seeded 80/20 SNR mixture and random 85/15 train/val split."""
    if base_cfg.num_examples < 3:
        raise ValueError("Training requires at least 3 examples for BatchNorm and validation.")
    train_count = int(0.85 * base_cfg.num_examples)
    if batch_size < 2 or train_count % batch_size == 1:
        raise ValueError("Choose a batch size leaving no singleton training batch (BatchNorm).")
    main_count = int(0.8 * base_cfg.num_examples)

    seeds = training_seeds(base_cfg)
    cfg_main = replace(base_cfg, num_examples=main_count, ebn0_db_min=4.0,
                       ebn0_db_max=8.0, seed=seeds["main"], seed_noise=seeds["main_noise"])
    cfg_rand = replace(base_cfg, num_examples=base_cfg.num_examples - main_count,
                       ebn0_db_min=0.0, ebn0_db_max=12.0,
                       seed=seeds["broad"], seed_noise=seeds["broad_noise"])

    ds_main = generate_dataset_offline(cfg_main)
    ds_rand = generate_dataset_offline(cfg_rand)

    x_rx = np.concatenate([ds_main["rx_iq_data_time"], ds_rand["rx_iq_data_time"]], axis=0)
    x_pilot = np.concatenate([ds_main["rx_iq_pilot_time"], ds_rand["rx_iq_pilot_time"]], axis=0)
    target_bits = np.concatenate([ds_main["coded_bits"], ds_rand["coded_bits"]], axis=0)

    tx_grid_main = ds_main["tx_grid"]
    tx_grid_rand = ds_rand["tx_grid"]
    n_pilot_sym = base_cfg.ofdm_num_pilot_symbols

    x_data_main = tx_grid_main[:, n_pilot_sym:, :, :].reshape(tx_grid_main.shape[0], -1, 2)
    x_data_rand = tx_grid_rand[:, n_pilot_sym:, :, :].reshape(tx_grid_rand.shape[0], -1, 2)
    x_data = np.concatenate([x_data_main, x_data_rand], axis=0)

    phi0 = np.concatenate([ds_main["phi0"], ds_rand["phi0"]], axis=0)
    cfo = np.concatenate([ds_main["cfo_norm"], ds_rand["cfo_norm"]], axis=0)

    x_rx_t = torch.tensor(x_rx).permute(0, 2, 1).float()
    x_pilot_t = torch.tensor(x_pilot).permute(0, 2, 1).float()
    target_bits_t = torch.tensor(target_bits).reshape(-1, base_cfg.seq_length, 4).permute(0, 2, 1).float()
    target_iq_t = torch.tensor(x_data).permute(0, 2, 1).float()

    phi0_t = torch.tensor(phi0).float().squeeze(-1)
    cfo_t = torch.tensor(cfo).float().squeeze(-1)

    dataset = TensorDataset(x_rx_t, x_pilot_t, phi0_t, cfo_t, target_bits_t, target_iq_t)

    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seeds["split"]),
    )

    return DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                      generator=torch.Generator().manual_seed(seeds["shuffle"])), DataLoader(val_ds, batch_size=batch_size, shuffle=False)


def train(cfg: SimConfig, device: str = "cpu") -> HybridNeuralReceiverOFDM:
    """Train and return the final-epoch model (no checkpoint selection)."""
    set_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    train_loader, val_loader = get_pytorch_ofdm_loaders(cfg, cfg.batch_size)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_mse = nn.MSELoss()

    for epoch in range(cfg.epochs):
        model.train()
        metrics = {'loss': 0, 'ber': 0}

        mix = min(1.0, epoch / cfg.teacher_forcing_epochs) # Keep teacher forcing

        aux_w = cfg.auxiliary_weight

        for x, p, y_p, y_c, target_bits, target_iq in train_loader:
            x, p, y_p, y_c = x.to(device), p.to(device), y_p.to(device), y_c.to(device)
            target_bits, target_iq = target_bits.to(device), target_iq.to(device)
            smooth_target_bits = target_bits * (1 - cfg.label_smoothing) + cfg.label_smoothing / 2

            optimizer.zero_grad()
            bit_logits, est_phi, est_cfo, est_iq = model(x, p, gt_phi=y_p, gt_cfo=y_c, mix_ratio=mix)

            loss_bits = criterion_bce(bit_logits, smooth_target_bits)
            loss_phi = criterion_mse(est_phi, y_p)
            loss_cfo = criterion_mse(est_cfo * 1000.0, y_c * 1000.0)
            loss_iq = criterion_mse(est_iq, target_iq)
            loss = loss_bits + aux_w * (cfg.phase_loss_weight * loss_phi + cfg.cfo_loss_weight * loss_cfo + cfg.iq_loss_weight * loss_iq)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.gradient_clip)
            optimizer.step()

            metrics['loss'] += loss.item()
            with torch.no_grad():
                pred_bits = (bit_logits > 0).float()
                metrics['ber'] += (pred_bits != target_bits).float().mean().item()

        scheduler.step()
        n_train = len(train_loader)
        model.eval()
        val_metrics = {'loss':0, 'ber':0}
        with torch.no_grad():
            for x, p, y_p, y_c, target_bits, target_iq in val_loader:
                x, p, y_p, y_c = x.to(device), p.to(device), y_p.to(device), y_c.to(device)
                target_bits, target_iq = target_bits.to(device), target_iq.to(device)
                bit_logits, est_phi, est_cfo, est_iq = model(x, p, gt_phi=y_p, gt_cfo=y_c, mix_ratio=1.0)

                loss_bits = criterion_bce(bit_logits, target_bits)
                loss_phi = criterion_mse(est_phi, y_p)
                loss_cfo = criterion_mse(est_cfo * 1000.0, y_c * 1000.0)
                loss_iq = criterion_mse(est_iq, target_iq)
                loss = loss_bits + aux_w * (cfg.phase_loss_weight * loss_phi + cfg.cfo_loss_weight * loss_cfo + cfg.iq_loss_weight * loss_iq)

                val_metrics['loss'] += loss.item()
                pred_bits = (bit_logits > 0).float()
                val_metrics['ber'] += (pred_bits != target_bits).float().mean().item()

        n_val = len(val_loader)
        print(f"Ep {epoch+1:02d}/{cfg.epochs} | "
              f"Train Loss: {metrics['loss']/n_train:.4f} (BER: {metrics['ber']/n_train*100:.2f}%) | "
              f"Val Loss: {val_metrics['loss']/n_val:.4f} (BER: {val_metrics['ber']/n_val*100:.2f}%)")
    return model
