"""Lightweight PyTorch model on synthetic OFDM dataset inspired by dataset_comunication.ipynb.

This script keeps the same OFDM generation logic used in the notebook:
- OFDM grid with pilot symbols + data symbols
- IFFT + CP
- time-domain phase/CFO impairment and AWGN
- pilot-aided phase/CFO estimation for a classical baseline

Then it trains a lightweight hybrid model that receives I/Q + baseline LLR and
must beat the classical OFDM hard-demapper baseline.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class OFDMConfig:
    n_samples: int = 8000
    bits_per_symbol: int = 4
    fft_size: int = 64
    cp_len: int = 16
    n_pilot_symbols: int = 1
    n_data_symbols: int = 2
    ebn0_db_min: float = 0.0
    ebn0_db_max: float = 12.0
    cfo_norm_min: float = -2e-4  # cycles/sample
    cfo_norm_max: float = 2e-4
    seed: int = 46

    @property
    def seq_len(self) -> int:
        return self.fft_size * self.n_data_symbols


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def bits_to_16qam(bits: np.ndarray) -> np.ndarray:
    b0, b1, b2, b3 = [bits[..., i] for i in range(4)]
    i_level = (1 - 2 * b0) * (3 - 2 * b1)
    q_level = (1 - 2 * b2) * (3 - 2 * b3)
    return ((i_level + 1j * q_level) / math.sqrt(10.0)).astype(np.complex64)


def hard_bits_16qam(sym: np.ndarray) -> np.ndarray:
    z = sym * math.sqrt(10.0)
    i, q = z.real, z.imag
    b0 = (i < 0).astype(np.int64)
    b1 = (np.abs(i) < 2).astype(np.int64)
    b2 = (q < 0).astype(np.int64)
    b3 = (np.abs(q) < 2).astype(np.int64)
    return np.stack([b0, b1, b2, b3], axis=-1)


def baseline_llr_16qam(sym: np.ndarray, no: np.ndarray) -> np.ndarray:
    z = sym * math.sqrt(10.0)
    i, q = z.real, z.imag
    no = no.astype(np.float32)[:, None]
    llr_b0 = -2.0 * i / (no + 1e-8)
    llr_b2 = -2.0 * q / (no + 1e-8)
    llr_b1 = 2.0 * (2.0 - np.abs(i)) / (no + 1e-8)
    llr_b3 = 2.0 * (2.0 - np.abs(q)) / (no + 1e-8)
    return np.stack([llr_b0, llr_b1, llr_b2, llr_b3], axis=-1).astype(np.float32)


def symbol_to_class(bits4: np.ndarray) -> np.ndarray:
    return (bits4[..., 0] * 8 + bits4[..., 1] * 4 + bits4[..., 2] * 2 + bits4[..., 3]).astype(np.int64)


def class_to_bits(cls: np.ndarray) -> np.ndarray:
    b = np.zeros(cls.shape + (4,), dtype=np.int64)
    b[..., 0] = (cls >> 3) & 1
    b[..., 1] = (cls >> 2) & 1
    b[..., 2] = (cls >> 1) & 1
    b[..., 3] = cls & 1
    return b


def ofdm_modulate(grid: np.ndarray, cp_len: int) -> np.ndarray:
    # grid: [N, n_sym, fft]
    td = np.fft.ifft(grid, axis=-1) * np.sqrt(grid.shape[-1])
    cp = td[..., -cp_len:]
    return np.concatenate([cp, td], axis=-1).reshape(grid.shape[0], -1).astype(np.complex64)


def ofdm_demodulate(wave: np.ndarray, fft_size: int, cp_len: int, n_symbols: int) -> np.ndarray:
    x = wave.reshape(wave.shape[0], n_symbols, fft_size + cp_len)
    no_cp = x[..., cp_len:]
    grid = np.fft.fft(no_cp, axis=-1) / np.sqrt(fft_size)
    return grid.astype(np.complex64)


def estimate_phi_cfo_ml(y_pilot: np.ndarray, x_pilot: np.ndarray, cfo_min: float, cfo_max: float, grid_size: int = 121) -> Tuple[np.ndarray, np.ndarray]:
    n, p = y_pilot.shape
    t = np.arange(p, dtype=np.float32)[None, :]
    cfo_grid = np.linspace(cfo_min, cfo_max, grid_size, dtype=np.float32)
    best_metric = np.full((n,), -1e30, dtype=np.float32)
    best_cfo = np.zeros((n,), dtype=np.float32)
    best_phi = np.zeros((n,), dtype=np.float32)

    # vectorized over samples, loop only over candidate CFOs
    for cfo in cfo_grid:
        der = np.exp(-1j * 2 * math.pi * cfo * t).astype(np.complex64)
        z = np.sum(y_pilot * np.conj(x_pilot) * der, axis=1)
        metric = np.abs(z)
        mask = metric > best_metric
        best_metric[mask] = metric[mask]
        best_cfo[mask] = cfo
        best_phi[mask] = np.angle(z[mask]).astype(np.float32)

    return best_phi[:, None], best_cfo[:, None]


def generate_dataset(cfg: OFDMConfig) -> Dict[str, np.ndarray]:
    set_seed(cfg.seed)
    n, fft = cfg.n_samples, cfg.fft_size
    n_total_sym = cfg.n_pilot_symbols + cfg.n_data_symbols
    l = cfg.seq_len

    # data bits/symbols
    bits_data = np.random.randint(0, 2, size=(n, l, 4), dtype=np.int64)
    x_data = bits_to_16qam(bits_data).reshape(n, cfg.n_data_symbols, fft)

    # deterministic pilots (QPSK) similar notebook approach
    pilot_bits = np.random.RandomState(cfg.seed + 999).randint(0, 2, size=(cfg.n_pilot_symbols * fft, 2))
    pilot_iq = (1 - 2 * pilot_bits[:, 0]) + 1j * (1 - 2 * pilot_bits[:, 1])
    pilot_iq = (pilot_iq / np.sqrt(2.0)).astype(np.complex64)
    x_pilot = np.tile(pilot_iq[None, :], (n, 1)).reshape(n, cfg.n_pilot_symbols, fft)

    x_grid = np.concatenate([x_pilot, x_data], axis=1)
    x_time = ofdm_modulate(x_grid, cfg.cp_len)

    tlen = x_time.shape[1]
    t = np.arange(tlen, dtype=np.float32)[None, :]
    phi0 = np.random.uniform(-math.pi, math.pi, size=(n, 1)).astype(np.float32)
    cfo = np.random.uniform(cfg.cfo_norm_min, cfg.cfo_norm_max, size=(n, 1)).astype(np.float32)

    y_clean = x_time * np.exp(1j * (phi0 + 2 * math.pi * cfo * t)).astype(np.complex64)

    # noise variance per complex sample with same accounting idea as notebook
    ebn0_db = np.random.uniform(cfg.ebn0_db_min, cfg.ebn0_db_max, size=(n,)).astype(np.float32)
    ebn0_lin = 10 ** (ebn0_db / 10.0)
    n_data_bits = cfg.bits_per_symbol * cfg.n_data_symbols * fft
    rate_like = 1.0  # uncoded synthetic target
    overhead = (n_total_sym / cfg.n_data_symbols) * ((fft + cfg.cp_len) / fft)
    no = (1.0 / (cfg.bits_per_symbol * rate_like * ebn0_lin)) * overhead

    w = (np.random.randn(n, tlen) + 1j * np.random.randn(n, tlen)).astype(np.complex64)
    w *= np.sqrt(no[:, None] / 2.0).astype(np.float32)
    y_rx = y_clean + w

    p_time = cfg.n_pilot_symbols * (fft + cfg.cp_len)
    x_pilot_time = x_time[:, :p_time]
    y_pilot_time = y_rx[:, :p_time]

    phi_hat, cfo_hat = estimate_phi_cfo_ml(y_pilot_time, x_pilot_time, cfg.cfo_norm_min, cfg.cfo_norm_max)

    # classical correction + demod
    t_all = np.arange(tlen, dtype=np.float32)[None, :]
    y_corr = y_rx * np.exp(-1j * (phi_hat + 2 * math.pi * cfo_hat * t_all)).astype(np.complex64)
    y_grid = ofdm_demodulate(y_corr, fft, cfg.cp_len, n_total_sym)
    y_data = y_grid[:, cfg.n_pilot_symbols:, :].reshape(n, l)

    llr = baseline_llr_16qam(y_data, no)
    bits_base = (llr > 0).astype(np.int64)
    err_mask = (bits_base != bits_data.reshape(n, l, 4)).astype(np.float32)

    feats = np.stack([y_data.real, y_data.imag, np.abs(y_data), np.angle(y_data) / math.pi], axis=-1).astype(np.float32)
    x_in = np.concatenate([feats, llr], axis=-1).astype(np.float32)

    return {
        "x": x_in,
        "y_cls": symbol_to_class(bits_data.reshape(n, l, 4)),
        "y_bits": bits_data.reshape(n, l, 4),
        "baseline_bits": bits_base,
        "err_mask": err_mask,
    }


class OFDMSymbolDataset(Dataset):
    def __init__(self, x: np.ndarray, y_cls: np.ndarray, err: np.ndarray):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y_cls)
        self.err = torch.from_numpy(err)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int):
        return self.x[i], self.y[i], self.err[i]


class EfficientHybridReceiver(nn.Module):
    def __init__(self, in_ch: int = 8, hidden: int = 64):
        super().__init__()
        self.in_proj = nn.Conv1d(in_ch, hidden, kernel_size=1)
        self.dw1 = nn.Conv1d(hidden, hidden, 5, padding=2, groups=hidden)
        self.pw1 = nn.Conv1d(hidden, hidden, 1)
        self.dw2 = nn.Conv1d(hidden, hidden, 3, padding=1, groups=hidden)
        self.pw2 = nn.Conv1d(hidden, hidden, 1)
        self.bn = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(0.1)
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.cls = nn.Linear(hidden * 2, 16)
        self.err = nn.Linear(hidden * 2, 4)

    def forward(self, x: torch.Tensor):
        x = x.transpose(1, 2)
        h = F.gelu(self.in_proj(x))
        h = h + F.gelu(self.pw1(self.dw1(h)))
        h = h + F.gelu(self.pw2(self.dw2(h)))
        h = self.drop(self.bn(h)).transpose(1, 2)
        h, _ = self.gru(h)
        return self.cls(h), self.err(h)


def ber_bits(pred_cls: np.ndarray, true_cls: np.ndarray) -> float:
    return float((class_to_bits(pred_cls) != class_to_bits(true_cls)).mean())


def hybrid_select(pred_cls: np.ndarray, pred_conf: np.ndarray, base_cls: np.ndarray, thr: float) -> np.ndarray:
    use_model = pred_conf >= thr
    return np.where(use_model, pred_cls, base_cls)


def train_and_eval() -> None:
    torch.set_num_threads(4)
    cfg = OFDMConfig()
    data = generate_dataset(cfg)

    n = cfg.n_samples
    n_tr = int(0.7 * n)
    n_va = int(0.15 * n)

    train_ds = OFDMSymbolDataset(data["x"][:n_tr], data["y_cls"][:n_tr], data["err_mask"][:n_tr])
    val_ds = OFDMSymbolDataset(data["x"][n_tr:n_tr+n_va], data["y_cls"][n_tr:n_tr+n_va], data["err_mask"][n_tr:n_tr+n_va])

    dl_tr = DataLoader(train_ds, batch_size=96, shuffle=True)
    dl_va = DataLoader(val_ds, batch_size=192, shuffle=False)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientHybridReceiver().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=14)
    best_val = 1e9
    best_state = None

    for ep in range(1, 15):
        model.train()
        tr_loss = 0.0
        for xb, yb, eb in dl_tr:
            xb, yb, eb = xb.to(dev), yb.to(dev), eb.to(dev)
            lg, er = model(xb)
            l1 = F.cross_entropy(lg.reshape(-1, 16), yb.reshape(-1))
            l2 = F.binary_cross_entropy_with_logits(er, eb)
            loss = l1 + 0.35 * l2
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb, eb in dl_va:
                xb, yb, eb = xb.to(dev), yb.to(dev), eb.to(dev)
                lg, er = model(xb)
                l1 = F.cross_entropy(lg.reshape(-1, 16), yb.reshape(-1))
                l2 = F.binary_cross_entropy_with_logits(er, eb)
                va_loss += (l1 + 0.25 * l2).item()

        sch.step()
        tr_mean = tr_loss / len(dl_tr)
        va_mean = va_loss / max(1, len(dl_va))
        print(f"Epoch {ep:02d} | train={tr_mean:.4f} | val={va_mean:.4f}", flush=True)

        if va_mean < best_val:
            best_val = va_mean
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    val_x = torch.from_numpy(data["x"][n_tr:n_tr+n_va]).to(dev)
    val_y = data["y_cls"][n_tr:n_tr+n_va]
    val_base_cls = symbol_to_class(data["baseline_bits"][n_tr:n_tr+n_va])

    test_x = torch.from_numpy(data["x"][n_tr+n_va:]).to(dev)
    test_y = data["y_cls"][n_tr+n_va:]
    test_base_bits = data["baseline_bits"][n_tr+n_va:]
    test_base_cls = symbol_to_class(test_base_bits)

    model.eval()
    with torch.no_grad():
        val_logits = model(val_x)[0]
        val_prob = torch.softmax(val_logits, dim=-1)
        val_conf, val_pred = val_prob.max(dim=-1)
        val_pred = val_pred.cpu().numpy()
        val_conf = val_conf.cpu().numpy()

        test_logits = model(test_x)[0]
        test_prob = torch.softmax(test_logits, dim=-1)
        test_conf, test_pred = test_prob.max(dim=-1)
        test_pred = test_pred.cpu().numpy()
        test_conf = test_conf.cpu().numpy()

    # tune confidence threshold on validation set for hybrid selection model/baseline
    thr_grid = np.linspace(0.35, 0.95, 25)
    best_thr, best_val_ber = 0.5, 1e9
    for thr in thr_grid:
        val_h = hybrid_select(val_pred, val_conf, val_base_cls, float(thr))
        b = ber_bits(val_h, val_y)
        if b < best_val_ber:
            best_val_ber = b
            best_thr = float(thr)

    pred_h = hybrid_select(test_pred, test_conf, test_base_cls, best_thr)

    ber_base = float((test_base_bits != class_to_bits(test_y)).mean())
    ber_model = ber_bits(pred_h, test_y)
    gain = (ber_base - ber_model) / max(1e-12, ber_base) * 100.0

    print("\n=== OFDM baseline vs lightweight DL ===", flush=True)
    print(f"Selected threshold (val): {best_thr:.3f}", flush=True)
    print(f"Baseline BER: {ber_base:.6f}", flush=True)
    print(f"Hybrid model BER: {ber_model:.6f}", flush=True)
    print(f"Relative gain: {gain:.2f}%", flush=True)

    if ber_model >= ber_base:
        raise RuntimeError("Model did not beat OFDM baseline. Rerun/tune required.")


if __name__ == "__main__":
    train_and_eval()
