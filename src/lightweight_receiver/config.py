"""Configuration for the pilot-starved experiment."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import tensorflow as tf

@dataclass(frozen=True)
class SimConfig:
    """Simulation defaults from notebook 09, plus its training hyperparameters."""
    num_examples: int = 10_000
    seq_length: int = 512
    bits_per_symbol: int = 4
    k: int = 1024
    pilot_seed: int = 999
    ebn0_db_min: float = 0.0
    ebn0_db_max: float = 12.0
    phase_min: float = -np.pi
    phase_max: float = np.pi
    cfo_mode: str = "direct"
    # Normalized CFO is cycles/sample; phi0 is radians at full-frame t=0.
    cfo_norm_min: float = -0.0002
    cfo_norm_max: float = 0.0002
    cfo_eps_min: float = -0.05
    cfo_eps_max: float = +0.05
    demap_method: str = "app"
    dec_num_iter: int = 15
    cn_update: str = "minsum"
    seed: int = 46
    seed_noise: Optional[int] = None
    llr_noise_scaling: float = 1.0
    waveform: str = "ofdm"
    pilot_bits_per_symbol: int = 2
    ofdm_fft_size: int = 64
    ofdm_cp_len: int = 16
    ofdm_num_pilot_symbols: int = 1
    ofdm_num_data_symbols: int = 8
    ofdm_l_min: int = 0
    include_pilot_overhead_in_no: bool = True
    include_cp_overhead_in_no: bool = True
    channels_last: bool = True
    tf_rdtype: tf.dtypes.DType = tf.float32
    tf_cdtype: tf.dtypes.DType = tf.complex64

    epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.02
    auxiliary_weight: float = 0.5
    phase_loss_weight: float = 0.5
    cfo_loss_weight: float = 0.5
    iq_loss_weight: float = 1.0
    gradient_clip: float = 1.0
    teacher_forcing_epochs: float = 10.0

    def __post_init__(self) -> None:
        if self.waveform != "ofdm" or self.bits_per_symbol != 4 or not self.channels_last:
            raise ValueError("This pipeline requires OFDM, 16-QAM, and channels_last=True.")
        if self.seq_length != self.ofdm_num_data_symbols * self.ofdm_fft_size:
            raise ValueError("seq_length must equal data symbols times FFT size.")
        if self.teacher_forcing_epochs <= 0:
            raise ValueError("teacher_forcing_epochs must be positive.")
        if self.num_examples < 1 or self.epochs < 1 or self.batch_size < 2:
            raise ValueError("Require positive examples/epochs and batch_size >= 2.")


def training_seeds(cfg: SimConfig) -> dict[str, int]:
    """Explicit independent data, noise, split, and shuffle streams."""
    noise_base = cfg.seed if cfg.seed_noise is None else cfg.seed_noise
    return {"main": cfg.seed + 1, "broad": cfg.seed + 2,
            "split": cfg.seed + 3, "shuffle": cfg.seed + 4,
            "main_noise": noise_base + 101, "broad_noise": noise_base + 102}
