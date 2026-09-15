"""Sionna OFDM frames with LDPC, phase offset, CFO, and seeded AWGN."""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
import tensorflow as tf
import sionna
from sionna.phy.mapping import Constellation, Mapper, Demapper
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.ofdm.modulator import OFDMModulator
from sionna.phy.ofdm.demodulator import OFDMDemodulator
from sionna.phy import config as sn_config
from .config import SimConfig

def set_global_seed(seed: int, seed_sionna: int | None = None) -> None:
    np.random.seed(int(seed))
    tf.random.set_seed(int(seed))
    if sn_config is not None and seed_sionna is not None:
        sn_config.seed = int(seed_sionna)

def complex_to_2ch_real(x: tf.Tensor, channels_last: bool) -> tf.Tensor:
    i = tf.math.real(x)
    q = tf.math.imag(x)
    return tf.stack([i, q], axis=-1 if channels_last else -2)

def expj(theta: tf.Tensor, cdtype: tf.DType) -> tf.Tensor:
    return tf.cast(tf.complex(tf.math.cos(theta), tf.math.sin(theta)), cdtype)

def awgn_manual_with_seed(x: tf.Tensor, no: tf.Tensor, seed: int, rdtype: tf.DType, cdtype: tf.DType) -> tf.Tensor:
    rng = tf.random.Generator.from_seed(int(seed))
    std = tf.sqrt(no / tf.constant(2.0, rdtype))
    nr = rng.normal(tf.shape(x), dtype=rdtype) * std
    ni = rng.normal(tf.shape(x), dtype=rdtype) * std
    w = tf.complex(nr, ni)
    return tf.cast(x, cdtype) + tf.cast(w, cdtype)

def build_sionna_blocks(cfg: SimConfig):
    m = int(cfg.bits_per_symbol)
    constellation = Constellation("qam", num_bits_per_symbol=m)
    mapper = Mapper(constellation=constellation)
    demapper = Demapper(cfg.demap_method, constellation=constellation)
    n = int(cfg.seq_length) * m
    encoder = LDPC5GEncoder(k=int(cfg.k), n=int(n), num_bits_per_symbol=m)
    decoder = LDPC5GDecoder(
        encoder=encoder, num_iter=int(cfg.dec_num_iter),
        return_infobits=True, hard_out=True, cn_update=str(cfg.cn_update),
    )
    return constellation, mapper, demapper, encoder, decoder

def make_deterministic_symbols_tf(cfg: SimConfig, N: int, num_symbols: int, bits_per_symbol: int, seed: int) -> Dict[str, tf.Tensor]:
    rng = tf.random.Generator.from_seed(int(seed))
    bits = rng.uniform([num_symbols * bits_per_symbol], minval=0, maxval=2, dtype=tf.int32)
    const = Constellation("qam", num_bits_per_symbol=int(bits_per_symbol))
    mapper = Mapper(constellation=const)
    x1 = tf.cast(mapper(tf.reshape(bits, [1, num_symbols * bits_per_symbol])), cfg.tf_cdtype)
    x = tf.tile(x1, [N, 1])
    return {"bits": bits, "x": x}

def simulate_full_frames_ofdm_tf(cfg: SimConfig) -> Dict[str, Any]:
    set_global_seed(cfg.seed, seed_sionna=cfg.seed)
    rng = tf.random.Generator.from_seed(int(cfg.seed))

    N, m, k = int(cfg.num_examples), int(cfg.bits_per_symbol), int(cfg.k)
    fft_size, cp_len = int(cfg.ofdm_fft_size), int(cfg.ofdm_cp_len)
    n_pilot_sym, n_data_sym = int(cfg.ofdm_num_pilot_symbols), int(cfg.ofdm_num_data_symbols)
    n_total_sym = n_pilot_sym + n_data_sym

    L = int(cfg.seq_length)
    if L != n_data_sym * fft_size:
        raise ValueError(f"OFDM requires seq_length = ofdm_num_data_symbols*ofdm_fft_size ({n_data_sym}*{fft_size}={n_data_sym*fft_size}), got {L}.")

    n = L * m
    R = float(k) / float(n)

    _, mapper_data, _, encoder, _ = build_sionna_blocks(cfg)
    info_bits = rng.uniform([N, k], minval=0, maxval=2, dtype=tf.int32)
    coded_bits_f = encoder(tf.cast(info_bits, cfg.tf_rdtype))
    coded_bits = tf.cast(coded_bits_f, tf.int32)

    x_data_flat = tf.cast(mapper_data(coded_bits), cfg.tf_cdtype)
    x_data_grid = tf.reshape(x_data_flat, [N, n_data_sym, fft_size])

    pilot_bps = int(cfg.pilot_bits_per_symbol)
    pilot_pack = make_deterministic_symbols_tf(cfg, N=N, num_symbols=n_pilot_sym * fft_size, bits_per_symbol=pilot_bps, seed=int(cfg.pilot_seed))
    pilot_bits = pilot_pack["bits"]
    x_pilot_grid = tf.reshape(pilot_pack["x"], [N, n_pilot_sym, fft_size])

    x_grid = tf.concat([x_pilot_grid, x_data_grid], axis=1)
    mod = OFDMModulator(cyclic_prefix_length=cp_len)
    x_time = tf.cast(mod(x_grid), cfg.tf_cdtype)

    P_time = n_pilot_sym * (fft_size + cp_len)
    x_pilot_time = x_time[:, :P_time]

    T_time = int(n_total_sym * (fft_size + cp_len))
    phi0 = rng.uniform([N, 1], minval=cfg.phase_min, maxval=cfg.phase_max, dtype=cfg.tf_rdtype)

    if str(cfg.cfo_mode).lower() == "epsilon":
        eps = rng.uniform([N, 1], minval=cfg.cfo_eps_min, maxval=cfg.cfo_eps_max, dtype=cfg.tf_rdtype)
        cfo = eps / tf.constant(float(fft_size), dtype=cfg.tf_rdtype)
        cfo_eps = eps
        cfo_unit = "cycles/sample (from epsilon)"
    else:
        cfo = rng.uniform([N, 1], minval=cfg.cfo_norm_min, maxval=cfg.cfo_norm_max, dtype=cfg.tf_rdtype)
        cfo_eps = cfo * tf.constant(float(fft_size), dtype=cfg.tf_rdtype)
        cfo_unit = "cycles/sample"

    t = tf.range(T_time, dtype=cfg.tf_rdtype)[None, :]
    phi_total = phi0 + tf.constant(2.0*np.pi, cfg.tf_rdtype) * cfo * t
    x_tx_phase = x_time * expj(phi_total, cfg.tf_cdtype)

    ebn0_db = rng.uniform([N], minval=cfg.ebn0_db_min, maxval=cfg.ebn0_db_max, dtype=cfg.tf_rdtype)
    ebn0_lin = tf.pow(tf.constant(10.0, cfg.tf_rdtype), ebn0_db / 10.0)

    overhead_val = 1.0
    if cfg.include_pilot_overhead_in_no: overhead_val *= (n_total_sym / float(n_data_sym))
    if cfg.include_cp_overhead_in_no: overhead_val *= ((fft_size + cp_len) / float(fft_size))
    overhead = tf.constant(overhead_val, dtype=cfg.tf_rdtype)

    es = tf.reduce_mean(tf.abs(x_tx_phase) ** 2, axis=1, keepdims=True)
    no = es / (tf.reshape(ebn0_lin, [-1, 1]) * (m * R)) * overhead

    noise_seed = cfg.seed + 101 if cfg.seed_noise is None else int(cfg.seed_noise)
    y_rx = awgn_manual_with_seed(x_tx_phase, no, seed=noise_seed, rdtype=cfg.tf_rdtype, cdtype=cfg.tf_cdtype)

    return {
        "info_bits": info_bits, "coded_bits": coded_bits, "pilot_bits": pilot_bits,
        "x_pilot": x_pilot_time, "x_data": x_data_flat, "x_grid": x_grid, "x_tx": x_time, "y_rx": y_rx,
        "phi0": phi0, "cfo_norm": cfo, "cfo_eps": cfo_eps, "ebn0_db": ebn0_db, "no": no,
        "meta": {
            "waveform": "ofdm", "N": N, "m": m, "k": k, "n": int(n), "R": float(R), "L": int(L),
            "P": int(P_time), "T": int(T_time), "fft_size": fft_size, "cp_len": cp_len,
            "num_pilot_ofdm_symbols": n_pilot_sym, "num_data_ofdm_symbols": n_data_sym, "num_total_ofdm_symbols": n_total_sym,
            "ofdm_l_min": int(cfg.ofdm_l_min), "seed": int(cfg.seed), "seed_noise": int(noise_seed), "pilot_seed": int(cfg.pilot_seed),
            "pilot_bits_per_symbol": int(cfg.pilot_bits_per_symbol), "include_pilot_overhead_in_no": bool(cfg.include_pilot_overhead_in_no),
            "include_cp_overhead_in_no": bool(cfg.include_cp_overhead_in_no), "cfo_mode": str(cfg.cfo_mode), "cfo_unit": cfo_unit,
            "sionna_version": getattr(sionna, "__version__", "unknown"), "channel": "OFDM; time-domain phi0 + CFO ramp + AWGN",
        }
    }

def generate_dataset_offline(cfg: SimConfig, full: Dict[str, Any] | None = None) -> Dict[str, Any]:
    full = simulate_full_frames_ofdm_tf(cfg) if full is None else full
    waveform = full["meta"].get("waveform", "sc")
    P_meta, T_meta = int(full["meta"]["P"]), int(full["meta"]["T"])

    tx_iq_full = complex_to_2ch_real(full["x_tx"], cfg.channels_last).numpy().astype(np.float32)
    rx_iq_full = complex_to_2ch_real(full["y_rx"], cfg.channels_last).numpy().astype(np.float32)

    if waveform == "ofdm":
        fft_size, cp_len = int(full["meta"]["fft_size"]), int(full["meta"]["cp_len"])
        n_pilot_sym, n_data_sym = int(full["meta"]["num_pilot_ofdm_symbols"]), int(full["meta"]["num_data_ofdm_symbols"])
        P_time = n_pilot_sym * (fft_size + cp_len)
        D_time = n_data_sym * (fft_size + cp_len)

    if cfg.channels_last:
        tx_iq_pilot_time, rx_iq_pilot_time = tx_iq_full[:, :P_time, :], rx_iq_full[:, :P_time, :]
        tx_iq_data_time, rx_iq_data_time = tx_iq_full[:, P_time:P_time + D_time, :], rx_iq_full[:, P_time:P_time + D_time, :]
    else:
        tx_iq_pilot_time, rx_iq_pilot_time = tx_iq_full[:, :, :P_time], rx_iq_full[:, :, :P_time]
        tx_iq_data_time, rx_iq_data_time = tx_iq_full[:, :, P_time:P_time + D_time], rx_iq_full[:, :, P_time:P_time + D_time]

    ds = {
        "info_bits": full["info_bits"].numpy().astype(np.uint8),
        "coded_bits": full["coded_bits"].numpy().astype(np.uint8),
        "pilot_bits": full["pilot_bits"].numpy().astype(np.uint8),
        "tx_iq_full_time": tx_iq_full,
        "rx_iq_full_time": rx_iq_full,
        "tx_iq_pilot_time": tx_iq_pilot_time,
        "rx_iq_pilot_time": rx_iq_pilot_time,
        "tx_iq_data_time": tx_iq_data_time,
        "rx_iq_data_time": rx_iq_data_time,
        "phi0": full["phi0"].numpy().astype(np.float32),
        "cfo_norm": full["cfo_norm"].numpy().astype(np.float32),
        "ebn0_db": full["ebn0_db"].numpy().astype(np.float32),
        "no": full["no"].numpy().astype(np.float32),
        "meta": full["meta"],
    }

    if waveform == "ofdm":
        fft_size, cp_len = int(full["meta"]["fft_size"]), int(full["meta"]["cp_len"])
        l_min = int(full["meta"].get("ofdm_l_min", 0))
        demod = OFDMDemodulator(fft_size=fft_size, l_min=l_min, cyclic_prefix_length=cp_len)
        y_grid_raw = demod(full["y_rx"])
        ds["rx_grid_raw"] = complex_to_2ch_real(y_grid_raw, channels_last=True).numpy().astype(np.float32)
        ds["tx_grid"] = complex_to_2ch_real(full["x_grid"], channels_last=True).numpy().astype(np.float32)
        ds["cfo_eps"] = full["cfo_eps"].numpy().astype(np.float32)

    return ds
