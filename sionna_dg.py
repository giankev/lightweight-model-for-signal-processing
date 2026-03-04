from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import tensorflow as tf

import sionna_dg
from sionna.phy.mapping import Constellation, Mapper, Demapper
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.ofdm.modulator import OFDMModulator
from sionna.phy.ofdm.demodulator import OFDMDemodulator

try:
    from sionna.phy import config as sn_config
except Exception:
    sn_config = None


@dataclass(frozen=True)
class SimConfig:
    num_examples: int = 10_000
    seq_length: int = 128
    bits_per_symbol: int = 4
    k: int = 256
    pilot_len: int = 16
    pilot_seed: int = 999
    ebn0_db_min: float = 0.0
    ebn0_db_max: float = 10.0
    phase_min: float = -np.pi
    phase_max: float = np.pi
    cfo_mode: str = "direct"
    cfo_norm_min: float = -0.002
    cfo_norm_max: float = 0.002
    cfo_eps_min: float = -0.05
    cfo_eps_max: float = +0.05
    demap_method: str = "app"
    dec_num_iter: int = 20
    cn_update: str = "minsum"
    seed: int = 46
    seed_noise: Optional[int] = None
    llr_noise_scaling: float = 1.0
    waveform: str = "sc"
    pilot_bits_per_symbol: int = 2
    ofdm_fft_size: int = 64
    ofdm_cp_len: int = 16
    ofdm_num_pilot_symbols: int = 1
    ofdm_num_data_symbols: int = 2
    ofdm_l_min: int = 0
    include_pilot_overhead_in_no: bool = True
    include_cp_overhead_in_no: bool = True
    channels_last: bool = True
    tf_rdtype: tf.dtypes.DType = tf.float32
    tf_cdtype: tf.dtypes.DType = tf.complex64


def set_global_seed(seed: int, seed_sionna: Optional[int] = None) -> None:
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
    set_global_seed(cfg.seed)
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

    noise_seed = cfg.seed if cfg.seed_noise is None else int(cfg.seed_noise)
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
            "sionna_version": getattr(sionna_dg, "__version__", "unknown"), "channel": "OFDM; time-domain phi0 + CFO ramp + AWGN",
        }
    }

def simulate_full_frames_sc_tf(cfg: SimConfig) -> Dict[str, Any]:
    set_global_seed(cfg.seed)
    rng = tf.random.Generator.from_seed(int(cfg.seed))

    N, L, m, P = int(cfg.num_examples), int(cfg.seq_length), int(cfg.bits_per_symbol), int(cfg.pilot_len)
    T = P + L
    n = L * m
    k = int(cfg.k)
    R = float(k) / float(n)

    _, mapper, _, encoder, _ = build_sionna_blocks(cfg)

    pilot_pack = make_deterministic_symbols_tf(cfg, N=N, num_symbols=P, bits_per_symbol=int(cfg.pilot_bits_per_symbol), seed=int(cfg.pilot_seed))
    pilot_bits = pilot_pack["bits"]
    x_pilot = pilot_pack["x"]

    info_bits = rng.uniform([N, k], minval=0, maxval=2, dtype=tf.int32)
    coded_bits_f = encoder(tf.cast(info_bits, cfg.tf_rdtype))
    coded_bits = tf.cast(coded_bits_f, tf.int32)

    x_data = tf.cast(mapper(coded_bits), cfg.tf_cdtype)
    x_tx = tf.concat([x_pilot, x_data], axis=1)

    phi0 = rng.uniform([N, 1], minval=cfg.phase_min, maxval=cfg.phase_max, dtype=cfg.tf_rdtype)
    cfo = rng.uniform([N, 1], minval=cfg.cfo_norm_min, maxval=cfg.cfo_norm_max, dtype=cfg.tf_rdtype)
    t = tf.range(T, dtype=cfg.tf_rdtype)[None, :]
    phi_total = phi0 + tf.constant(2.0 * np.pi, cfg.tf_rdtype) * cfo * t
    x_tx_phase = x_tx * expj(phi_total, cfg.tf_cdtype)

    ebn0_db = rng.uniform([N], minval=cfg.ebn0_db_min, maxval=cfg.ebn0_db_max, dtype=cfg.tf_rdtype)
    ebn0_lin = tf.pow(tf.constant(10.0, cfg.tf_rdtype), ebn0_db / 10.0)

    overhead_val = 1.0
    if cfg.include_pilot_overhead_in_no: overhead_val *= (P + L) / float(L)
    overhead = tf.constant(overhead_val, dtype=cfg.tf_rdtype)

    es = tf.reduce_mean(tf.abs(x_tx_phase) ** 2, axis=1, keepdims=True)
    no = es / (tf.reshape(ebn0_lin, [-1, 1]) * (m * R)) * overhead

    noise_seed = cfg.seed if cfg.seed_noise is None else int(cfg.seed_noise)
    y_rx = awgn_manual_with_seed(x_tx_phase, no, seed=noise_seed, rdtype=cfg.tf_rdtype, cdtype=cfg.tf_cdtype)

    return {
        "info_bits": info_bits, "coded_bits": coded_bits, "pilot_bits": pilot_bits,
        "x_pilot": x_pilot, "x_data": x_data, "x_tx": x_tx, "y_rx": y_rx,
        "phi0": phi0, "cfo_norm": cfo, "ebn0_db": ebn0_db, "no": no,
        "meta": {
            "waveform": "sc", "N": int(N), "L": int(L), "P": int(P), "T": int(T),
            "m": int(m), "k": int(k), "n": int(n), "R": float(R),
            "seed": int(cfg.seed), "seed_noise": int(noise_seed), "pilot_seed": int(cfg.pilot_seed),
            "demap_method": str(cfg.demap_method), "dec_num_iter": int(cfg.dec_num_iter),
            "cn_update": str(cfg.cn_update), "channels_last": bool(cfg.channels_last),
            "include_pilot_overhead_in_no": bool(cfg.include_pilot_overhead_in_no),
            "include_cp_overhead_in_no": bool(cfg.include_cp_overhead_in_no),
            "sionna_version": getattr(sionna_dg, "__version__", "unknown"),
            "channel": "single-carrier; phi0 + CFO-like ramp + AWGN", "cfo_unit": "cycles/symbol",
        }
    }

def simulate_full_frames_tf(cfg: SimConfig) -> Dict[str, Any]:
    if cfg.waveform.lower() == "sc":
        return simulate_full_frames_sc_tf(cfg)
    if cfg.waveform.lower() == "ofdm":
        return simulate_full_frames_ofdm_tf(cfg)
    raise ValueError(f"Unknown waveform={cfg.waveform}. Use 'sc' or 'ofdm'.")

def generate_dataset_offline(cfg: SimConfig) -> Dict[str, Any]:
    full = simulate_full_frames_tf(cfg)
    waveform = full["meta"].get("waveform", "sc")
    P_meta, T_meta = int(full["meta"]["P"]), int(full["meta"]["T"])

    tx_iq_full = complex_to_2ch_real(full["x_tx"], cfg.channels_last).numpy().astype(np.float32)
    rx_iq_full = complex_to_2ch_real(full["y_rx"], cfg.channels_last).numpy().astype(np.float32)

    if waveform == "sc":
        P_time, D_time = P_meta, T_meta - P_meta
    elif waveform == "ofdm":
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

def estimate_phi_cfo_ml_grid(cfg: SimConfig, y_pilot: tf.Tensor, x_pilot: tf.Tensor, cfo_min: float, cfo_max: float, grid_size: int = 129):
    rdtype, cdtype = cfg.tf_rdtype, cfg.tf_cdtype
    P = tf.shape(y_pilot)[1]
    two_pi, eps_f = tf.constant(2.0*np.pi, dtype=rdtype), tf.constant(1e-9, dtype=rdtype)
    x_pow = tf.abs(x_pilot)**2
    r = y_pilot * tf.math.conj(x_pilot) / tf.cast(x_pow + eps_f, cdtype)
    f_grid = tf.linspace(tf.constant(float(cfo_min), rdtype), tf.constant(float(cfo_max), rdtype), int(grid_size))
    p = tf.cast(tf.range(P)[None, :], rdtype)
    f = tf.reshape(f_grid, [-1, 1])
    theta = -two_pi * f * p
    E = tf.cast(tf.complex(tf.math.cos(theta), tf.math.sin(theta)), cdtype)
    z = tf.einsum("np,mp->nm", r, E)
    mags = tf.abs(z)
    idx = tf.argmax(mags, axis=1, output_type=tf.int32)
    cfo_hat = tf.gather(f_grid, idx)[:, None]
    z_best = tf.gather(z, idx, axis=1, batch_dims=1)[:, None]
    phi_hat = tf.cast(tf.math.angle(z_best), rdtype)
    denom = tf.reduce_sum(tf.abs(r), axis=1, keepdims=True) + eps_f
    conf = tf.abs(z_best) / denom
    return phi_hat, cfo_hat, conf

def classical_receiver_with_phi_cfo(cfg: SimConfig, full: Dict[str, Any], phi_hat: tf.Tensor, cfo_hat: tf.Tensor) -> Dict[str, Any]:
    _, _, demapper, _, decoder = build_sionna_blocks(cfg)
    meta, no = full["meta"], tf.cast(full["no"], cfg.tf_rdtype)
    y_rx, coded_bits, info_bits = full["y_rx"], tf.cast(full["coded_bits"], tf.uint8), tf.cast(full["info_bits"], tf.uint8)
    waveform = meta.get("waveform", "sc")
    T = int(meta["T"])
    t = tf.cast(tf.range(T)[None, :], cfg.tf_rdtype)
    phi_total_hat = phi_hat + tf.constant(2.0*np.pi, dtype=cfg.tf_rdtype) * cfo_hat * t
    y_corr = y_rx * expj(-phi_total_hat, cfg.tf_cdtype)

    if waveform == "sc":
        y_data_symbols = y_corr[:, int(meta["P"]):]
    elif waveform == "ofdm":
        fft_size, cp_len, n_pilot_sym = int(meta["fft_size"]), int(meta["cp_len"]), int(meta["num_pilot_ofdm_symbols"])
        demod = OFDMDemodulator(fft_size=fft_size, l_min=int(meta.get("ofdm_l_min", 0)), cyclic_prefix_length=cp_len)
        y_grid = demod(y_corr)
        y_data_grid = y_grid[:, n_pilot_sym:, :]
        y_data_symbols = tf.reshape(y_data_grid, [tf.shape(y_data_grid)[0], -1])

    alpha = tf.constant(float(getattr(cfg, "llr_noise_scaling", 1.0)), dtype=cfg.tf_rdtype)
    no_eff = no * alpha
    llr = tf.cast(demapper(y_data_symbols, no_eff), cfg.tf_rdtype)
    hard_coded = tf.cast(llr > 0.0, tf.uint8)
    info_hat = tf.cast(decoder(llr) > 0.5, tf.uint8)

    ber_pre = tf.reduce_mean(tf.cast(hard_coded != coded_bits, cfg.tf_rdtype))
    ber_post = tf.reduce_mean(tf.cast(info_hat != info_bits, cfg.tf_rdtype))
    bler_post = tf.reduce_mean(tf.cast(tf.reduce_any(info_hat != info_bits, axis=1), cfg.tf_rdtype))

    return {
        "phi_hat": phi_hat, "cfo_hat": cfo_hat, "llr": llr, "info_bits_hat": info_hat,
        "metrics": { "ber_pre_coded": float(ber_pre.numpy()), "ber_post_info": float(ber_post.numpy()), "bler_post_info": float(bler_post.numpy()) }
    }

def classical_receiver(cfg: SimConfig, full: Dict[str, Any]) -> Dict[str, Any]:
    meta = full["meta"]
    waveform, P = meta.get("waveform", "sc"), int(meta["P"])
    y_pilot, x_pilot = full["y_rx"][:, :P], full["x_pilot"]

    if waveform == "ofdm":
        fft_size = int(meta["fft_size"])
        if str(cfg.cfo_mode).lower() == "epsilon":
            cfo_min, cfo_max = float(cfg.cfo_eps_min) / float(fft_size), float(cfg.cfo_eps_max) / float(fft_size)
        else:
            cfo_min, cfo_max = float(cfg.cfo_norm_min), float(cfg.cfo_norm_max)
    else:
        cfo_min, cfo_max = float(cfg.cfo_norm_min), float(cfg.cfo_norm_max)

    if P >= 2:
        phi_hat, cfo_hat, conf = estimate_phi_cfo_ml_grid(cfg, y_pilot, x_pilot, cfo_min=cfo_min, cfo_max=cfo_max, grid_size=129)
    else:
        z = tf.reduce_sum(y_pilot * tf.math.conj(x_pilot), axis=1, keepdims=True)
        phi_hat, cfo_hat, conf = tf.cast(tf.math.angle(z), cfg.tf_rdtype), tf.zeros_like(phi_hat), tf.ones_like(phi_hat)

    out = classical_receiver_with_phi_cfo(cfg, full, phi_hat, cfo_hat)
    out["conf"] = conf
    return out