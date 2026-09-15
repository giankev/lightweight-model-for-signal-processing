"""Classical baseline and calibrated neural LDPC evaluation."""
from __future__ import annotations
from dataclasses import replace
from typing import Any, Dict
import numpy as np
import tensorflow as tf
import torch
from sionna.phy.ofdm.demodulator import OFDMDemodulator
from .config import SimConfig
from .simulation import build_sionna_blocks, expj, simulate_full_frames_ofdm_tf, generate_dataset_offline
from .metrics import bit_error_rate, block_error_rate

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

    if waveform == "ofdm":
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


def evaluate(model, cfg: SimConfig, ebn0_values=range(13), batch_size: int = 128,
             device: str = "cpu") -> list[dict]:
    """Compare receivers on identical frames, using notebook LLR calibration."""
    if batch_size < 1:
        raise ValueError("Evaluation batch size must be positive.")
    model = model.to(device)
    model.eval()
    results = []
    for ebno in ebn0_values:
        cfg_eval = replace(cfg, ebn0_db_min=float(ebno), ebn0_db_max=float(ebno))
        full = simulate_full_frames_ofdm_tf(cfg_eval)
        baseline = classical_receiver(cfg_eval, full)["metrics"]
        ds = generate_dataset_offline(cfg_eval, full=full)
        x = torch.tensor(ds["rx_iq_data_time"]).permute(0, 2, 1).contiguous().float()
        pilots = torch.tensor(ds["rx_iq_pilot_time"]).permute(0, 2, 1).contiguous().float()
        logits = []
        with torch.no_grad():
            for start in range(0, cfg.num_examples, batch_size):
                bits, _, _, _ = model(x[start:start + batch_size].to(device),
                                      pilots[start:start + batch_size].to(device), mix_ratio=1.0)
                logits.append(bits.permute(0, 2, 1).reshape(bits.size(0), -1).cpu().numpy())
        llr = np.concatenate(logits)
        scale = 10 ** (ebno / 10.0) / 4.0 if ebno > 6 else 1.0 + ebno * 0.5
        _, _, _, _, decoder = build_sionna_blocks(cfg_eval)
        decoded = (decoder(tf.convert_to_tensor(llr * scale, dtype=tf.float32)).numpy() > 0.5)
        results.append({"ebn0_db": float(ebno), "classical": baseline, "neural": {
            "ber_pre_coded": bit_error_rate(llr > 0, ds["coded_bits"]),
            "ber_post_info": bit_error_rate(decoded, ds["info_bits"]),
            "bler_post_info": block_error_rate(decoded, ds["info_bits"]),
        }})
    return results
