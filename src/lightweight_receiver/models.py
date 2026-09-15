"""Linear Attention hybrid architecture extracted from notebook 09."""
import torch
from torch import nn
from torch.nn import functional as F

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.gamma = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)

    def forward(self, x, latent_z):
        style_scale = self.gamma(latent_z).unsqueeze(1)
        style_shift = self.beta(latent_z).unsqueeze(1)
        x = x * (1 + style_scale) + style_shift
        b, n, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, self.heads, self.dim_head).transpose(1, 2), qkv)
        q = q.softmax(dim=-1) * self.scale
        k = k.softmax(dim=-2)
        context = torch.matmul(k.transpose(-1, -2), v)
        out = torch.matmul(q, context)
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.to_out(out)

class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(x + self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))))

class HybridNeuralReceiverOFDM(nn.Module):
    """Notebook receiver; retains its original CFO units and data-local time ramp."""
    def __init__(self, seq_len=160, n_pilots=80, ofdm_fft_size=64, ofdm_cp_len=16):
        super().__init__()
        self.fft_size = ofdm_fft_size
        self.cp_len = ofdm_cp_len

        in_dim = (2 * seq_len) + (2 * n_pilots)
        self.input_norm = nn.BatchNorm1d(in_dim)

        self.estimator = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 2)
        )
        self.latent_proj = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 64), nn.Tanh())

        self.feat_extract = nn.Conv1d(2, 64, 7, padding=3)
        self.res1 = ResidualBlock1D(64)
        self.attn = LinearAttention(64)
        self.norm_attn = nn.LayerNorm(64)
        self.res2 = ResidualBlock1D(64)

        self.iq_head = nn.Conv1d(64, 2, 1)
        self.demap_mlp = nn.Sequential(
            nn.Conv1d(64, 32, 1), nn.ReLU(),
            nn.Conv1d(32, 16, 1), nn.ReLU(),
            nn.Conv1d(16, 4, 1)
        )

    def forward(self, x, pilots, gt_phi=None, gt_cfo=None, mix_ratio=0.0):
        x_flat = x.reshape(x.size(0), -1)
        p_flat = pilots.reshape(pilots.size(0), -1)
        est_feats = self.input_norm(torch.cat([x_flat, p_flat], dim=1))

        est_params = self.estimator(est_feats)
        est_phi = est_params[:, 0]
        est_cfo = est_params[:, 1] / 1000.0

        if gt_phi is not None and mix_ratio < 1.0:
            phi = (1 - mix_ratio) * gt_phi + mix_ratio * est_phi
            cfo = (1 - mix_ratio) * gt_cfo + mix_ratio * est_cfo
        else:
            phi, cfo = est_phi, est_cfo

        z_input = torch.stack([torch.cos(phi), cfo * 1000.0], dim=1)
        z = self.latent_proj(z_input)

        B, _, L = x.shape
        t = torch.arange(L, device=x.device).float().unsqueeze(0)
        # Preserve notebook behavior: no 2*pi factor or pilot-time offset.
        phase_ramp = phi.unsqueeze(1) + (cfo.unsqueeze(1) * t)
        cos_t, sin_t = torch.cos(-phase_ramp), torch.sin(-phase_ramp)
        r_I = x[:, 0, :] * cos_t - x[:, 1, :] * sin_t
        r_Q = x[:, 0, :] * sin_t + x[:, 1, :] * cos_t

        x_comp = torch.complex(r_I, r_Q)
        n_sym = L // (self.fft_size + self.cp_len)
        x_sym = x_comp.reshape(B, n_sym, self.fft_size + self.cp_len)

        x_no_cp = x_sym[:, :, self.cp_len:]
        x_freq = torch.fft.fft(x_no_cp, norm="ortho", dim=-1)
        x_freq = torch.fft.fftshift(x_freq, dim=-1)
        x_freq_flat = x_freq.reshape(B, n_sym * self.fft_size)

        feat_in = torch.stack([x_freq_flat.real, x_freq_flat.imag], dim=1)

        feat = F.relu(self.feat_extract(feat_in))
        feat = self.res1(feat)
        feat_attn = feat.permute(0, 2, 1)
        feat_attn = self.attn(self.norm_attn(feat_attn), z)
        feat = feat + feat_attn.permute(0, 2, 1)
        feat = self.res2(feat)

        est_iq = self.iq_head(feat)
        logits = self.demap_mlp(feat)
        return logits, est_phi, est_cfo, est_iq

def build_model(cfg) -> HybridNeuralReceiverOFDM:
    """Construct the unchanged architecture for the configured frame lengths."""
    symbol_len = cfg.ofdm_fft_size + cfg.ofdm_cp_len
    return HybridNeuralReceiverOFDM(
        seq_len=cfg.ofdm_num_data_symbols * symbol_len,
        n_pilots=cfg.ofdm_num_pilot_symbols * symbol_len,
        ofdm_fft_size=cfg.ofdm_fft_size, ofdm_cp_len=cfg.ofdm_cp_len,
    )
