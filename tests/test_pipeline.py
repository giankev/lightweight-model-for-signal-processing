"""Run with: python3 -m unittest discover -s tests -v."""
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
DEPENDENCIES = ("numpy", "torch", "tensorflow", "sionna")
MISSING = [name for name in DEPENDENCIES if importlib.util.find_spec(name) is None]


@unittest.skipIf(MISSING, "Missing dependencies: " + ", ".join(MISSING))
class PipelineTests(unittest.TestCase):
    def test_true_parameters_recover_ofdm(self):
        """Nonzero signed CFO/phase catch units, sign, and pilot-offset errors."""
        import numpy as np
        import torch
        from lightweight_receiver.config import SimConfig
        from lightweight_receiver.models import compensate_data
        from lightweight_receiver.simulation import simulate_full_frames_ofdm_tf

        for phase, cfo in ((0.7, 0.0002), (-1.1, -0.0002)):
            with self.subTest(phase=phase, cfo=cfo):
                cfg = SimConfig(num_examples=3, phase_min=phase, phase_max=phase,
                                cfo_norm_min=cfo, cfo_norm_max=cfo,
                                ebn0_db_min=120.0, ebn0_db_max=120.0)
                full = simulate_full_frames_ofdm_tf(cfg)
                offset = full["meta"]["P"]
                received = full["y_rx"].numpy()[:, offset:]
                iq = torch.from_numpy(np.stack([received.real, received.imag], axis=1))
                corrected = compensate_data(
                    iq, torch.from_numpy(full["phi0"].numpy()[:, 0]),
                    torch.from_numpy(full["cfo_norm"].numpy()[:, 0]), offset,
                )
                np.testing.assert_allclose(corrected.numpy(), full["x_tx"].numpy()[:, offset:],
                                           atol=1e-5, rtol=1e-5)
                # Also verify CP removal and FFT alignment against the transmit grid.
                symbols = corrected.reshape(3, cfg.ofdm_num_data_symbols, 80)[:, :, 16:]
                grid = torch.fft.fftshift(torch.fft.fft(symbols, norm="ortho"), dim=-1)
                np.testing.assert_allclose(grid.numpy(), full["x_grid"].numpy()[:, 1:],
                                           atol=1e-5, rtol=1e-5)

    def test_loaders_reproduce_independent_of_global_rng(self):
        import torch
        from lightweight_receiver.config import SimConfig, training_seeds
        from lightweight_receiver.training import get_pytorch_ofdm_loaders

        cfg = SimConfig(num_examples=16)
        seeds = training_seeds(cfg)
        self.assertEqual(len(set(seeds.values())), len(seeds))
        first_train, first_val = get_pytorch_ofdm_loaders(cfg, batch_size=8)
        torch.manual_seed(98765)
        torch.rand(123)
        second_train, second_val = get_pytorch_ofdm_loaders(cfg, batch_size=8)
        self.assertEqual(first_train.dataset.indices, second_train.dataset.indices)
        self.assertEqual(first_val.dataset.indices, second_val.dataset.indices)
        self.assertFalse(set(first_train.dataset.indices) & set(first_val.dataset.indices))
        for first, second in zip(first_train, second_train):
            for actual, expected in zip(first, second):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_main_and_broad_streams_differ(self):
        from dataclasses import replace
        import numpy as np
        from lightweight_receiver.config import SimConfig, training_seeds
        from lightweight_receiver.simulation import generate_dataset_offline

        cfg = SimConfig(num_examples=4)
        seeds = training_seeds(cfg)
        main = generate_dataset_offline(replace(cfg, seed=seeds["main"],
                                               seed_noise=seeds["main_noise"]))
        broad = generate_dataset_offline(replace(cfg, seed=seeds["broad"],
                                                seed_noise=seeds["broad_noise"]))
        self.assertFalse(np.array_equal(main["info_bits"], broad["info_bits"]))
        self.assertFalse(np.array_equal(main["rx_iq_data_time"], broad["rx_iq_data_time"]))


if __name__ == "__main__":
    unittest.main()
