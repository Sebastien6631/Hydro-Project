"""Interprétabilité BiLSTM (attention) -- port fidèle de _explain_bilstm
(train_meta.py, Previ_v2), avec 2 corrections de fidélité : sélection des
pics de crue sur le dernier pas (l'original casse en multi-horizon,
np.argsort sur un array (N,H) au lieu de (N,)) et fenêtre de dates
indexée sur df_test (l'original indexe le df complet avant split,
donnant des labels d'axe X incorrects -- n'affecte pas les valeurs
d'attention, seulement leur légende)."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def plot_attention_heatmaps(
    bilstm,
    X_seq_test: np.ndarray,
    y_test: np.ndarray,
    lstm_idx_test: np.ndarray,
    df_test: pd.DataFrame,
    seq_len: int,
    output_dir,
    n_peaks: int = 5,
) -> list:
    """Heatmaps d'attention sur les n_peaks plus grosses crues du test set (dernier pas de y_test)."""
    y_t_last = y_test[:, -1] if y_test.ndim == 2 else y_test
    n_peaks = min(n_peaks, len(y_t_last))
    peak_idx = np.argsort(y_t_last)[-n_peaks:]

    n, s, f = X_seq_test.shape
    X_norm = bilstm.scalers[-1].transform(X_seq_test.reshape(-1, f)).reshape(n, s, f) if bilstm.scalers else X_seq_test
    attn = bilstm.get_attention_weights(torch.tensor(X_norm[peak_idx], dtype=torch.float32))

    paths = []
    for k, pi in enumerate(peak_idx):
        ts = df_test.index[max(0, lstm_idx_test[pi] - seq_len):lstm_idx_test[pi]]
        fig, ax = plt.subplots(figsize=(14, 2))
        ax.imshow(attn[k][np.newaxis, :], aspect="auto", cmap="YlOrRd", vmin=0)
        step = max(1, len(ts) // 12)
        ax.set_xticks(range(0, len(ts), step))
        ax.set_xticklabels([str(t)[-8:-3] for t in ts[::step]], rotation=45, fontsize=7)
        ax.set_yticks([])
        ax.set_title(f"Crue {k + 1} — q_obs={y_t_last[pi]:.1f} m³/s")
        plt.tight_layout()
        path = output_dir / f"attention_crue_{k + 1}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
