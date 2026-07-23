"""Modèle BiLSTM (meta) -- port fidèle de BiLSTMHydro (bilstm_hydro.py:184-417,
Previ_v2). Classe nn.Module (PyTorch l'exige), pas de fonctions pures ici --
contrairement à LightGBM. fit_oof reste une méthode : elle deep-copy self
par fold puis mute self en place à la fin (entraînement final), c'est
l'idiome PyTorch, pas un choix indépendant. self.scalers (pas self._scalers,
règle durable previ-R2-D2)."""

from __future__ import annotations

import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from previ_r2d2.model.architectures.bilstm.metrics import kge_loss_torch, kge_numpy

logger = logging.getLogger(__name__)


class BiLSTMHydro(nn.Module):
    """BiLSTM bidirectionnel avec mécanisme d'attention temporelle."""

    def __init__(
        self,
        n_features: int,
        horizon: int = 1,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.horizon = horizon
        self.scalers = []

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.attn_w = nn.Linear(hidden_size * 2, 1, bias=False)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, seq_len, n_features) -> (batch, horizon)."""
        out, _ = self.lstm(x)
        weights = F.softmax(self.attn_w(out), dim=1)
        context = (weights * out).sum(dim=1)
        out_h = self.head(context)
        if self.horizon == 1:
            return out_h.squeeze(-1)
        return out_h

    def get_attention_weights(self, x: torch.Tensor) -> np.ndarray:
        """Poids d'attention (batch, seq_len) sans gradient ; x doit déjà être normalisé via self.scalers[fold]."""
        self.eval()
        with torch.no_grad():
            out, _ = self.lstm(x)
            weights = F.softmax(self.attn_w(out), dim=1)
        return weights.squeeze(-1).cpu().numpy()

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        """Prédit en batch sur des séquences déjà construites, normalisées avec le dernier scaler OOF."""
        self.eval()
        n, s, f = X_seq.shape
        X_norm = self.scalers[-1].transform(X_seq.reshape(-1, f)).reshape(n, s, f) if self.scalers else X_seq
        with torch.no_grad():
            return self(torch.tensor(X_norm, dtype=torch.float32)).numpy()

    def fit_oof(
        self,
        X_seq: np.ndarray,
        y: np.ndarray,
        n_splits: int = 3,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 64,
        horizon: int = 7,
        dates=None,
    ) -> np.ndarray:
        """OOF (TimeSeriesSplit) puis réentraînement final 85/15 -- modifie self en place, retourne oof."""
        if y.ndim == 1:
            y = y[:, np.newaxis]
        n_steps = y.shape[1]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("BiLSTMHydro.fit_oof -- device=%s, folds=%d, epochs=%d, horizon=%d", device, n_splits, epochs, n_steps)

        oof = np.full((len(y), n_steps), np.nan, dtype=np.float32)
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=horizon)
        self.scalers = []
        PATIENCE = 10

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_seq)):
            if dates is not None:
                tr_d, val_d = dates[tr_idx], dates[val_idx]
                logger.info(
                    "BiLSTM OOF Fold %d/%d -- train : %s -> %s (%d obs) | val : %s -> %s (%d obs)",
                    fold + 1, n_splits, tr_d[0].date(), tr_d[-1].date(), len(tr_idx),
                    val_d[0].date(), val_d[-1].date(), len(val_idx),
                )
            X_tr, X_val = X_seq[tr_idx], X_seq[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            n_tr, s, f = X_tr.shape
            scaler = StandardScaler()
            X_tr_n = scaler.fit_transform(X_tr.reshape(-1, f)).reshape(n_tr, s, f)
            n_v = X_val.shape[0]
            X_val_n = scaler.transform(X_val.reshape(-1, f)).reshape(n_v, s, f)
            self.scalers.append(scaler)

            tr_ds = TensorDataset(torch.tensor(X_tr_n, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
            tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)

            model_fold = copy.deepcopy(self).to(device)
            optimizer = torch.optim.Adam(model_fold.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-5)

            best_kge = -np.inf
            best_state = None
            patience_cnt = 0

            for epoch in range(epochs):
                model_fold.train()
                for xb, yb in tr_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    kge_loss_torch(model_fold(xb), yb).backward()
                    nn.utils.clip_grad_norm_(model_fold.parameters(), 1.0)
                    optimizer.step()

                model_fold.eval()
                with torch.no_grad():
                    val_t = torch.tensor(X_val_n, dtype=torch.float32).to(device)
                    preds = model_fold(val_t).cpu().numpy()

                kge_steps = [kge_numpy(np.expm1(y_val[:, k]), np.expm1(preds[:, k])) for k in range(n_steps)]
                val_kge = float(np.mean(kge_steps))
                scheduler.step(-val_kge)
                logger.debug(
                    "Fold %d/%d | Epoch %d/%d | KGE_moy=%.4f [t+1=%.3f t+%d=%.3f]",
                    fold + 1, n_splits, epoch + 1, epochs, val_kge, kge_steps[0], n_steps, kge_steps[-1],
                )

                if val_kge > best_kge:
                    best_kge = val_kge
                    best_state = copy.deepcopy(model_fold.state_dict())
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= PATIENCE:
                        logger.debug("Early stopping fold %d epoch %d", fold + 1, epoch + 1)
                        break

            model_fold.load_state_dict(best_state)
            model_fold.eval()
            with torch.no_grad():
                val_t = torch.tensor(X_val_n, dtype=torch.float32).to(device)
                oof[val_idx] = model_fold(val_t).cpu().numpy()

            logger.info("BiLSTM Fold %d/%d -- best KGE_moy=%.4f", fold + 1, n_splits, best_kge)

        logger.info("BiLSTM final -- split 85/15, early stopping sur KGE_val (out-of-sample)")
        n_full, s, f = X_seq.shape
        val_size = max(int(n_full * 0.15), 200)
        n_tr_fin = n_full - val_size

        scaler_full = StandardScaler()
        X_full_n = scaler_full.fit_transform(X_seq.reshape(-1, f)).reshape(n_full, s, f)
        self.scalers.append(scaler_full)

        X_fin_tr, X_fin_val = X_full_n[:n_tr_fin], X_full_n[n_tr_fin:]
        y_fin_tr, y_fin_val = y[:n_tr_fin], y[n_tr_fin:]

        fin_tr_ds = TensorDataset(torch.tensor(X_fin_tr, dtype=torch.float32), torch.tensor(y_fin_tr, dtype=torch.float32))
        fin_tr_dl = DataLoader(fin_tr_ds, batch_size=batch_size, shuffle=True)
        X_fin_val_t = torch.tensor(X_fin_val, dtype=torch.float32)

        self.to(device)
        optimizer_full = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler_full = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_full, patience=5, factor=0.5, min_lr=1e-5)
        best_kge_final = -np.inf
        best_state_final = None
        patience_cnt_final = 0

        for epoch in range(epochs):
            self.train()
            for xb, yb in fin_tr_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer_full.zero_grad()
                kge_loss_torch(self(xb), yb).backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer_full.step()

            self.eval()
            with torch.no_grad():
                preds_val = self(X_fin_val_t.to(device)).cpu().numpy()
            kge_steps_val = [kge_numpy(np.expm1(y_fin_val[:, k]), np.expm1(preds_val[:, k])) for k in range(n_steps)]
            kge_val = float(np.mean(kge_steps_val))
            scheduler_full.step(-kge_val)
            logger.info(
                "Final | Epoch %d/%d | KGE_val=%.4f [t+1=%.3f t+%d=%.3f]",
                epoch + 1, epochs, kge_val, kge_steps_val[0], n_steps, kge_steps_val[-1],
            )

            if kge_val > best_kge_final:
                best_kge_final = kge_val
                best_state_final = copy.deepcopy(self.state_dict())
                patience_cnt_final = 0
            else:
                patience_cnt_final += 1
                if patience_cnt_final >= PATIENCE:
                    logger.info("Final early stopping epoch %d -- best KGE_val=%.4f", epoch + 1, best_kge_final)
                    break

        self.load_state_dict(best_state_final)
        self.eval()
        logger.info("BiLSTM final -- poids meilleurs chargés (KGE_val=%.4f)", best_kge_final)

        return oof
