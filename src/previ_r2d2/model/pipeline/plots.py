"""Plots cross-architecture (comparent LGBM/BiLSTM/Stacking) -- port
fidèle de plot_test_comparison (train_meta.py, Previ_v2) + adaptations de
plot_kge_radar/plot_shap_bar (mlflow_tracker.py, Previ_v2) aux clés
réelles de kge_components/results. KGE-by-step et coefficients Ridge sont
des ajouts (pas dans Previ_v2), demandés en session."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_test_comparison(
    df_sub: pd.DataFrame,
    df_test: pd.DataFrame,
    yt_v: np.ndarray,
    pl_v: np.ndarray,
    pt_v: np.ndarray,
    stk_v: np.ndarray,
    valid: np.ndarray,
    y_test_v: np.ndarray,
    stk_all_v: np.ndarray,
    valid_pos: np.ndarray,
    horizon_steps: int,
    dossier: str,
    horizon: int,
    components: dict,
    output_path,
) -> None:
    """Plot LGBM/BiLSTM/Stacking vs observé sur le test set + trajectoires H-pas + résidus."""
    dates = pd.DatetimeIndex(df_sub.index)

    fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1], "hspace": 0.08})
    ax, ax_traj, ax_err = axes

    ax.plot(dates, yt_v, color="steelblue", linewidth=1.4, label="Observé (m³/s)", zorder=4)
    ax.plot(dates, pl_v, color="darkorange", linewidth=1.0, alpha=0.8, label=f"LightGBM  KGE={components['LightGBM seul']['kge']:.3f}", zorder=2)
    ax.plot(dates, pt_v, color="mediumpurple", linewidth=1.0, alpha=0.8, label=f"BiLSTM    KGE={components['BiLSTM seul']['kge']:.3f}", zorder=2)
    ax.plot(dates, stk_v, color="crimson", linewidth=1.6, label=f"Stacking  KGE={components['Stacking']['kge']:.3f}", zorder=3)

    precip_cols = [c for c in df_sub.columns if "precipitation" in c.lower() and "_s" in c.lower()]
    if precip_cols:
        ax2 = ax.twinx()
        for col in precip_cols:
            ax2.bar(dates, df_sub[col].values, width=0.04, alpha=0.18, color="blue")
        p_max = df_sub[precip_cols].max().max()
        ax2.set_ylim(max(p_max * 6, 1), 0)
        ax2.set_ylabel("Précipitations (mm)", color="blue", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="blue", labelsize=7)

    ax.set_ylabel("Débit (m³/s)")
    ax.set_title(
        f"Test hybride — {dossier} | h={horizon}h\n"
        f"Stacking r={components['Stacking']['r']:.3f}  α={components['Stacking']['alpha']:.3f}  β={components['Stacking']['beta']:.3f}"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    step = max(1, 24)
    legend_obs = legend_stk = False
    for i in range(0, len(valid_pos), step):
        vp = valid_pos[i]
        t_start = vp - (horizon_steps - 1)
        t_end = vp + 1
        if t_start < 0 or t_end > len(df_test):
            continue
        traj_dates = pd.DatetimeIndex(df_test.index[t_start:t_end])
        obs_traj = np.expm1(y_test_v[i])
        stk_traj = stk_all_v[i]

        lbl_obs = "Trajectoire observée" if not legend_obs else "_nolegend_"
        lbl_stk = f"Trajectoire stacking (t+1..t+{horizon_steps})" if not legend_stk else "_nolegend_"
        ax_traj.plot(traj_dates, obs_traj, color="steelblue", linewidth=0.8, alpha=0.35, label=lbl_obs)
        ax_traj.plot(traj_dates, stk_traj, color="crimson", linewidth=0.8, alpha=0.35, label=lbl_stk)
        legend_obs = legend_stk = True

    ax_traj.set_ylabel(f"Débit (m³/s)\n(trajectoires {horizon_steps} pas)", fontsize=9)
    ax_traj.legend(loc="upper left", fontsize=9)
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_ylim(bottom=0)

    residuals = np.where(valid[valid], stk_v - yt_v, np.nan)
    ax_err.bar(dates, residuals, color=np.where(np.nan_to_num(residuals) >= 0, "salmon", "skyblue"), width=0.04, alpha=0.7)
    ax_err.axhline(0, color="black", linewidth=0.8)
    ax_err.set_ylabel("Résidu stacking (m³/s)", fontsize=9)
    ax_err.grid(True, alpha=0.3)
    ax_err.xaxis.set_major_locator(mdates.MonthLocator())
    ax_err.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax_err.get_xticklabels(), rotation=45, fontsize=8)

    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_kge_by_step(results: dict, output_path) -> None:
    """Line chart KGE par pas d'horizon (LGBM/BiLSTM/Stacking)."""
    by_step = results["kge_by_step"]
    steps = range(1, len(by_step["stack"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, by_step["lgbm"], marker="o", label="LightGBM", color="darkorange")
    ax.plot(steps, by_step["lstm"], marker="o", label="BiLSTM", color="mediumpurple")
    ax.plot(steps, by_step["stack"], marker="o", label="Stacking", color="crimson", linewidth=2)
    ax.set_xlabel("Pas d'horizon (t+h)")
    ax.set_ylabel("KGE")
    ax.set_title(f"{results['centrale']} — KGE par pas d'horizon (h{results['horizon']})")
    ax.set_xticks(list(steps))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_ridge_coefficients(results: dict, output_path) -> None:
    """Bar chart des coefficients Ridge moyens (|coef|) par groupe de features."""
    ridge_coef = results.get("interpretability", {}).get("ridge_coef")
    if not ridge_coef:
        return
    groups = list(ridge_coef.keys())
    mean_abs = [ridge_coef[g]["mean_abs"] for g in groups]
    colors = ["seagreen" if ridge_coef[g]["sign"] == "+" else "firebrick" for g in groups]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(groups, mean_abs, color=colors)
    ax.set_ylabel("Coefficient Ridge moyen (|coef|)")
    ax.set_title(f"{results['centrale']} — Poids du meta-learner par groupe de features (h{results['horizon']})")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_kge_radar(components: dict, title: str, output_path) -> None:
    """Radar r (timing) / alpha (amplitude) / beta (volume) pour un modèle."""
    labels = ["r (timing)", "α (amplitude)", "β (volume)"]
    values = [components.get("r", 0), components.get("alpha", 0), components.get("beta", 0)]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    vp, ap = values + [values[0]], angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.plot(ap, vp, "o-", linewidth=2, color="#1f77b4")
    ax.fill(ap, vp, alpha=0.25, color="#1f77b4")
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.2)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title(f"KGE={components.get('kge', 0):.3f}\n{title}", fontsize=10, pad=15)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def generate_training_plots(
    results: dict,
    lgbm_result: dict,
    df_full_ctx: pd.DataFrame,
    exutoire: dict,
    bv_params: dict,
    steps_per_day: int,
    horizon_steps: int,
    transit_amont: dict,
    bilstm,
    X_seq_test: np.ndarray,
    y_test: np.ndarray,
    lstm_idx_test: np.ndarray,
    df_test: pd.DataFrame,
    seq_len: int,
    df_sub: pd.DataFrame,
    yt_v: np.ndarray,
    pl_v: np.ndarray,
    pt_v: np.ndarray,
    stk_v: np.ndarray,
    valid: np.ndarray,
    y_test_v: np.ndarray,
    stk_all_v: np.ndarray,
    valid_pos: np.ndarray,
    dossier: str,
    horizon: int,
    weights_dir,
    outputs_dir,
) -> None:
    """Génère les 7 plots d'entraînement (SHAP, attention, comparaison test, KGE-by-step, Ridge, KGE radar x3)."""
    from previ_r2d2.model.architectures.bilstm.explain import plot_attention_heatmaps
    from previ_r2d2.model.architectures.lightgbm.explain import build_shap_explanation, plot_shap_bar, plot_shap_beeswarm

    plots_dir = weights_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    shap_values, X_top = build_shap_explanation(lgbm_result, df_full_ctx, exutoire, bv_params, steps_per_day, horizon_steps, transit_amont)
    if shap_values is not None:
        plot_shap_beeswarm(shap_values, X_top, plots_dir / "shap_lgbm_beeswarm.png")
        plot_shap_bar(shap_values, X_top, plots_dir / "shap_lgbm_bar.png")

    plot_attention_heatmaps(bilstm, X_seq_test, y_test, lstm_idx_test, df_test, seq_len, plots_dir)

    plot_kge_by_step(results, plots_dir / "kge_by_step.png")
    plot_ridge_coefficients(results, plots_dir / "ridge_coefficients.png")

    for key, filename, title in [
        ("LightGBM seul", "kge_radar_lgbm.png", "LightGBM"),
        ("BiLSTM seul", "kge_radar_lstm.png", "BiLSTM"),
        ("Stacking", "kge_radar_stacking.png", "Stacking"),
    ]:
        plot_kge_radar(results["components"][key], title, plots_dir / filename)

    plot_test_comparison(
        df_sub, df_test, yt_v, pl_v, pt_v, stk_v, valid, y_test_v, stk_all_v, valid_pos,
        horizon_steps, dossier, horizon, results["components"],
        outputs_dir / f"hybrid_test_{dossier}_{horizon}h.png",
    )
