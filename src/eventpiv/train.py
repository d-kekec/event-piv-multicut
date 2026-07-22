#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py -- BAUSTEIN [3]: Training des Kantenkosten-Netzes.

p = sigmoid(logit), BCE gegen das Paar-Label "gleiches Partikel?".
Metrik ist AUC: rangbasiert, schwellenunabhaengig und bei starker Imbalance
ehrlich -- Accuracy waere bei 266:1 wertlos.

WICHTIG (BA_NN_Kantenkosten.md §3.6, Nachtrag 19.07.):
AUC bewertet die Trennschaerfe des NETZES und ist dafuer die richtige Metrik
(Feature-Ablation, Kapazitaetsdiagnose, Trainingsverlauf). Sie taugt aber
NICHT zur Wahl von alpha/r_max: kleinere Nachbarschaften machen die
Paar-Trennung trivial (AUC steigt), waehrend die Cluster-Fragmentierung
steigt. Fuer die Parameterwahl zaehlen Cluster-Metriken, nicht AUC.

Val-Split NACH RUNS, stratifiziert (beide Klassen in val, sonst NaN bei
AUC/TPR). Split innerhalb eines Runs waere Leakage: Paare desselben Runs sind
hochgradig abhaengig.
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn

from eventpiv import config
from eventpiv.model import (EdgeMLP, SWAP_RAW6, build_features, check_mode,
                            feature_names, is_symmetric, n_features,
                            save_checkpoint)

IMBALANCE_MODES = ("sample", "weight")


def roc_auc(y_true, y_score) -> float:
    """AUC ueber die Rangsumme (Mann-Whitney-U). Ohne sklearn-Abhaengigkeit."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) \
        / (len(pos) * len(neg))


def split_runs_stratified(label, run_arr, val_frac, rng):
    """Val-Runs so waehlen, dass BEIDE Klassen vertreten sind.

    Ein Run ist eine JA- oder eine NEIN-Quelle (so wurde gesammelt); aus
    jeder Gruppe wandert val_frac in die Validierung.
    """
    ist_ja = {r: (label[run_arr == r].mean() >= 0.5)
              for r in np.unique(run_arr)}
    ja = np.array([r for r, v in ist_ja.items() if v])
    nein = np.array([r for r, v in ist_ja.items() if not v])
    rng.shuffle(ja)
    rng.shuffle(nein)

    def nimm(runs):
        k = max(1, int(round(len(runs) * val_frac)))
        return set(runs[:k].tolist())

    val_runs = nimm(ja) | nimm(nein)
    return val_runs, np.array([r in val_runs for r in run_arr])


def evaluate(model, X, y, mu, sig, symmetric=True):
    """Rueckgabe: (acc, auc, tpr, tnr). Bei raw6 werden die Logits ueber
    den Tausch (i,j)->(j,i) gemittelt -- sonst waere c_ij != c_ji."""
    model.eval()
    with torch.no_grad():
        logit = model(X)
        if not symmetric:
            roh = X * sig + mu                      # zurueck in Rohgroessen
            getauscht = (roh[:, SWAP_RAW6] - mu) / sig
            logit = 0.5 * (logit + model(getauscht))
        p = torch.sigmoid(logit)
        pred = (p >= 0.5).float()
        acc = (pred == y).float().mean().item()
        auc = roc_auc(y.squeeze().numpy(), p.squeeze().numpy())
        tpr = (pred[y == 1] == 1).float().mean().item() \
            if (y == 1).any() else float("nan")
        tnr = (pred[y == 0] == 0).float().mean().item() \
            if (y == 0).any() else float("nan")
    return acc, auc, tpr, tnr


def train_edge_nn(data_path=None, *, feature_mode="dr", imbalance="sample",
                  sample_ratio=3.0, hidden=8, epochs=30, batch=65536,
                  lr=1e-2, val_frac=0.2, seed=0, out_dir=None, verbose=True):
    """Traeniert das Kanten-Netz und schreibt edge_nn_<mode>_<imbalance>.pt.

    Rueckgabe: (pfad, kennzahlen-dict).
    """
    check_mode(feature_mode)
    if imbalance not in IMBALANCE_MODES:
        raise ValueError(f"imbalance muss in {IMBALANCE_MODES} liegen")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    data_path = (config.pairs_dir(create=False) / "pairs_dataset.npz"
                 if data_path is None else data_path)
    d = np.load(data_path, allow_pickle=False)

    dt = d["dt"].astype(np.float32)
    dx = d["dx"].astype(np.float32)
    dy = d["dy"].astype(np.float32)
    label = d["label"].astype(np.float32)
    run_arr = d["run"]

    # alpha/r_max wandern aus dem Datensatz in die .pt -- damit kann solve
    # spaeter HART pruefen, ob Netz und Einsatzparameter zusammenpassen.
    # Bisher war das eine Konvention (Orchestrator reicht beides durch).
    alpha = float(d["alpha"]) if "alpha" in d else None
    r_max = float(d["r_max"]) if "r_max" in d else None

    if verbose:
        print(f"Datensatz: {data_path}")
        print(f"  {len(label):,} Paare "
              f"(JA {int(label.sum()):,} / NEIN {int((label == 0).sum()):,})")
        print(f"  alpha={alpha}, r_max={r_max}")

    kwargs = {}
    if feature_mode == "raw6":
        kwargs = {"ti": d["ti"].astype(np.float32),
                  "xi": d["xi"].astype(np.float32),
                  "yi": d["yi"].astype(np.float32)}
    feats = build_features(dt, dx, dy, mode=feature_mode, xp=np,
                           **kwargs).astype(np.float32)
    sym = is_symmetric(feature_mode)
    if verbose:
        print(f"Feature-Modus '{feature_mode}': {feature_names(feature_mode)} "
              f"({'symmetrisch' if sym else 'NICHT symm. -> Mittelung'})")

    val_runs, is_val = split_runs_stratified(label, run_arr, val_frac, rng)
    tr_idx = np.flatnonzero(~is_val)
    val_idx = np.flatnonzero(is_val)
    if verbose:
        vlab = label[is_val]
        print(f"Holdout-Runs (aus demselben Split wie der Paar-Datensatz): "
              f"{sorted(val_runs)}"
              f"(train {len(tr_idx):,} / val {len(val_idx):,} Paare)")
        print(f"  Val-Klassen: JA {int((vlab == 1).sum()):,} / "
              f"NEIN {int((vlab == 0).sum()):,}")

    # --- Imbalance: Sampling ODER Gewichtung, nie beides ---
    pos_weight = None
    if imbalance == "sample":
        tr_lab = label[tr_idx]
        ja_i = tr_idx[tr_lab == 1]
        nein_i = tr_idx[tr_lab == 0]
        n_keep = min(len(ja_i), int(len(nein_i) * sample_ratio))
        tr_idx = np.concatenate([rng.choice(ja_i, n_keep, replace=False),
                                 nein_i])
        rng.shuffle(tr_idx)
        if verbose:
            print(f"Sampling: JA {n_keep:,} : NEIN {len(nein_i):,} "
                  f"(~{n_keep / max(len(nein_i), 1):.1f}:1)")
    else:
        tr_lab = label[tr_idx]
        n_pos = (tr_lab == 1).sum()
        n_neg = (tr_lab == 0).sum()
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        if verbose:
            print(f"Gewichtung: pos_weight = {pos_weight.item():.5f}")

    # Standardisierung NUR aus den Trainingsdaten (sonst Leakage)
    mu = feats[tr_idx].mean(axis=0)
    sig = feats[tr_idx].std(axis=0) + 1e-8

    def std(a):
        return (a - mu) / sig

    Xtr = torch.tensor(std(feats[tr_idx]))
    ytr = torch.tensor(label[tr_idx]).unsqueeze(1)
    Xva = torch.tensor(std(feats[val_idx]))
    yva = torch.tensor(label[val_idx]).unsqueeze(1)
    mu_t = torch.tensor(mu)
    sig_t = torch.tensor(sig)

    model = EdgeMLP(n_features(feature_mode), hidden)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Netz: {n_features(feature_mode)}->{hidden}->1, "
              f"{n_params} Parameter")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n = len(Xtr)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for b in range(0, n, batch):
            bi = perm[b:b + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[bi]), ytr[bi])
            loss.backward()
            opt.step()
            tot += loss.item() * len(bi)
        if verbose and (ep % 5 == 0 or ep == 1):
            acc, auc, tpr, tnr = evaluate(model, Xva, yva, mu_t, sig_t, sym)
            print(f"  ep {ep:3d}  loss {tot/n:.4f} | val acc {acc:.4f} "
                  f"AUC {auc:.4f} TPR(JA) {tpr:.3f} TNR(NEIN) {tnr:.3f}")

    acc, auc, tpr, tnr = evaluate(model, Xva, yva, mu_t, sig_t, sym)
    kennzahlen = {"acc": acc, "auc": auc, "tpr": tpr, "tnr": tnr,
                  "n_params": n_params, "val_runs": sorted(val_runs)}
    if verbose:
        print(f"\nFinal: acc {acc:.4f} | AUC {auc:.4f} | "
              f"TPR(JA) {tpr:.3f} | TNR(NEIN) {tnr:.3f}")

    out_dir = config.model_dir() if out_dir is None else out_dir
    tag = f"{feature_mode}_{imbalance}"
    path = save_checkpoint(
        f"{out_dir}/edge_nn_{tag}.pt", model, mu, sig, feature_mode, hidden,
        **json.loads(config.meta_json(
            stage="edge_nn", tag=tag, alpha=alpha, r_max=r_max,
            imbalance=imbalance, sample_ratio=sample_ratio, epochs=epochs,
            lr=lr, batch=batch, val_frac=val_frac, seed=seed,
            data_path=str(data_path), **kennzahlen)))
    if verbose:
        print(f"-> {path}")
    return path, kennzahlen
