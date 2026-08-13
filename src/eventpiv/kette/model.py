#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model.py -- Mini-MLP fuer die Kantenkosten + die FEATURE-DEFINITION.

Winziges MLP n_in->hidden->1, Tanh, Ausgabe ist ein LOGIT ohne Sigmoid.
Zusammen mit BCEWithLogits lernt der Logit die Log-Odds log(p/(1-p)); die
Nullstelle liegt damit automatisch bei p = 0.5 -- genau die Kalibrierung,
die RAMA braucht. Deshalb geht der Logit DIREKT als Kantenkosten weiter,
ohne Nachskalierung. Daran haengt auch, warum die Schwelle 0 und keine
andere der Arbeitspunkt ist (metrics.py).

WARUM DIE FEATURES HIER LIEGEN: die Vorschrift existierte zweimal -- als
NumPy-Code im Training, als Torch-Code auf der GPU. Zwei Implementierungen
derselben Formel, verbunden nur durch Sorgfalt. build_features() ist
backend-agnostisch (xp=np oder xp=torch), also gibt es jetzt genau eine.
Dasselbe Argument wie bei graph.neighborhood_pairs.

FEATURE-MODI, gemessen:
  "dr"    (dt, dr)            AUC 0.792 -- Arbeitspunkt
  "absxy" (dt, |dx|, |dy|)    AUC 0.788 -- gleichauf

SYMMETRIE c_ij == c_ji ist eine Eigenschaft der MERKMALE, kein Laufzeit-
thema: |dx|, |dy| und dr sind invariant unter i<->j, und dt >= 0
garantiert der graph.py-Kontrakt. Multicut setzt diese Symmetrie voraus --
eine Kante hat EINEN Preis, unabhaengig davon, welches Ende man zuerst
nennt.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from eventpiv.kern.runconfig import FEATURE_MODES

_FEATURE_NAMES = {
    "dr":    ("dt", "dr"),
    "absxy": ("dt", "|dx|", "|dy|"),
}


def check_mode(mode: str) -> str:
    if mode not in FEATURE_MODES:
        raise ValueError(f"Unbekannter feature_mode {mode!r}. "
                         f"Erlaubt: {FEATURE_MODES}")
    return mode


def feature_names(mode: str) -> list[str]:
    return list(_FEATURE_NAMES[check_mode(mode)])


def n_features(mode: str) -> int:
    return len(_FEATURE_NAMES[check_mode(mode)])


def build_features(dt, dx, dy, *, mode, xp=np):
    """Baut die Feature-Matrix (n, n_in) aus den Kanten-Deltas.

    dt, dx, dy : 1D-Arrays/Tensoren, dt >= 0 (graph.py-Kontrakt)
    xp         : Backend -- numpy (Training) oder torch (Einsatz auf GPU).

    Rueckgabe: Matrix im dtype der Eingaben (Caller castet auf float32).
    """
    check_mode(mode)
    if mode == "dr":
        cols = [dt, xp.sqrt(dx * dx + dy * dy)]
    else:                                    # absxy
        cols = [dt, xp.abs(dx), xp.abs(dy)]
    return xp.stack(cols, 1)


class EdgeMLP(nn.Module):
    """n_in -> hidden -> 1, Tanh. Gibt einen LOGIT zurueck (kein Sigmoid!)."""

    def __init__(self, n_in: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x)


def save_checkpoint(path, model, mu, sig, feature_mode, hidden, **meta):
    """Speichert Netz + Standardisierung + Metadaten in EINER Datei.

    mu/sig gehoeren zwingend dazu: das Netz sieht standardisierte Features,
    also muss der Einsatz exakt dieselbe Transformation anwenden.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in
                               model.state_dict().items()},
                "mu": np.asarray(mu), "sig": np.asarray(sig),
                "feature_mode": feature_mode, "hidden": hidden,
                "n_in": n_features(feature_mode), "meta": meta}, path)
    return path


def load_checkpoint(path, device="cpu"):
    """Laedt ein Netz. Rueckgabe: (model, mu, sig, feature_mode, ckpt).

    mu/sig kommen als Tensoren auf 'device' zurueck, direkt einsetzbar.
    weights_only=False: die .pt ist unsere eigene Datei und enthaelt
    numpy-Arrays (mu/sig), die der sichere Default ab PyTorch 2.6 blockiert.
    """
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    model = EdgeMLP(ckpt["n_in"], ckpt["hidden"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mu = torch.as_tensor(ckpt["mu"], dtype=torch.float32, device=device)
    sig = torch.as_tensor(ckpt["sig"], dtype=torch.float32, device=device)
    return model, mu, sig, ckpt["feature_mode"], ckpt

