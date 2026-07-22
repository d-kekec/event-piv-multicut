#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model.py -- Mini-MLP fuer die Kantenkosten + die FEATURE-DEFINITION.

Architektur (BA_NN_Kantenkosten.md §3): winziges MLP n_in->hidden->1, Tanh,
Ausgabe ist ein LOGIT ohne Sigmoid. Zusammen mit BCEWithLogits lernt der
Logit die Log-Odds log(p/(1-p)); die Nullstelle liegt damit automatisch bei
p = 0.5 -- genau die Kalibrierung, die RAMA braucht
(BA_RAMA_Verstaendnis.md §1.4). Deshalb wird der Logit DIREKT als
Kantenkosten verwendet, ohne Nachskalierung.

WARUM DIE FEATURES HIER LIEGEN:
Bisher existierte die Feature-Vorschrift zweimal -- in train_edge_nn als
NumPy-Code, in solve_with_nn als Torch-Code auf der GPU. Zwei Implementierungen
derselben Formel, verbunden nur durch Sorgfalt. build_features() ist
backend-agnostisch (xp=np oder xp=torch), also gibt es jetzt genau eine.
Dasselbe Argument wie bei graph.neighborhood_pairs.

FEATURE-MODI (Ablation BA_NN_Kantenkosten.md §3.6):
  "dr"    (dt, dr)            AUC 0.792 -- gewaehlt, symmetrisch
  "absxy" (dt, |dx|, |dy|)    AUC 0.788 -- gleichauf, symmetrisch
  "raw6"  (ti,xi,yi,tj,xj,yj) AUC 0.29-0.39 -- VERWORFEN: memoriert
          Trainingspositionen (TNR ~0.02), bricht auf ungesehenen Runs ein.
          Vorschlag des Profs, empirisch widerlegt. Code bleibt fuer die
          Ablation erhalten, ist aber kein Arbeitspunkt.

SYMMETRIE c_ij == c_ji: bei dr/absxy durch die Features garantiert
(|dx|,|dy|,dr invariant unter i<->j; dt >= 0 durch den graph.py-Kontrakt).
raw6 ist NICHT symmetrisch -> Logits werden ueber den Tausch gemittelt.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

FEATURE_MODES = ("dr", "absxy", "raw6")

_FEATURE_NAMES = {
    "dr":    ("dt", "dr"),
    "absxy": ("dt", "|dx|", "|dy|"),
    "raw6":  ("ti", "xi", "yi", "tj", "xj", "yj"),
}

# Spaltenpermutation, die (i,j) -> (j,i) entspricht. Nur fuer raw6 noetig.
SWAP_RAW6 = [3, 4, 5, 0, 1, 2]


def check_mode(mode: str) -> str:
    if mode not in FEATURE_MODES:
        raise ValueError(f"Unbekannter feature_mode {mode!r}. "
                         f"Erlaubt: {FEATURE_MODES}")
    return mode


def feature_names(mode: str) -> list[str]:
    return list(_FEATURE_NAMES[check_mode(mode)])


def n_features(mode: str) -> int:
    return len(_FEATURE_NAMES[check_mode(mode)])


def is_symmetric(mode: str) -> bool:
    """False nur fuer raw6 -- dort muss ueber den Tausch gemittelt werden."""
    return check_mode(mode) != "raw6"


def build_features(dt, dx, dy, *, mode, ti=None, xi=None, yi=None, xp=np):
    """Baut die Feature-Matrix (n, n_in) aus den Kanten-Deltas.

    dt, dx, dy : 1D-Arrays/Tensoren, dt >= 0 (graph.py-Kontrakt)
    ti, xi, yi : Absolutkoordinaten des FRUEHEREN Events, nur fuer raw6
    xp         : Backend -- numpy (Training) oder torch (Einsatz auf GPU).
                 Beide bieten abs/sqrt/stack mit gleicher Signatur; die
                 Achse wird POSITIONAL uebergeben (np: axis, torch: dim).

    Rueckgabe: Matrix im dtype der Eingaben (Caller castet auf float32).
    """
    check_mode(mode)
    if mode == "dr":
        cols = [dt, xp.sqrt(dx * dx + dy * dy)]
    elif mode == "absxy":
        cols = [dt, xp.abs(dx), xp.abs(dy)]
    else:                                    # raw6
        if ti is None or xi is None or yi is None:
            raise ValueError("feature_mode 'raw6' braucht ti, xi, yi")
        cols = [ti, xi, yi, ti + dt, xi + dx, yi + dy]
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

    mu/sig kommen als Tensoren auf 'device' zurueck -- direkt einsetzbar.
    weights_only=False: die .pt ist unsere EIGENE Datei und enthaelt
    numpy-Arrays (mu/sig), die der sichere Default ab PyTorch 2.6 blockiert.
    """
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    model = EdgeMLP(ckpt["n_in"], ckpt["hidden"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mu = torch.as_tensor(ckpt["mu"], dtype=torch.float32, device=device)
    sig = torch.as_tensor(ckpt["sig"], dtype=torch.float32, device=device)
    return model, mu, sig, ckpt["feature_mode"], ckpt
