#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report.py -- Auswertung mehrerer Solve-Laeufe.

Uebernommen aus pipeline.py, das mit dem Hydra-Umbau entfallen ist. Die
Orchestrierung dort war Geruest (Subprozess-Kommandozeilen, Parameter
durchreichen) -- DIES hier ist Fachlogik und bleibt:

  CROSSING/OVERLAP-Runs = Schnitt-Test (fordern das Trennen)
  CLEAN-Runs            = Kontrolle (duerfen NICHT fragmentieren)
  Beide Gruppen GETRENNT mitteln -- ein Mittelwert ueber beide koennte
  schlechte Trennung durch gute Kontrolle kaschieren.

"Nicht loesbar" ist ein ERGEBNIS, kein Fehler: n_ok/n_total sind die Spalten
OV_ok/CL_ok aus BA_NN_Kantenkosten.md §3.8 ("alpha=3000/r_max=5 loest als
EINZIGE alle Overlap-Runs"). Frueher entstand das aus abgestuerzten
Subprozessen; jetzt aus fehlenden solves/*.json.
"""
from __future__ import annotations

import numpy as np


def summarize(name, infos, verbose=True) -> dict:
    """Mittelt ueber die loesbaren Runs einer Gruppe."""
    gueltig = [i for i in infos if i]
    if not gueltig:
        if verbose:
            print(f"\n{name}: kein Run loesbar.")
        return {"n_ok": 0, "n_total": len(infos), "purity": float("nan"),
                "frag": float("nan"), "frag_max": None}
    z = {"n_ok": len(gueltig), "n_total": len(infos),
         "purity": float(np.mean([i["purity_mean"] for i in gueltig])),
         "frag": float(np.mean([i["frag_mean"] for i in gueltig])),
         "frag_max": int(max(i["frag_max"] for i in gueltig))}
    if verbose:
        print(f"\n{name} ({z['n_ok']}/{z['n_total']} Runs gemittelt):")
        print(f"  Reinheit       {z['purity']:.3f}")
        print(f"  Fragmentierung {z['frag']:.2f}  (max {z['frag_max']})")
    return z