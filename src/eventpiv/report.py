#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report.py -- Auswertung mehrerer Solve-Laeufe.

Zwei Gruppen, GETRENNT gemittelt:

  CROSSING/OVERLAP-Runs = Schnitt-Test (fordern das Trennen)
  CLEAN-Runs            = Kontrolle (duerfen NICHT fragmentieren)

Ein Mittelwert ueber beide koennte schlechte Trennung durch gute Kontrolle
kaschieren.

"Nicht loesbar" ist ein ERGEBNIS, kein Fehler: n_ok/n_total ist die Zahl,
an der sich zeigte, dass eine Einstellung als einzige alle Overlap-Runs
loeste. Eine fehlende solves/*.json bedeutet Absturz, VRAM oder kein
GT-Abgleich -- ein FEHLENDES VERZEICHNIS dagegen bedeutet, dass die
Overrides nicht passen, und bricht ab.
"""
from __future__ import annotations

import numpy as np
import json

from eventpiv import config
from eventpiv.runconfig import Selection

def summarize(name, infos, verbose=True) -> dict:
    """Mittelt ueber die loesbaren Runs einer Gruppe.

    UNGEWICHTETES Mittel ueber RUNS -- ein grosser Run zaehlt so viel wie ein
    kleiner. Dieselbe Weiche wie eine Ebene tiefer bei purity_cluster gegen
    purity_event, hier bewusst gleich entschieden: fuer den Vergleich von
    KONFIGURATIONEN ist das ungewichtete Mittel richtig, sonst dominieren
    grosse Runs die Aussage.
    """
    gueltig = [i for i in infos if i]
    leer = {"n_ok": 0, "n_total": len(infos),
            "purity_cluster": float("nan"), "purity_event": float("nan"),
            "frag": float("nan"), "frag_max": None}
    if not gueltig:
        if verbose:
            print(f"\n{name}: kein Run loesbar.")
        return leer
    z = {"n_ok": len(gueltig), "n_total": len(infos),
         "purity_cluster": float(np.mean([i["purity_cluster_mean"]
                                          for i in gueltig])),
         "purity_event": float(np.mean([i["purity_event_mean"]
                                        for i in gueltig])),
         "frag": float(np.mean([i["frag_mean"] for i in gueltig])),
         "frag_max": int(max(i["frag_max"] for i in gueltig))}
    if verbose:
        print(f"\n{name} ({z['n_ok']}/{z['n_total']} Runs gemittelt):")
        print(f"  Reinheit       cluster {z['purity_cluster']:.3f}, "
              f"event {z['purity_event']:.3f}")
        print(f"  Fragmentierung {z['frag']:.2f}  (max {z['frag_max']})")
    return z


def load_solves(cfg, runs) -> list:
    """Sammelt die solves/*.json der angegebenen Runs.

    None = kein Ergebnis, und das ist ein ERGEBNIS: eine fehlende Datei
    heisst Absturz, VRAM-OOM oder kein GT-Abgleich moeglich.
    """
    out = []
    if not runs:
        return []
    # Fehlt das VERZEICHNIS, ist die Konfiguration falsch, nicht das
    # Ergebnis. Ohne diese Pruefung meldet der Bericht "0/N loesbar" -- und
    # das sieht aus wie eine Messung, ist aber ein vergessener Override.
    # (create=False verhindert nur, dass das Verzeichnis dabei ENTSTEHT.)
    erster = config.solve_file(
        cfg.neighborhood, cfg.pairs, cfg.training,
        Selection(split=cfg.report.split, run_nr=runs[0]),
        cfg.window, cfg.solver, create=False)
    if not erster.parent.exists():
        raise FileNotFoundError(
            f"{erster.parent} existiert nicht -- zu dieser Konfiguration "
            "gibt es gar keine Solves. report_solves.py braucht DIESELBEN "
            "neighborhood/pairs/training-Overrides wie der Solve-Lauf.")
    for r in runs:
        p = config.solve_file(cfg.neighborhood, cfg.pairs, cfg.training,
                              Selection(split=cfg.report.split, run_nr=r),
                              cfg.window, cfg.solver, create=False)
        if not p.exists():
            out.append(None)
            continue
        info = json.loads(p.read_text())
        out.append(info if info.get("purity_cluster_mean") is not None
                   else None)
    return out


def report(cfg) -> dict:
    """Bewertung eines Netzes ueber CROSSING- und CLEAN-Runs."""
    rp, win = cfg.report, cfg.window
    print(f"Bewertung, Split '{rp.split}', Fenster "
          f"[{win.t0/1000:.0f}ms +{win.dt/1000:.0f}ms], Modus {cfg.solver.mode}")
    print(config.model_file(cfg.neighborhood, cfg.pairs, cfg.training,
                            create=False).parent)
    print("Reinheit unten als cluster/event gewichtet; Singl. = Cluster aus "
          "einem Event.")

    ergebnisse = {}
    for name, runs in (("CROSSING", rp.crossing), ("CLEAN", rp.clean)):
        print(f"\n{name}-Runs:")
        infos = load_solves(cfg, runs)
        for r, i in zip(runs, infos):
            if i:
                print(f"  run{r:<3d}: Rein {i['purity_cluster_mean']:.3f}/"
                      f"{i['purity_event_mean']:.3f}  "
                      f"Frag {i['frag_mean']:.2f} (max {i['frag_max']})  "
                      f"{i['n_clusters']} Cl / {i['n_gt']} GT, "
                      f"{i['n_singletons']} Singl., "
                      f"{i['n_edges']:,} Kanten")
            else:
                print(f"  run{r:<3d}: KEIN ERGEBNIS (Absturz / VRAM / kein GT)")
        ergebnisse[name.lower()] = summarize(name, infos)

    print("\nCROSSING = Schnitt-Test (fordert das Trennen), "
          "CLEAN = Kontrolle (darf nicht fragmentieren).")
    print("Metrik-Hierarchie: Fragmentierung > Reinheit >> AUC -- AUC steigt "
          "bei kleinem r_max truegerisch, waehrend Cluster zerfallen.")
    return ergebnisse