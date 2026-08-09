#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
overlap.py -- Runs nach Partikel-Ueberlappung klassifizieren.

Grobe VORAUSWAHL, wenn es weder Labels noch ein Netz noch Solves gibt --
etwa beim Erschliessen eines neuen Datensatzes. Braucht nur die
Bounding-Box-Zentren.

KEINE Bewertung. Wie gut Stufe 1 trennt, sagt der Treiber ueber
lauf/nah_merge_event_frac -- getrennt je PARTIKEL und je Fenster, nicht als
Aggregat ueber einen ganzen Run.

MASS (je 60-Hz-Rasterzeitpunkt, dann ueber den Run aggregiert):
  min_dist  kleinster je erreichter Zentrumsabstand
  frac11    Anteil Zeitpunkte mit minimalem Abstand < box_w
  mean11    mittlere Zahl PAARE unter box_w je Zeitpunkt

mean11 zaehlt PAARE, nicht Boxen -- drei sich beruehrende Boxen ergeben 3.
Die Schwelle 1.4 ist empirisch kalibriert, nicht hergeleitet.

KATEGORIEN:
  MAX_OVERLAP  frac11 == 1.0                         durchgehende Ueberlappung
  MIN_OVERLAP  min_dist >= box_w UND mean11 < 0.05   praktisch sauber
  CROSSING     frac11 == 1.0 UND mean11 >= 1.4       echte Kreuzungen

GRENZE, gemessen: alle drei Masse sind Aggregate ueber den GANZEN Run und
sagen nicht, WANN eine Ueberlappung passiert. Die Einteilung traegt
entsprechend grob -- run11 steht als Overlap-Run in der Liste und hat 531
von 798 Tracklet-Fenstern ohne eine einzige NEIN-Kante. Wer die Zeitachse
braucht, nimmt fenster_n_nein aus dem Tracklet-Artefakt.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from eventpiv.runconfig import Overlap, OverlapCfg


def overlap_metrics(bbox, *, box_w) -> dict:
    """Kennzahlen fuer EINEN Run. Reine Funktion -- bbox rein, dict raus."""
    raster_t = np.unique(bbox["t"])
    bx = bbox["x"].astype(np.float64) + box_w / 2
    by = bbox["y"].astype(np.float64) + box_w / 2

    min_dist = np.inf
    n_hits = 0                 # Zeitpunkte mit minimalem Paarabstand < box_w
    paare_kumuliert = 0
    n_t = len(raster_t)

    for rt in raster_t:
        m = bbox["t"] == rt
        cx, cy = bx[m], by[m]
        if len(cx) < 2:
            continue
        pts = np.column_stack([cx, cy])
        tree = cKDTree(pts)
        d, _ = tree.query(pts, k=2)          # k=2: sich selbst + naechster
        this_min = d[:, 1].min()
        min_dist = min(min_dist, this_min)
        if this_min < box_w:
            n_hits += 1
        paare_kumuliert += len(tree.query_pairs(r=box_w))

    return {"tracks": int(len(np.unique(bbox["track_id"]))),
            "n_raster": int(n_t),
            "min_dist": float(min_dist),
            "frac11": n_hits / max(n_t, 1),
            "mean11": paare_kumuliert / max(n_t, 1)}


def classify(results: dict, ov: Overlap) -> dict:
    """Teilt {run_nr: metriken} in die drei Kategorien auf."""
    return {
        "MAX_OVERLAP": sorted(r for r, i in results.items()
                              if i["frac11"] >= 0.999),
        "MIN_OVERLAP": sorted(r for r, i in results.items()
                              if i["min_dist"] >= ov.box_w
                              and i["mean11"] < ov.mean_clean_max),
        "CROSSING": sorted(r for r, i in results.items()
                           if i["frac11"] >= 0.999
                           and i["mean11"] >= ov.cross_thresh),
    }


def analyze_split(cfg: OverlapCfg, verbose=True):
    """Analysiert alle (oder die angegebenen) Runs eines Splits.

    Leere Run-Liste heisst ALLE Runs des Splits, wie bei labels.build_labels.
    Rueckgabe: (results, categories). Schreibt nichts.
    """
    from eventpiv import config
    from eventpiv.data import load_bbox

    split, ov = cfg.selection.split, cfg.overlap
    runs = list(cfg.selection.runs) or config.available_runs(split)
    if verbose:
        print(f"Split '{split}': {len(runs)} Runs analysieren ...\n")

    results = {}
    for r in runs:
        try:
            info = overlap_metrics(load_bbox(split, r), box_w=ov.box_w)
        except FileNotFoundError:
            if verbose:
                print(f"  run{r}: kein _bbox.npy, uebersprungen")
            continue
        results[r] = info
        if verbose:
            md = "inf " if np.isinf(info["min_dist"]) else f"{info['min_dist']:4.1f}"
            print(f"  run{r:<3d}: tracks={info['tracks']:3d}  min_dist={md}px  "
                  f"frac<{ov.box_w:.0f}={info['frac11']:.3f}  "
                  f"mean<{ov.box_w:.0f}={info['mean11']:.2f}")

    cats = classify(results, ov)
    if verbose:
        _print_lists(split, cats, ov.cross_thresh)
    return results, cats


def _print_lists(split, cats, cross_thresh):
    def als_liste(xs):
        return "[" + ", ".join(str(x) for x in xs) + "]"

    print("\n" + "=" * 56)
    print(f"KOPIERFERTIGE LISTEN  (Split '{split}')")
    print("=" * 56)
    print("\n# Durchgehende Ueberlappung -- NEIN-Quelle:")
    print(f"MAX_OVERLAP = {als_liste(cats['MAX_OVERLAP'])}")
    print("\n# Praktisch sauber (Boxen beruehren sich nie, keine Kreuzungen)"
          " -- JA-Quelle / Kontrolle:")
    print(f"MIN_OVERLAP = {als_liste(cats['MIN_OVERLAP'])}")
    print(f"\n# Echte Kreuzungen (mean >= {cross_thresh}) -- beste NEIN-Quelle:")
    print(f"CROSSING    = {als_liste(cats['CROSSING'])}")
    print()
