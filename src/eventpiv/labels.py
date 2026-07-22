#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
labels.py -- BAUSTEIN [1]: Event -> track_id + Qualitaetsstatus.

Weist jedem Event eine track_id zu (TRAININGS-WAHRHEIT, NIE NN-Input!) und
markiert dabei drei Qualitaetsklassen:

    status = 0  SAUBER      eindeutig einem Partikel zugeordnet   -> Training
    status = 1  MEHRDEUTIG  fast gleich weit zu zwei Boxen        -> verwerfen
    status = 2  RAUSCH      keiner Box nah genug                  -> raushalten

Grundlage: validierter GT-Abgleich (BA_Datensatz_und_Pipeline.md §2.2):
  Event -> zeitlich naechster 60-Hz-Rasterpunkt -> raeumlich naechstes
  Boxzentrum. ERWEITERT um die relative (2.-naechste) Pruefung, weil
  min_dist>=11px auf RUN-Ebene einzelne Grenz-Events INNERHALB eines Runs
  nicht ausschliesst (Eventwolken verschmelzen erst ~19px, BA_NN §3.5).

Ausgabe haengt NICHT von alpha/r_max/WIN_DT ab -> einmal pro Run/Split bauen.

Aufteilung in dieser Datei:
  assign_track_ids()      reiner Rechenkern (Arrays rein, Arrays raus)
  build_labels_for_run()  laedt, rechnet, schreibt EINEN Run
  build_labels()          Schleife ueber mehrere Runs, Fehlerbehandlung
"""
from __future__ import annotations

import gc
import time

import numpy as np
from scipy.spatial import cKDTree

from eventpiv import config
from eventpiv.data import load_bbox, load_events

from pathlib import Path

# --- Statuscodes (statt magischer Zahlen im Code) ---
STATUS_CLEAN = 0
STATUS_AMBIG = 1
STATUS_NOISE = 2

# --- Defaults (Herkunft: run3-Abgleich, BA_Datensatz §2.2) ---
BOX_W_DEFAULT    = 11.0   # px, Boxbreite = Hoehe, Konstante des Datensatzes
R_ASSIGN_DEFAULT = 8.0    # px, max. Abstand Event <-> naechstes Zentrum
R_AMBIG_DEFAULT  = 2.0    # px, (d2-d1) darunter -> mehrdeutig


def assign_track_ids(t, x, y, bbox,
                     r_assign=R_ASSIGN_DEFAULT,
                     r_ambig=R_AMBIG_DEFAULT,
                     box_w=BOX_W_DEFAULT):
    """Ordnet jedem Event (t,x,y) eine track_id + Status zu.

    Reine Funktion: keine Pfade, keine Globals, kein I/O -> ohne Datensatz
    testbar (tests/test_labels.py).

    Vorgehen:
      1) Jedes Event dem zeitlich naechsten 60-Hz-Rasterpunkt zuordnen.
      2) Pro Rasterpunkt KDTree ueber die Boxzentren zu DIESEM Zeitpunkt,
         Events gegen die 2 naechsten Zentren abfragen.
      3) Status aus (d1, d2):
           d1 > r_assign        -> RAUSCH (2),      track_id = -1
           (d2 - d1) < r_ambig  -> MEHRDEUTIG (1),  track_id = -1
           sonst                -> SAUBER (0),      track_id des naechsten

    Rueckgabe: (track_id int32, status uint8), beide der Laenge len(t).
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n = len(t)
    track_id = np.full(n, -1, dtype=np.int32)
    status   = np.full(n, STATUS_NOISE, dtype=np.uint8)  # Default, wird ueberschrieben
    if n == 0:
        return track_id, status

    center_offset = box_w / 2.0          # (x,y)=Ecke oben links -> +5.5 = Zentrum

    # --- 60-Hz-Raster aus den bbox-Zeitstempeln ---
    raster_t = np.unique(bbox["t"])
    # Jedem Event den INDEX des zeitlich naechsten Rasterpunkts zuordnen:
    # searchsorted gibt die Einfuegeposition, wir vergleichen linken/rechten Nachbarn.
    if len(raster_t) == 1:
        # Sonderfall: alle Events gehoeren zum einzigen Rasterpunkt.
        # Muss explizit behandelt werden -- die searchsorted-Logik unten
        # setzt mindestens zwei Rasterpunkte voraus (sonst greift der clip
        # ins Leere und ALLE Events fielen still auf RAUSCH zurueck).
        nearest_idx = np.zeros(len(t), dtype=np.intp)
    else:
        pos = np.searchsorted(raster_t, t)
        pos = np.clip(pos, 1, len(raster_t) - 1)
        left, right = raster_t[pos - 1], raster_t[pos]
        nearest_idx = np.where(np.abs(t - left) <= np.abs(t - right),
                               pos - 1, pos)

    # --- Boxzentren vorbereiten (Ecke -> Zentrum) ---
    bx = bbox["x"].astype(np.float64) + center_offset
    by = bbox["y"].astype(np.float64) + center_offset

    # --- Events nach Rasterpunkt gruppieren (ein Durchgang je Rasterpunkt) ---
    order = np.argsort(nearest_idx, kind="stable")
    grp = nearest_idx[order]
    grp_start = np.searchsorted(grp, np.arange(len(raster_t)), side="left")
    grp_end   = np.searchsorted(grp, np.arange(len(raster_t)), side="right")

    for ri in range(len(raster_t)):
        sl = order[grp_start[ri]:grp_end[ri]]
        if len(sl) == 0:
            continue
        mask_box = bbox["t"] == raster_t[ri]
        if not mask_box.any():
            continue

        centers = np.column_stack([bx[mask_box], by[mask_box]])
        tids    = bbox["track_id"][mask_box].astype(np.int32)

        tree = cKDTree(centers)
        k = min(2, len(centers))          # bei nur EINEM Partikel gibt es kein d2
        d, idx = tree.query(np.column_stack([x[sl], y[sl]]), k=k)

        if k == 1:
            d1, i1 = d, idx
            d2 = np.full_like(d1, np.inf)  # kein zweiter -> nie mehrdeutig
        else:
            d1, d2 = d[:, 0], d[:, 1]
            i1 = idx[:, 0]

        is_noise = d1 > r_assign
        is_ambig = (~is_noise) & ((d2 - d1) < r_ambig)
        is_clean = (~is_noise) & (~is_ambig)

        status[sl[is_noise]] = STATUS_NOISE
        status[sl[is_ambig]] = STATUS_AMBIG
        status[sl[is_clean]] = STATUS_CLEAN
        track_id[sl[is_clean]] = tids[i1[is_clean]]

    return track_id, status


def label_stats(status) -> dict:
    """Kennzahlen fuer Ausgabe und Tests."""
    n = len(status)
    return {
        "n": n,
        "n_clean": int((status == STATUS_CLEAN).sum()),
        "n_ambig": int((status == STATUS_AMBIG).sum()),
        "n_noise": int((status == STATUS_NOISE).sum()),
    }


def build_labels_for_run(split, run_nr, out_dir=None, t0=None, dt=None,
                         r_assign=R_ASSIGN_DEFAULT, r_ambig=R_AMBIG_DEFAULT,
                         box_w=BOX_W_DEFAULT, overwrite=False, verbose=True):
    """Laedt einen Run, labelt ihn, schreibt <run>_eventlabels.npz.

    Rueckgabe: (Pfad, stats-dict). Wenn die Datei existiert und
    overwrite=False, wird nur der Pfad zurueckgegeben (stats=None).
    """
    out_dir = config.label_dir(split) if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"run{run_nr}_eventlabels.npz"

    if out_path.exists() and not overwrite:
        if verbose:
            print(f"  run{run_nr}: existiert schon -> uebersprungen")
        return out_path, None

    bbox = load_bbox(split, run_nr)
    ev = load_events(split, run_nr, t0=t0, dt=dt)

    tic = time.time()
    track_id, status = assign_track_ids(ev["t"], ev["x"], ev["y"], bbox,
                                        r_assign=r_assign, r_ambig=r_ambig,
                                        box_w=box_w)
    elapsed = time.time() - tic

    st = label_stats(status)
    if verbose:
        n = max(st["n"], 1)
        print(f"run{run_nr}: {st['n']:,} Events in {elapsed:.1f}s")
        print(f"  SAUBER      {st['n_clean']:9,}  ({st['n_clean']/n*100:5.2f}%)")
        print(f"  MEHRDEUTIG  {st['n_ambig']:9,}  ({st['n_ambig']/n*100:5.2f}%)")
        print(f"  RAUSCH      {st['n_noise']:9,}  ({st['n_noise']/n*100:5.2f}%)")
        print(f"  track_ids (sauber): "
              f"{len(np.unique(track_id[status == STATUS_CLEAN]))}")

    # SCHLANK speichern -- verlustfreie Integer-Casts (Rohdaten sind ganzzahlig):
    #   t: µs, max ~8e6 << 2^32 -> uint32 | x,y: 0..1279 -> uint16 | p: uint8
    # t,x,y BLEIBEN erhalten (Interpretationsgrundlage), nur kompakter.
    np.savez(out_path,
             t=ev["t"].astype(np.uint32),
             x=ev["x"].astype(np.uint16),
             y=ev["y"].astype(np.uint16),
             p=ev["p"].astype(np.uint8),
             track_id=track_id.astype(np.int16),
             status=status.astype(np.uint8),
             meta=config.meta_json(stage="event_labels", split=split,
                                   run=run_nr, r_assign=r_assign,
                                   r_ambig=r_ambig, box_w=box_w,
                                   t0=t0, dt=dt, **st))

    del ev, track_id, status
    gc.collect()                      # RAM-Druck bei Batch-Laeufen daempfen
    if verbose:
        print(f"  -> {out_path.name}\n")
    return out_path, st


def build_labels(runs, split="train", **kwargs):
    """Mehrere Runs. Rueckgabe: Liste der fehlgeschlagenen Run-Nummern."""
    print(f"Event-Label-Erzeugung, Split '{split}', {len(runs)} Run(s), "
          f"R_ASSIGN={kwargs.get('r_assign', R_ASSIGN_DEFAULT)}px, "
          f"R_AMBIG={kwargs.get('r_ambig', R_AMBIG_DEFAULT)}px\n")
    failed = []
    for r in runs:
        try:
            build_labels_for_run(split, r, **kwargs)
        except Exception as e:
            print(f"  !! run{r} uebersprungen: {type(e).__name__}: {e}")
            failed.append(r)
    if failed:
        print(f"\nFehlgeschlagene Runs ({len(failed)}): {failed}")
    return failed
