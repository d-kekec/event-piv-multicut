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
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from eventpiv import config
from eventpiv.data import load_bbox, load_events
from eventpiv.runconfig import LabelCfg, Labeling, identity_meta

# --- Statuscodes (statt magischer Zahlen im Code) ---
STATUS_CLEAN = 0
STATUS_AMBIG = 1
STATUS_NOISE = 2

# --- Defaults: EINE Quelle, runconfig.Labeling. Die Modulkonstanten bleiben
# als Namen bestehen (tests/test_labels.py benutzt sie), beziehen ihren Wert
# aber von dort -- sonst gaebe es zwei Wahrheiten ueber dieselbe Zahl.
_LB_DEFAULT      = Labeling()
BOX_W_DEFAULT    = _LB_DEFAULT.box_w      # px, Boxbreite = Hoehe (Datensatz-Konstante)
R_ASSIGN_DEFAULT = _LB_DEFAULT.r_assign   # px, max. Abstand Event <-> naechstes Zentrum
R_AMBIG_DEFAULT  = _LB_DEFAULT.r_ambig    # px, (d2-d1) darunter -> mehrdeutig


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


def _labeling_passt(path, lb) -> bool:
    """Wurde die vorhandene Datei mit GENAU diesen Labeling-Parametern gebaut?

    Label-Dateien tragen bewusst keinen Slug -- sie haengen von nichts ab
    ausser Labeling. Der Preis: das Dateisystem kann nicht ausdruecken, mit
    welchen Parametern eine Datei entstand. Die Antwort steht statt dessen im
    meta-Block der .npz und wird hier gelesen.

    Fehlt oder bricht der Block, gilt die Datei als unbrauchbar: eine Datei
    ohne nachweisbare Herkunft ist kein Grund, das Rechnen zu ueberspringen.
    """
    try:
        meta = json.loads(str(np.load(path)["meta"]))
    except Exception:
        return False
    return meta.get("labeling") == identity_meta(labeling=lb)["labeling"]


def build_labels_for_run(cfg: LabelCfg, run_nr, verbose=True):
    """Laedt einen Run, labelt ihn, schreibt <run>_eventlabels.npz.

    KEIN Zeitfenster mehr: Labels gelten immer fuer den GANZEN Run. Ein
    Teilfenster brach spaeter in evaluate.align_gt_window ab -- die
    Fehlermeldung dort ist der Beleg, dass der Fall schon vorgekommen ist.
    Ausgabe haengt weiterhin NICHT von alpha/r_max/geom ab, deshalb liegen
    Labels ausserhalb von wp/ (BA_Konfiguration.md §4).

    Rueckgabe: (Pfad, stats-dict); stats=None, wenn uebersprungen.
    """
    split = cfg.selection.split
    lb = cfg.labeling
    out_dir = (Path(cfg.io.out_path) if cfg.io.out_path
               else config.label_dir(split))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"run{run_nr}_eventlabels.npz"

    if out_path.exists() and not cfg.io.overwrite:
        if _labeling_passt(out_path, lb):
            if verbose:
                print(f"  run{run_nr}: existiert schon -> uebersprungen")
            return out_path, None
        raise ValueError(
            f"{out_path.name} wurde mit ANDEREN Labeling-Parametern gebaut. "
            f"Jetzt verlangt: r_assign={lb.r_assign}, r_ambig={lb.r_ambig}, "
            f"box_w={lb.box_w}. Ohne io.overwrite=true wuerde still die alte "
            "Datei weiterverwendet -- jedes Folgeartefakt haenge dann an "
            "Parametern, die nie angewendet wurden.")

    bbox = load_bbox(split, run_nr)
    ev = load_events(split, run_nr)

    tic = time.time()
    track_id, status = assign_track_ids(ev["t"], ev["x"], ev["y"], bbox,
                                        r_assign=lb.r_assign,
                                        r_ambig=lb.r_ambig, box_w=lb.box_w)
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

    if len(track_id):
        assert track_id.max() < 2**15, \
            f"track_id {track_id.max()} sprengt int16 -- Speicherformat anpassen!"

    # SCHLANK speichern -- verlustfreie Integer-Casts (Rohdaten sind ganzzahlig):
    #   t: µs, max ~8e6 << 2^32 -> uint32 | x,y: 0..1279 -> uint16 | p: uint8
    np.savez(out_path,
             t=ev["t"].astype(np.uint32),
             x=ev["x"].astype(np.uint16),
             y=ev["y"].astype(np.uint16),
             p=ev["p"].astype(np.uint8),
             track_id=track_id.astype(np.int16),
             status=status.astype(np.uint8),
             meta=config.meta_json(stage="event_labels", split=split,
                                   run=run_nr, **identity_meta(labeling=lb),
                                   **st))

    del ev, track_id, status
    gc.collect()                      # RAM-Druck bei Batch-Laeufen daempfen
    if verbose:
        print(f"  -> {out_path.name}\n")
    return out_path, st

def build_labels(cfg: LabelCfg, verbose=True) -> list[int]:
    """Alle Runs aus cfg.selection. Rueckgabe: fehlgeschlagene Run-Nummern.

    LEERE Run-Liste bedeutet ALLE Runs des Splits -- das ersetzt das
    fruehere --all_split. Keine Einschraenkung anzugeben heisst, keine zu
    wollen; ein zweites Flag daneben waere eine Zustandsverdopplung.
    """
    split = cfg.selection.split
    runs = list(cfg.selection.runs) or config.available_runs(split)
    lb = cfg.labeling
    print(f"Event-Label-Erzeugung, Split '{split}', {len(runs)} Run(s), "
          f"R_ASSIGN={lb.r_assign}px, R_AMBIG={lb.r_ambig}px\n")
    failed = []
    for r in runs:
        try:
            build_labels_for_run(cfg, r, verbose=verbose)
        except Exception as e:
            print(f"  !! run{r} uebersprungen: {type(e).__name__}: {e}")
            failed.append(r)
    if failed:
        print(f"\nFehlgeschlagene Runs ({len(failed)}): {failed}")
    return failed
