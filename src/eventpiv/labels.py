#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
labels.py -- BAUSTEIN [1]: Event -> track_id + Qualitaetsstatus.

Weist jedem Event eine track_id zu (TRAININGS-WAHRHEIT, NIE NN-Input!) und
markiert dabei drei Qualitaetsklassen:

    status = 0  SAUBER      eindeutig einem Partikel zugeordnet   -> Training
    status = 1  MEHRDEUTIG  fast gleich weit zu zwei Boxen        -> nicht ins
                            Training, aber zugeordnet (Bewertung)
    status = 2  RAUSCH      keiner Box nah genug                  -> raushalten

Verfahren: Event -> zeitlich naechster 60-Hz-Rasterpunkt -> raeumlich
naechstes Boxzentrum. Die zweite Bedingung (Abstand zum ZWEITnaechsten) ist
noetig, weil ein Run als Ganzes sauber sein kann und trotzdem einzelne
Grenz-Events enthaelt.

track_id und status beantworten ZWEI Fragen: track_id sagt, WOHIN ein Event
gehoert, status, wie VERLAESSLICH diese Zuordnung ist. Nur RAUSCH traegt -1
-- dort ist kein Zentrum nahe genug. Wer Trainingsdaten will, filtert auf
status == STATUS_CLEAN; wer bewerten will, nimmt track_id.

Ausgabe haengt NICHT von alpha/r_max/win_dt ab -> einmal je Run und Split
bauen.

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

from eventpiv.kern import config
from eventpiv.kern.data import load_bbox, load_events
from eventpiv.kern.runconfig import LabelCfg, Labeling, identity_meta
from eventpiv.kern.status import STATUS_AMBIG, STATUS_CLEAN, STATUS_NOISE

# Version des LABEL-INHALTS, nicht der Labeling-Parameter. Wird erhoeht,
# wenn sich die Bedeutung der Felder aendert, ohne dass sich `Labeling`
# aendert -- sonst gaelten alte Dateien als gueltig und wuerden still
# weiterverwendet (dieselbe Fehlerklasse wie FIELDS/raw6).
#   1 -> track_id nur fuer SAUBERE Events
#   2 -> track_id auch fuer MEHRDEUTIGE (-1 nur noch fuer RAUSCH)
LABEL_FORMAT = 2

# --- Defaults: EINE Quelle, runconfig.Labeling. Die Namen bleiben hier
# bestehen, ihren Wert beziehen sie von dort.
_LB_DEFAULT      = Labeling()
BOX_W_DEFAULT    = _LB_DEFAULT.box_w      # px, Kantenlaenge der GT-Box
R_ASSIGN_DEFAULT = _LB_DEFAULT.r_assign   # px, max. Abstand zum naechsten Zentrum
R_AMBIG_DEFAULT  = _LB_DEFAULT.r_ambig    # px, (d2-d1) darunter -> mehrdeutig


def assign_track_ids(t, x, y, bbox,
                     r_assign=R_ASSIGN_DEFAULT,
                     r_ambig=R_AMBIG_DEFAULT,
                     box_w=BOX_W_DEFAULT):
    """Ordnet jedem Event (t,x,y) eine track_id + Status zu.

    Reine Funktion: keine Pfade, keine Globals, kein I/O -> ohne Datensatz
    testbar.

    Der KDTree enthaelt BOXZENTREN, rund 50 Punkte -- nie Events. Jedes
    Event wird gegen die Partikel abgefragt, nie gegen andere Events;
    d1/d2 sind die Abstaende zum naechsten und zweitnaechsten PARTIKEL.

      1) Jedes Event dem zeitlich naechsten 60-Hz-Rasterpunkt zuordnen.
      2) Pro Rasterpunkt ein KDTree ueber die Zentren zu DIESEM Zeitpunkt,
         Abfrage nach den zwei naechsten.
      3) Status aus (d1, d2):
           d1 > r_assign        -> RAUSCH (2),      track_id = -1
           (d2 - d1) < r_ambig  -> MEHRDEUTIG (1),  track_id = naechstes
           sonst                -> SAUBER (0),      track_id des naechsten

    track_id ist die ZUORDNUNG, status die VERLAESSLICHKEIT dieser
    Zuordnung. Nur RAUSCH traegt -1, weil dort kein Zentrum nahe genug ist.

    Die BOXFORM geht nie ein: box_w dient allein dazu, die Ecke oben links
    in das Zentrum umzurechnen. Danach ist ein Partikel ein Punkt, und
    r_assign ist ein Radius um diesen Punkt.

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
        # Die searchsorted-Logik unten setzt mindestens zwei Rasterpunkte
        # voraus; sonst greift der clip ins Leere und ALLE Events fielen
        # still auf RAUSCH zurueck.
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

    # --- Events nach Rasterpunkt buendeln, ohne Python-Schleife ueber
    # Events: argsort plus zwei searchsorted-Laeufe geben die Gruppengrenzen.
    # `sl` enthaelt GLOBALE Eventindizes, die Masken darunter (is_clean etc.)
    # sind gruppenlokal -- daher die Doppelindizierung sl[is_clean].
    order = np.argsort(nearest_idx, kind="stable")
    grp = nearest_idx[order]
    grp_start = np.searchsorted(grp, np.arange(len(raster_t)), side="left")
    grp_end   = np.searchsorted(grp, np.arange(len(raster_t)), side="right")

    for ri in range(len(raster_t)):
        sl = order[grp_start[ri]:grp_end[ri]]
        if len(sl) == 0:
            continue
        # raster_t stammt aus np.unique(bbox["t"]) -- zu jedem Rasterpunkt
        # gibt es per Konstruktion Boxen.
        mask_box = bbox["t"] == raster_t[ri]

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
        # track_id auch fuer MEHRDEUTIGE. Das naechste Zentrum ist bereits
        # gerechnet und wurde bisher verworfen.
        # status entscheidet, was ins TRAINING geht -- daran aendert sich
        # nichts. track_id sagt nur, WOHIN ein Event gehoert, und wird fuer
        # die BEWERTUNG gebraucht: verschmolzene Bereiche bestehen
        # ueberwiegend aus mehrdeutigen Events, und solange die -1 tragen,
        # ist die Merge-Rate genau dort blind, wo sie hinschauen soll.
        zuordenbar = is_clean | is_ambig
        track_id[sl[zuordenbar]] = tids[i1[zuordenbar]]

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


class LabelingMismatch(RuntimeError):
    """Vorhandene Datei stammt aus anderen Labeling-Parametern.

    Kein Run-Problem: es betrifft alle Runs gleich und wird deshalb in
    build_labels NICHT je Run gefangen, sondern bricht den Lauf ab.
    """


def _labeling_passt(path, lb) -> bool:
    """Wurde die vorhandene Datei mit GENAU diesen Labeling-Parametern gebaut?

    Label-Dateien tragen bewusst keinen Slug. Der Preis: das Dateisystem
    kann nicht ausdruecken, mit welchen Parametern eine Datei entstand --
    die Antwort steht im meta-Block der .npz und wird hier gelesen.

    Fehlt oder bricht der Block, gilt die Datei als unbrauchbar: eine Datei
    ohne nachweisbare Herkunft ist kein Grund, das Rechnen zu ueberspringen.
    """
    try:
        with np.load(path) as z:
            meta = json.loads(str(z["meta"]))
    except Exception:
        return False
    return (meta.get("labeling") == identity_meta(labeling=lb)["labeling"]
            and meta.get("label_format") == LABEL_FORMAT)


def build_labels_for_run(cfg: LabelCfg, run_nr, verbose=True):
    """Laedt einen Run, labelt ihn, schreibt <run>_eventlabels.npz.

    KEIN Zeitfenster: Labels gelten immer fuer den GANZEN Run. Ein
    Teilfenster bricht spaeter in evaluate.align_gt_window ab.
    Die Ausgabe haengt von keinem Nachbarschaftsparameter ab, deshalb liegen
    Labels ausserhalb von wp/.

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
        raise LabelingMismatch(
            f"{out_path.name} passt nicht: entweder ANDERE Labeling-Parameter "
            f"(jetzt verlangt: r_assign={lb.r_assign}, r_ambig={lb.r_ambig}, "
            f"box_w={lb.box_w}) oder ein aelteres LABEL_FORMAT "
            f"(jetzt {LABEL_FORMAT}; Format 1 trug track_id nur fuer SAUBERE "
            "Events). Ohne io.overwrite=true wuerde still die alte Datei "
            "weiterverwendet, und jedes Folgeartefakt hinge an einer "
            "Bedeutung, die nie angewendet wurde.")

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

    if len(track_id) and track_id.max() >= 2**15:
        # Kein assert: python -O entfernt asserts, und der int16-Cast unten
        # liefe dann still ueber.
        raise ValueError(
            f"track_id {track_id.max()} sprengt int16 -- Speicherformat "
            "in labels.py anpassen.")

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
                                   label_format=LABEL_FORMAT, **st))

    del ev, track_id, status
    gc.collect()                      # RAM-Druck bei Batch-Laeufen daempfen
    if verbose:
        print(f"  -> {out_path.name}\n")
    return out_path, st

def build_labels(cfg: LabelCfg, verbose=True) -> list[int]:
    """Alle Runs aus cfg.selection. Rueckgabe: fehlgeschlagene Run-Nummern.

    LEERE Run-Liste bedeutet ALLE Runs des Splits: keine Einschraenkung
    anzugeben heisst, keine zu wollen. Ein zweites Flag daneben waere eine
    Zustandsverdopplung.
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
        except LabelingMismatch:
            raise
        except Exception as e:
            print(f"  !! run{r} uebersprungen: {type(e).__name__}: {e}")
            failed.append(r)
    if len(failed) == len(runs) and runs:
        # Die Schleife faengt jede Exception und macht weiter -- ohne diese
        # Grenze saehe ein Tippfehler im Code aus wie 100 kaputte Runs.
        raise RuntimeError(
            f"ALLE {len(runs)} Run(s) fehlgeschlagen -- das ist kein "
            "Datenproblem. Meldungen oben lesen (Pfad, Konfiguration, Code).")
    if failed:
        print(f"\nFehlgeschlagene Runs ({len(failed)}): {failed}")
    return failed
