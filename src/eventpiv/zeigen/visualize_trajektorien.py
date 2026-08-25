#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_trajektorien.py -- Darstellung [D]: was Stufe 2 aus Stufe 1 macht.

  animate_trajektorien   Tracklets | Trajektorien | GT, ganzer Run

DAS BILD DER STUFE. Links wechseln saemtliche Farben alle alpha µs, weil
dort neue Tracklets beginnen. Rechts stehen sie ueber acht Sekunden. Der
Unterschied zwischen beiden Panels IST das Ergebnis; Fragmentierung und
Identitaetswechsel sind darin sichtbar, ohne dass eine Kennzahl noetig
waere.

Getrennt von visualize.py, nicht angebaut: dort geht es um EIN Fenster
oder wenige, hier um den ganzen Run. Die gemeinsamen Bausteine --
color_lookup, render_panel, compose, save_gif -- werden importiert und
nicht kopiert.

DER TRACKLET-PANEL BRAUCHT KEINEN SOLVE. Artefakt (A) traegt tracklet_id
je ON-Event des ganzen Runs, fertig gerechnet. animate_tracklets ruft
solve_run nur, weil es zu seiner Entstehungszeit noch kein Artefakt gab;
ueber acht Sekunden waere das ein GPU-Job von Minuten.

Stufe 2 wird dagegen geloest (rund 30 s je Run). Die Cluster liegen
bewusst in keinem Artefakt -- sie sind aus [6a] und Checkpoint
reproduzierbar.

window schneidet nur die ANZEIGE. Geloest wird immer der ganze Run: die
Farben eines Ausschnitts waeren sonst andere als die des vollen Laufs, und
genau ihre Stabilitaet ist die Aussage.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

from eventpiv.kern import config
from eventpiv.kern.graph import polarity_mask
from eventpiv.kern.runconfig import AnimTrajCfg, slug
from eventpiv.kern.status import STATUS_CLEAN
from eventpiv.zeigen.visualize import (GREY, SENSOR_H, SENSOR_W, _crop_box,
                                       color_lookup, compose, render_panel,
                                       save_gif)

PANEL_NAMEN = ("roh", "tracklets", "trajektorien", "gt")


def _strom(cfg, run_nr):
    """ON-Strom des ganzen Runs, in der Reihenfolge von Artefakt (A).

    Bevorzugt aus der Label-Datei: sie traegt denselben Eventstrom und
    zusaetzlich track_id und status, und der Umweg ueber lade_fenster
    verlangte align_gt_window ohne Gewinn. Ohne Label-Datei -- der Fall
    echter Messdaten -- ueber data.load_events, dann ohne GT-Panel.

    Rueckgabe: (t, x, y, track_id oder None, status oder None).
    """
    lab = (Path(cfg.io.label_dir) / f"run{run_nr}_eventlabels.npz"
           if cfg.io.label_dir
           else config.label_file(cfg.selection.split, run_nr))
    if Path(lab).exists():
        with np.load(lab) as d:
            m = polarity_mask(d["p"], cfg.neighborhood.polarity)
            return (d["t"][m], d["x"][m].astype(np.int64),
                    d["y"][m].astype(np.int64),
                    d["track_id"][m], d["status"][m])
    from eventpiv.kern.data import load_events
    ev = load_events(cfg.selection.split, run_nr)
    m = polarity_mask(ev["p"], cfg.neighborhood.polarity)
    return (ev["t"][m], ev["x"][m].astype(np.int64),
            ev["y"][m].astype(np.int64), None, None)


def _farben(ids, min_size):
    """Farbe je Event aus einer Label-Spalte. EINMAL ueber den ganzen Run,
    nicht je Frame -- sonst wechselte die Farbe eines Partikels, sobald ein
    neuer Cluster ins Bild kommt."""
    uniq, tab = color_lookup(ids, min_size=min_size)
    return tab[np.searchsorted(uniq, ids)]


def animate_trajektorien(cfg: AnimTrajCfg):
    """Loest Stufe 2 fuer einen Run und animiert das Ergebnis.

    Rueckgabe: (gif_pfad, lauf).
    """
    from eventpiv.kette.solve2 import event_labels, solve2_run
    from eventpiv.kette.transformer import load_checkpoint

    sel, win, an = cfg.selection, cfg.window, cfg.anim
    run_nr = sel.run_nr
    if run_nr is None:
        raise SystemExit("selection.run_nr fehlt -- z.B. selection.run_nr=4")

    wahl = [p.strip() for p in an.panels.split(",") if p.strip()]
    unbekannt = [p for p in wahl if p not in PANEL_NAMEN]
    if unbekannt or not wahl:
        raise SystemExit(f"anim.panels={an.panels!r} -- erlaubt sind "
                         f"{', '.join(PANEL_NAMEN)} (Komma-getrennt)")

    # --- Stufe 1 aus dem Artefakt, ohne Solve
    tpfad = config.tracklet_file(cfg.neighborhood, cfg.pairs, cfg.training,
                                 sel, cfg.solver)
    if not tpfad.exists():
        raise SystemExit(
            f"{tpfad} fehlt.\n"
            f"`solver` waehlt die QUELLE, also welche Stufe-1-Tracklets "
            f"gelesen werden -- hier {cfg.solver.short()}. Den Solver von "
            f"STUFE 2 stellt `solver2` ein (aktuell "
            f"{cfg.solver2.short()}).\n"
            "Sonst erst scripts/build_tracklets.py mit denselben "
            "Parametern laufen lassen.")
    with np.load(tpfad) as d:
        tid_ev = d["tracklet_id"]

    t, x, y, gt, status = _strom(cfg, run_nr)
    if len(t) != len(tid_ev):
        raise SystemExit(
            f"VERTRAG VERLETZT: {len(t):,} ON-Events im Strom, "
            f"{len(tid_ev):,} im Artefakt {tpfad.name}.")

    # --- Stufe 2, ganzer Run
    mpath = (cfg.io.model_path or
             config.transformer_file(cfg.neighborhood, cfg.pairs,
                                     cfg.training, cfg.solver,
                                     cfg.kandidaten, cfg.transformer))
    if not Path(mpath).exists():
        raise SystemExit(f"{mpath} fehlt -- erst scripts/train_stufe2.py.")
    model, norm, _ = load_checkpoint(mpath, device=cfg.compute.device)
    lab_ev, lauf = solve2_run(cfg, model, norm, run_nr)

    # --- Farben, EINMAL ueber den ganzen Run
    quelle = {}
    if "roh" in wahl:
        quelle["roh"] = (np.tile(np.array([70, 70, 70], np.uint8),
                                 (len(t), 1)), "Rohevents")
    if "tracklets" in wahl:
        quelle["tracklets"] = (
            _farben(tid_ev, an.min_size),
            f"Stufe 1: Tracklets (grau: < {an.min_size} Events)")
    if "trajektorien" in wahl:
        quelle["trajektorien"] = (
            _farben(lab_ev, an.min_size),
            "Stufe 2: Trajektorien")
    if "gt" in wahl:
        if gt is None:
            # Kein Abbruch: ohne Labels gibt es das Panel nicht, der Rest
            # schon. Dieselbe Politik wie in animate_tracklets.
            wahl.remove("gt")
            if not wahl:
                raise SystemExit("nur 'gt' gewaehlt, aber keine Labels da")
        else:
            quelle["gt"] = (_farben(gt, 1),
                            "Ground Truth (grau: Rauschen)")

    # --- Anzeigeausschnitt. NACH dem Solve, damit die Farben die des
    # vollen Laufs bleiben.
    lo = t[0] if win.t0 is None else win.t0
    hi = t[-1] + 1 if win.dt is None else lo + win.dt
    sicht = (t >= lo) & (t < hi)
    if not sicht.any():
        raise SystemExit(
            f"Ausschnitt [{lo}, {hi}) enthaelt keine Events -- der Run laeuft "
            f"von {t[0]:.0f} bis {t[-1]:.0f} µs.")

    shape = (SENSOR_H // an.downscale, SENSOR_W // an.downscale)
    zuschnitt = (_crop_box(x[sicht], y[sicht], an.downscale, shape)
                 if an.crop else None)

    schritt = an.frame_ms * 1000.0
    starts = np.arange(lo, hi, schritt)
    print(f"{len(starts)} Frames à {an.frame_ms:.1f} ms, "
          f"{len(wahl)} Panel(s), Trail {an.trail_ms:.0f} ms, "
          f"{lauf['n_cluster']:,} Trajektorien aus "
          f"{lauf['n_knoten']:,} Tracklets", flush=True)
    if len(starts) * len(wahl) > 1200:
        print("  WARN: viele Frames mal Panels -- das GIF wird gross. "
              "anim.downscale erhoehen oder anim.panels einzeln waehlen.",
              flush=True)

    frames = []
    for s in starts:
        # GLEITENDER Trail, anders als in animate_tracklets: dort wird an
        # der Fenstergrenze zurueckgesetzt, damit sie sichtbar bleibt --
        # hier soll sie gerade verschwinden.
        m = sicht & (t < s + schritt) & (t >= s - an.trail_ms * 1000.0)
        gem = dict(shape=shape, downscale=an.downscale, point=an.point,
                   crop=zuschnitt)
        frames.append(compose(
            [render_panel(x, y, quelle[p][0], m, **gem) for p in wahl],
            [quelle[p][1] for p in wahl],
            f"{(s - t[0])/1000:.0f} ms"))

    out_dir = (config.results_dir() / "animations" if cfg.io.out_path is None
               else Path(cfg.io.out_path))
    p_tag = "_" + "".join(w[0] for w in wahl)
    # BEIDE Solver: Stufe 1 waehlt die Quelle, Stufe 2 die Rechnung. Ohne
    # den ersten schrieben zwei cost_offset-Werte in dieselbe Datei.
    name = (f"traj_{sel.split}_run{run_nr}_{slug(cfg.kandidaten)}_"
            f"{slug(cfg.transformer)}_{cfg.solver.short()}_"
            f"{cfg.solver2.short()}{p_tag}.gif")
    pfad = save_gif(frames, out_dir / name,
                    an.hold_ms or max(an.frame_ms * 2, 60))
    return pfad, lauf