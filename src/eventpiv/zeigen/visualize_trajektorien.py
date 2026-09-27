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

STUFE 2 WIRD GELESEN, NICHT GELOEST: die Partition liegt seit [6c] als
npz neben der Kennzahlen-JSON. Kein Rueckfallpfad auf Neurechnen -- zwei
Wege zur selben Partition, von denen einer selten laeuft, veralten still.
Damit braucht [D] weder GPU noch Netz.

DER FIT-PANEL legt die gefittete Bahn aus [7] ueber die Eventwolke, und
auch sie wird GELESEN: ein eigener Fit hier zeigte einen anderen als den
bewerteten, sobald ein Parameter nur einen der beiden Wege erreicht. Die
Fit-Gruppe muss deshalb dieselbe sein wie beim Fit-Lauf.

window schneidet nur die ANZEIGE. Die Farben werden einmal ueber den
ganzen Run vergeben: die eines Ausschnitts waeren sonst andere als die des
vollen Laufs, und genau ihre Stabilitaet ist die Aussage.
"""
from __future__ import annotations

import json

import numpy as np
from pathlib import Path

from eventpiv.kern import config
from eventpiv.kern.data import load_bbox
from eventpiv.kern.graph import polarity_mask
from eventpiv.kern.runconfig import AnimTrajCfg, slug
from eventpiv.kern.status import STATUS_CLEAN
from eventpiv.zeigen.visualize import (GREY, SENSOR_H, SENSOR_W, _crop_box,
                                       color_lookup, compose, render_panel,
                                       save_gif)

PANEL_NAMEN = ("roh", "tracklets", "trajektorien", "fit", "gt")


DUNKEL = 0.75           # Bahnlinie gegen ihre Eventwolke abgesetzt


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


def _partition(cfg, run_nr):
    """lab_ev und Kennzahlen aus den Artefakten von [6c]."""
    args = (cfg.neighborhood, cfg.pairs, cfg.training, cfg.solver,
            cfg.kandidaten, cfg.transformer, cfg.selection, cfg.solver2)
    npz = config.trajektorien_file(*args)
    js = config.solve2_file(*args)
    if not npz.exists() or not js.exists():
        raise SystemExit(
            f"{npz.name} oder {js.name} fehlt.\n"
            "Erst scripts/solve_stufe2.py mit denselben Gruppen laufen "
            "lassen -- [D] rechnet die Partition nicht nach.")
    with np.load(npz) as d:
        lab_ev = d["trajektorie_id"]
    return lab_ev, json.loads(js.read_text(encoding="utf-8"))["lauf"]


def _fit(cfg):
    """Gefittete Bahnen aus dem Artefakt von [7]: (t, x, y, trajektorie_id).

    Der Pfad haengt am Fit-Slug. Eine andere Fit-Gruppe als beim Fit-Lauf
    findet die Datei nicht, statt still einen anderen Fit zu zeigen.
    """
    pfad = config.fit_file(cfg.neighborhood, cfg.pairs, cfg.training,
                           cfg.solver, cfg.kandidaten, cfg.transformer,
                           cfg.selection, cfg.solver2, cfg.fit)
    if not pfad.exists():
        raise SystemExit(
            f"{pfad.name} fehlt.\n"
            "Erst scripts/fit_trajektorien.py mit denselben Gruppen laufen "
            "lassen -- [D] fittet nicht nach.")
    with np.load(pfad) as d:
        return d["t"], d["x"], d["y"], d["trajektorie_id"]


def _gt_mitten(cfg, run_nr, gt_ids):
    """Boxmitten je GT-Partikel. (t, x, y, track_id).

    Nur die Partikel, die im ON-Strom vorkommen -- sonst erscheinen im
    Panel Bahnen, zu denen es keine Events gibt.
    """
    bb = load_bbox(cfg.selection.split, run_nr)
    m = np.isin(bb["track_id"], np.unique(gt_ids[gt_ids >= 0]))
    return (bb["t"][m].astype(np.float64),
            bb["x"][m].astype(np.float64) + bb["w"][m].astype(np.float64) / 2,
            bb["y"][m].astype(np.float64) + bb["h"][m].astype(np.float64) / 2,
            bb["track_id"][m].astype(np.int64))


def animate_trajektorien(cfg: AnimTrajCfg):
    """Animiert Tracklets, Partition und Fit eines Runs aus den Artefakten.

    Rueckgabe: (gif_pfad, lauf).
    """
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

    # --- Stufe 2 aus dem Artefakt, ohne Solve
    lab_ev, lauf = _partition(cfg, run_nr)
    if len(lab_ev) != len(t):
        raise SystemExit(
            f"VERTRAG VERLETZT: {len(t):,} ON-Events im Strom, "
            f"{len(lab_ev):,} in der Partition.")

    # --- Farben, EINMAL ueber den ganzen Run. Je Panel eine Punktwolke
    # (x, y, Farbe, Zeit, kumulativ): die ersten Eintraege sind die
    # Events, angehaengte Punkte verblassen nicht mit dem Trail.
    quelle = {}

    def _panel(name, titel, xs, ys, farben, ts, kum):
        quelle[name] = (np.asarray(xs), np.asarray(ys), farben,
                        np.asarray(ts, np.float64), np.asarray(kum, bool),
                        titel)

    nein = np.zeros(len(t), bool)
    if "roh" in wahl:
        _panel("roh", "Rohevents", x, y,
               np.tile(np.array([70, 70, 70], np.uint8), (len(t), 1)),
               t, nein)
    if "tracklets" in wahl:
        _panel("tracklets",
               f"Stufe 1: Tracklets (grau: < {an.min_size} Events)",
               x, y, _farben(tid_ev, an.min_size), t, nein)
    if "trajektorien" in wahl:
        _panel("trajektorien", "Stufe 2: Trajektorien",
               x, y, _farben(lab_ev, an.min_size), t, nein)
    if "fit" in wahl:
        tfp, xfp, yfp, idfp = _fit(cfg)
        uniq, tab = color_lookup(lab_ev, min_size=an.min_size)
        # Fit und Partition teilen den ID-Raum. Eine ID, die die Partition
        # nicht kennt, bekaeme ueber searchsorted still eine fremde Farbe.
        if not np.isin(idfp, uniq).all():
            raise SystemExit(
                "Fit-Artefakt und Partition passen nicht zusammen: der Fit "
                "kennt Trajektorien, die die Partition nicht hat.")
        # Stuetzpunkte statt Linien: render_panel zeichnet Punkte, und die
        # Linie entsteht aus dem Gitter des Fits von selbst. Sie folgt
        # downscale und crop wie jedes Event.
        farb_bahn = (tab[np.searchsorted(uniq, idfp)]
                     * DUNKEL).astype(np.uint8)
        _panel("fit", "Stufe 2 + Fit (dunkel: gefittete Bahn)",
               np.concatenate([x, xfp]), np.concatenate([y, yfp]),
               np.concatenate([_farben(lab_ev, an.min_size), farb_bahn]),
               np.concatenate([t, tfp]),
               np.concatenate([nein, np.ones(len(tfp), bool)]))
        print(f"  Fit: {len(np.unique(idfp))} Bahnen, "
              f"{len(tfp):,} Stuetzpunkte", flush=True)
    if "gt" in wahl:
        if gt is None:
            # Kein Abbruch: ohne Labels gibt es das Panel nicht, der Rest
            # schon. Dieselbe Politik wie in animate_tracklets.
            wahl.remove("gt")
            if not wahl:
                raise SystemExit("nur 'gt' gewaehlt, aber keine Labels da")
        else:
            g_uniq, g_tab = color_lookup(gt, min_size=1)
            tgp, xgp, ygp, idgp = _gt_mitten(cfg, run_nr, gt)
            farb_gt = (g_tab[np.searchsorted(g_uniq, idgp)]
                       * DUNKEL).astype(np.uint8)
            _panel("gt", "Ground Truth (dunkel: Boxmitte)",
                   np.concatenate([x, xgp]), np.concatenate([y, ygp]),
                   np.concatenate([_farben(gt, 1), farb_gt]),
                   np.concatenate([t, tgp]),
                   np.concatenate([nein, np.ones(len(tgp), bool)]))

    # --- Anzeigeausschnitt. NACH der Farbvergabe, damit die Farben die des
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

    gem = dict(shape=shape, downscale=an.downscale, point=an.point,
               crop=zuschnitt)
    frames = []
    for s in starts:
        panels = []
        for p in wahl:
            xs, ys, farben, ts, kum, _ = quelle[p]
            # GLEITENDER Trail fuer Events, anders als in
            # animate_tracklets: dort wird an der Fenstergrenze
            # zurueckgesetzt, damit sie sichtbar bleibt -- hier soll sie
            # gerade verschwinden. Kumulative Punkte (Bahn, Boxmitte)
            # bleiben ab Fensteranfang stehen.
            unten = np.where(kum, lo, s - an.trail_ms * 1000.0)
            m = ((ts >= lo) & (ts < hi) & (ts < s + schritt) & (ts >= unten))
            panels.append(render_panel(xs, ys, farben, m, **gem))
        frames.append(compose(panels, [quelle[p][5] for p in wahl],
                              f"{(s - t[0])/1000:.0f} ms"))

    out_dir = (config.results_dir() / "animations" if cfg.io.out_path is None
               else Path(cfg.io.out_path))
    p_tag = "_" + "".join(w[0] for w in wahl)
    # Der Fit-Slug nur mit fit-Panel -- sonst schrieben zwei
    # Fit-Arbeitspunkte in dieselbe Datei.
    f_tag = f"_{slug(cfg.fit)}" if "fit" in wahl else ""
    # BEIDE Solver: Stufe 1 waehlt die Quelle, Stufe 2 die Rechnung. Ohne
    # den ersten schrieben zwei cost_offset-Werte in dieselbe Datei.
    name = (f"traj_{sel.split}_run{run_nr}_{slug(cfg.kandidaten)}_"
            f"{slug(cfg.transformer)}_{cfg.solver.short()}_"
            f"{cfg.solver2.short()}{f_tag}{p_tag}.gif")
    pfad = save_gif(frames, out_dir / name,
                    an.hold_ms or max(an.frame_ms * 2, 60))
    return pfad, lauf