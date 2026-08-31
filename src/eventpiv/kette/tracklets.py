#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
tracklets.py -- BAUSTEIN [5a]: einen ganzen Run in Tracklet-Fenstern
abschreiten und das Ergebnis als Artefakt schreiben.

  solve_run           einen Run in anstossenden Fenstern der Laenge alpha
  schreibe_tracklets  Artefakt (A): Event -> Tracklet, eine Datei je Run

Die Stufe darunter loest EIN Fenster (solve.solve_window) und weiss nichts
von Runs. Hier kommt dazu: die Fensterfolge, die globale Nummerierung der
Tracklets ueber Fenstergrenzen hinweg, die Aggregation der Fensterzeilen zu
Kennzahlen ueber den Lauf, und das Dateiformat.

Kein torch auf Modulebene: dieser Baustein rechnet nichts auf der GPU.
"""
from __future__ import annotations

import contextlib
import os
import sys
import time
import resource

import numpy as np
from pathlib import Path

from eventpiv.bewerten.evaluate import (NAH_PX, abstand_im_fenster, cluster_metrics,
                               merge_metrics, min_abstand_je_partikel,
                               nah_maske)
from eventpiv.kern import config
from eventpiv.kern.runconfig import SolveCfg, identity_meta
from eventpiv.kern.status import STATUS_AMBIG
from eventpiv.kette.fenster import Fenster, lade_fenster
from eventpiv.kette.solve import solve_window

@contextlib.contextmanager
def _stumm():
    """Unterdrueckt Ausgaben auf Dateideskriptor 1.

    RAMA meldet je Aufruf "Running purely primal solver"; bei 800 Fenstern
    je Run ist das 800-mal dieselbe Zeile. Die Meldung kommt aus dem
    C++-Teil, deshalb reicht contextlib.redirect_stdout nicht -- das ersetzt
    nur sys.stdout, nicht den Deskriptor darunter.

    stderr bleibt unberuehrt: Fehlermeldungen gehen dorthin und sollen
    sichtbar bleiben. Was hier verloren geht, sind ausschliesslich
    Statuszeilen auf stdout.
    """
    sys.stdout.flush()
    sicherung = os.dup(1)
    leer = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(leer, 1)
        yield
    finally:
        os.dup2(sicherung, 1)
        os.close(leer)
        os.close(sicherung)


def _nah_fern(zeilen):
    """Fasst die Merge-Rate getrennt fuer nahe und ferne Partikel zusammen.

    DIE Frage von Stufe 1: verschmilzt der Solver dort, wo Partikel sich
    begegnen -- und laesst er die uebrigen in Ruhe?

    Getrennt wird je PARTIKEL, nicht je Fenster. Ein Fenster mit 49 weit
    getrennten Partikeln und einem nahen Paar ist weder das eine noch das
    andere; ein Fenster-Etikett muesste sich fuer eine Seite entscheiden
    und waere fuer beide falsch.

    Aufsummiert werden absolute Zahlen, nicht Mittelwerte von Anteilen:
    sonst zaehlte ein Fenster mit zwanzig bewertbaren Events so viel wie
    eines mit zwanzigtausend.

    `fern_merge_event_frac` ist zugleich die Kontrolle. Sie sollte nahe
    null liegen -- verschmilzt dort etwas, sind zwei Partikel
    zusammengefallen, zwischen denen es gar keine Kanten geben duerfte.

    merge_event_frac ist dieselbe Rechnung ohne Trennung. Sie steht daneben,
    weil NAH und FERN allein nicht sagen, wie gross der Fehler insgesamt
    ist -- 8 % auf 7 % der Events sind etwas anderes als 8 % auf allen.
    """
    out = {"nah_schwelle_px": NAH_PX}
    mi_ges = 0
    for pre in ("nah", "fern"):
        ev = sum(z.get(f"{pre}_n_events", 0) for z in zeilen)
        mi = sum(z.get(f"{pre}_n_minderheit", 0) for z in zeilen)
        mi_ges += mi
        out[f"{pre}_n_events"] = ev
        out[f"{pre}_merge_event_frac"] = (mi / ev) if ev else float("nan")
    ges = out["nah_n_events"] + out["fern_n_events"]
    out["nah_event_anteil"] = (out["nah_n_events"] / ges) if ges else float("nan")
    # Die ungetrennte Zahl, aus denselben Summen -- exakt, nicht gemittelt.
    out["merge_event_frac"] = (mi_ges / ges) if ges else float("nan")
    return out


def _gewichtet(zeilen, key, gewicht):
    """Einen Anteil ueber die Fenster eines Runs mitteln, GEWICHTET.

    Ein Mittel von Anteilen zaehlte ein Fenster mit zwanzig Kanten so viel
    wie eines mit zwanzig Millionen -- dieselbe Begruendung wie in
    _nah_fern. Dort liegen die Zaehler vor und die Summe ist exakt; hier
    liegt nur der Anteil vor, also wird er mit seinem Nenner
    zurueckgewichtet.

    Fenster ohne den Schluessel bleiben aussen vor. Fehlt er ueberall, ist
    das Ergebnis nan -- eine 0 sahe wie eine Messung aus.
    """
    paare = [(z[key], z[gewicht]) for z in zeilen
             if key in z and gewicht in z]
    g = sum(w for _, w in paare)
    return (sum(v * w for v, w in paare) / g) if g else float("nan")


def solve_run(cfg: SolveCfg, model, mu, sig, feature_mode, *,
              verbose=True, on_fenster=None, stumm=True):
    """Schreitet EINEN Run in anstossenden Fenstern der Laenge alpha ab.

    Je Fenster unabhaengig geloest, NICHT gestitcht: ein an einer
    Fenstergrenze zerschnittenes Partikel ergibt zwei Tracklets, und das
    ist die beabsichtigte Ausgabe. Verbinden ist Aufgabe von Stufe 2.

    EINMAL geladen, danach mit searchsorted zerlegt -- bei rund 800
    Fenstern je Run traegt das die Laufzeit.

    TRACKLET-IDs SIND GLOBAL: RAMA nummeriert je Fenster ab 0, der Treiber
    addiert einen Versatz. -1 bleibt -1.

    on_fenster : optional Callable(zeile) je Fenster. Dieselbe Loesung wie
        on_epoch im Training -- ein Test haengt an eine Liste an, das Skript
        reicht wandb.log durch.
    
    stumm : RAMAs Statuszeilen unterdruecken. Bei 800 Fenstern je Run ist
        dieselbe Meldung 800-mal keine Information. Zum Nachsehen, was der
        Solver meldet, mit stumm=False aufrufen.

    Rueckgabe: (tracklet_id, zeilen, lauf).
      tracklet_id  int32 je ON-Event des Runs, in der Reihenfolge des
                   gefilterten Stroms. -1 = keinem Tracklet zugeordnet.
      zeilen       eine dict-Zeile je Fenster.
      lauf         Kennzahlen ueber den ganzen Run.
    """

    nb, sel = cfg.neighborhood, cfg.selection
    gt_path = (Path(cfg.io.label_dir) / f"run{sel.run_nr}_eventlabels.npz"
               if cfg.io.label_dir
               else config.label_file(sel.split, sel.run_nr))

    t_start = time.time()
    # cfg.window ist der AUSSCHNITT (None/None = ganzer Run). Die
    # Fensterlaenge des Abschreitens ist davon unberuehrt: die ist alpha.
    ganz = lade_fenster(sel, cfg.window, polarity=nb.polarity,
                        gt_path=gt_path)
    t_laden = time.time() - t_start
    n_ev = len(ganz)
    if verbose:
        ausschnitt = ("ganzer Run" if cfg.window.t0 is None
                      and cfg.window.dt is None
                      else f"AUSSCHNITT {cfg.window.short()}")
        print(f"run{sel.run_nr} ({sel.split}): {n_ev:,} Events "
              f"(polarity={nb.polarity}, {ausschnitt}), GT "
              f"{'ja' if ganz.hat_gt else 'nein'} [{t_laden:.1f}s]")
    if n_ev < 2:
        raise ValueError(f"run{sel.run_nr} hat {n_ev} Events -- nichts zu tun.")

    grenzen = np.searchsorted(ganz.t, np.arange(ganz.t[0], ganz.t[-1], nb.alpha))
    grenzen = np.append(grenzen, n_ev)

    tracklet_id = np.full(n_ev, -1, np.int32)
    versatz = 0
    zeilen = []

    raster_t = gt_ids = d_part = None
    if ganz.hat_gt:
        from eventpiv.kern.data import load_bbox
        raster_t, gt_ids, d_part = min_abstand_je_partikel(
            load_bbox(sel.split, sel.run_nr))

    for k, (lo, hi) in enumerate(zip(grenzen[:-1], grenzen[1:])):
        if hi - lo < 2:
            continue
        teil = Fenster(
            t=ganz.t[lo:hi], x=ganz.x[lo:hi], y=ganz.y[lo:hi],
            p=ganz.p[lo:hi],
            track_id=None if not ganz.hat_gt else ganz.track_id[lo:hi],
            status=None if not ganz.hat_gt else ganz.status[lo:hi])

        with (_stumm() if stumm else contextlib.nullcontext()):
            labels, info = solve_window(cfg, teil, model, mu, sig,
                                        feature_mode, verbose=False,
                                        dt_us=nb.alpha)
        info["fenster"] = k
        info["t0"] = float(teil.t[0])

        if labels is not None:
            gueltig = labels >= 0
            tracklet_id[lo:hi][gueltig] = labels[gueltig] + versatz
            # max()+1 statt der Clusterzahl: RAMA garantiert keine
            # luecklose Nummerierung, und verschenkte IDs kosten nichts.
            versatz += int(labels.max()) + 1
            if teil.hat_gt:
                info.update(cluster_metrics(labels, teil.track_id,
                                            teil.status, t=teil.t))
                info.update(merge_metrics(labels, teil.track_id, teil.status))
                # Cluster werden je Teilmenge neu gebildet -- unbedenklich,
                # weil zwei verschmolzene Partikel einander per Definition
                # nah sind, also beide in derselben Teilmenge liegen.
                nah = nah_maske(teil.track_id, gt_ids,
                                abstand_im_fenster(raster_t, gt_ids, d_part,
                                                   teil.t[0],
                                                   teil.t[0] + nb.alpha))
                for pre, m in (("nah", nah), ("fern", ~nah)):
                    mm = merge_metrics(labels, teil.track_id, teil.status,
                                       extra_maske=m)
                    info[f"{pre}_n_events"] = mm.get("merge_n_events", 0)
                    info[f"{pre}_n_minderheit"] = mm.get("merge_n_minderheit", 0)

        if "edge_ja_frac" in info:
            info["n_nein"] = int(round((1.0 - info["edge_ja_frac"])
                                       * info["edge_n_bewertbar"]))
        if teil.hat_gt:
            info["n_ambig"] = int((teil.status == STATUS_AMBIG).sum())
            info["ambig_frac"] = info["n_ambig"] / len(teil)
        zeilen.append(info)
        if on_fenster is not None:
            on_fenster(info)
        if verbose and (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(grenzen)-1} Fenster, "
                  f"{versatz:,} Tracklets bisher")

    n_tracklets = int((np.unique(tracklet_id) >= 0).sum())
    lauf = {
        "run": sel.run_nr, "split": sel.split,
        "n_events": n_ev,
        "n_fenster": len(zeilen),
        # Default -1 wie im Artefakt: ohne GT ist n_nein nicht bestimmbar.
        "n_fenster_ohne_nein": sum(1 for z in zeilen
                                   if z.get("n_nein", -1) == 0),
        "n_tracklets": n_tracklets,
        "n_events_unassigned": int((tracklet_id < 0).sum()),
        "tracklets_je_fenster": n_tracklets / max(len(zeilen), 1),
        "events_je_tracklet": ((tracklet_id >= 0).sum()
                               / max(n_tracklets, 1)),
        "frac_neg": _gewichtet(zeilen, "frac_neg", "n_edges"),
        "frag_mean": _gewichtet(zeilen, "frag_mean", "n_gt"),
        # _mittel, nicht _median: ein gewichtetes Mittel von Medianen ist
        # kein Median.
        "frag_groesster_anteil_mittel": _gewichtet(
            zeilen, "frag_groesster_anteil_median", "n_gt"),
        "merge_frac_strikt": _gewichtet(zeilen, "merge_frac_strikt",
                                        "merge_n_clusters"),
        "merge_frac_10": _gewichtet(zeilen, "merge_frac_10",
                                    "merge_n_clusters"),
        **{f"edge_{k}": sum(int(z.get(f"edge_{k}", 0)) for z in zeilen)
           for k in ("tp", "fp", "fn", "tn")},
        **_nah_fern(zeilen),
        "diag": {
            "sec_laden": round(t_laden, 1),
            "sec_total": round(time.time() - t_start, 1),
            # EINMAL je Run: ru_maxrss ist eine Hochwassermarke des Prozesses.
            "max_rss_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        },
    }
    _tp, _fn = lauf["edge_tp"], lauf["edge_fn"]
    _fp, _tn = lauf["edge_fp"], lauf["edge_tn"]
    lauf["edge_tpr"] = _tp / max(_tp + _fn, 1)
    lauf["edge_tnr"] = _tn / max(_tn + _fp, 1)
    if verbose:
        print(f"  {lauf['n_fenster']} Fenster, {n_tracklets:,} Tracklets, "
              f"{lauf['events_je_tracklet']:.0f} Events je Tracklet, "
              f"{lauf['n_fenster_ohne_nein']} Fenster ohne NEIN "
              f"[{lauf['diag']['sec_total']:.0f}s, "
              f"{lauf['diag']['max_rss_mb']:.0f} MB]")
        if lauf.get("nah_n_events"):
            print(f"  NAH (<{NAH_PX:.0f}px Nachbar, "
                  f"{lauf['nah_event_anteil']*100:.0f}% der Events): "
                  f"Merge {lauf['nah_merge_event_frac']*100:.2f}%   |   "
                  f"FERN: Merge "
                  f"{lauf['fern_merge_event_frac']*100:.2f}%")
    return tracklet_id, zeilen, lauf


def _spalte(zeilen, key, default=0):
    """Eine Kennzahl aus allen Fensterzeilen als Array."""
    return np.array([z.get(key, default) for z in zeilen])


def schreibe_tracklets(cfg: SolveCfg, tracklet_id, zeilen, lauf):
    """Artefakt (A): Event -> Tracklet, eine Datei je Run.

    INDIZIERUNG, und ohne diesen Satz ist die Datei unbrauchbar:
    `tracklet_id` hat EINEN Eintrag je ON-Event des Runs, in der Reihenfolge
    des polaritaetsgefilterten Label-Stroms. Ein Leser stellt die Zuordnung
    so her:

        with np.load(config.label_file(split, run)) as d:
            m = polarity_mask(d["p"], nb.polarity)
            t, x, y = d["t"][m], d["x"][m], d["y"][m]

    Danach gehoert `t[k]` zu `tracklet_id[k]`. -1 heisst KEIN Tracklet
    (kantenloser Knoten), nicht Tracklet 0.

    Warum nicht ueber alle Events: OFF-Events sind keine Knoten und
    bekaemen durchweg -1 -- das verdoppelte die Datei fuer nichts. Sie
    werden nachtraeglich ueber die Mehrheit ihrer k naechsten ON-Events
    zugeordnet.

    Daneben je Fenster sechs Kennzahlen. `fenster_n_nein` ist die
    Volkszaehlung der Begegnungen: sie sagt, welche Fenster ueberhaupt eine
    Trennung verlangen, und ist die Grundlage der CROSSING/CLEAN-Trennung
    bei der Bewertung. -1 heisst "keine GT, also nicht bestimmbar".

    Komprimiert gespeichert: das Array ist ueberwiegend aus kurzen Laeufen
    gleicher IDs aufgebaut und schrumpft dabei erheblich.
    """
    out = (Path(cfg.io.out_path) if cfg.io.out_path
           else config.tracklet_file(cfg.neighborhood, cfg.pairs,
                                     cfg.training, cfg.selection,
                                     cfg.solver, create=True))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        tracklet_id=np.asarray(tracklet_id, np.int32),
        fenster_t0=_spalte(zeilen, "t0", float("nan")).astype(np.float64),
        fenster_n_events=_spalte(zeilen, "n_events").astype(np.int32),
        fenster_n_edges=_spalte(zeilen, "n_edges").astype(np.int64),
        fenster_n_nein=_spalte(zeilen, "n_nein", -1).astype(np.int64),
        fenster_n_ambig=_spalte(zeilen, "n_ambig", -1).astype(np.int64),
        fenster_n_clusters=_spalte(zeilen, "n_clusters_rama", -1).astype(np.int32),
        meta=config.meta_json(
            stage="tracklets",
            **identity_meta(neighborhood=cfg.neighborhood, pairs=cfg.pairs,
                            training=cfg.training, solver=cfg.solver,
                            selection=cfg.selection),
            **lauf))
    return Path(out)