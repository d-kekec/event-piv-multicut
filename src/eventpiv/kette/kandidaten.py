#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
kandidaten.py -- BAUSTEIN [6a]: Kandidatenpaare zwischen Tracklets.

  kandidatenpaare      welche Knotenpaare der Transformer ueberhaupt sieht
  paar_labels          Label und Gueltigkeit je Paar
  kandidaten_fuer_run  einen Run laden, Knoten bauen, Paare bilden
  schreibe_kandidaten  Artefakt [6a], eine Datei je Run

DIE KANDIDATENSUCHE IST DIESELBE FUNKTION WIE IN STUFE 1, eine
Abstraktionsebene hoeher: graph.neighborhood_pairs, angewandt auf
TRACKLET-ZENTROIDE statt auf Events. Damit faellt der Kontrakt
t_min[i] <= t_min[j] gratis an -- also genau die kanonische Ordnung
"frueheres Tracklet ist Nummer 1", die c_ij = c_ji strukturell macht.
Bei Gleichstand gewinnt der kleinere Zeilenindex, und die Tabelle ist
nach tracklet_id sortiert; ein drittes Kriterium braucht es nicht.

DER RADIUS FOLGT DER PARTIKELGROESSE, NICHT DER DRIFT. Zwei Bruchstuecke
desselben Partikels im selben Fenster liegen bis zur Wolkenausdehnung
auseinander (box_w = 11 px); die Drift ueber alpha = 30 ms betraegt nur
v_max*alpha = 3,06 px. Daher hyperboloid: die Taille ist die Ausdehnung,
die Aufweitung die Drift.

PAARE IM SELBEN FENSTER SIND EINGESCHLOSSEN und keine Nebenwirkung.
Stufe 1 fragmentiert innerhalb eines Fensters absichtlich, damit bei
Ueberlappung nicht ein grosses verschmolzenes Tracklet entsteht --
zuruecknehmen kann das nur Stufe 2. Solche Paare haben dt nahe null und
fallen ueber dieselbe Nachbarschaft heraus.

ZEIT RELATIV ZUM KNOTEN. Die Stichprobe wird mit t - t_min des eigenen
Knotens abgelegt. Der Grund ist zuerst die Sache selbst: fuer die
Normierung je Paar -- Ursprung auf t_min des FRUEHEREN Knotens -- ist die
relative Form die natuerliche. Nebenbei entfaellt damit die float32-
Grenze: exakt sind ganze Zahlen bis 2^24 = 16.777.216, ein Run ist
8.000.000 µs lang, also Faktor 2 Abstand. Relativ liegen alle Werte unter
alpha und die Frage stellt sich nicht mehr.
"""
from __future__ import annotations

import resource
import time
from dataclasses import replace

import numpy as np
from pathlib import Path
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components

from eventpiv.kern import config
from eventpiv.kern.graph import neighborhood_pairs, polarity_mask
from eventpiv.kern.runconfig import KandidatenCfg, Window, identity_meta
from eventpiv.kern.status import STATUS_NOISE
from eventpiv.kette.fenster import lade_fenster
from eventpiv.kette.knoten import knoten_bauen, stichprobe


def kandidatenpaare(knoten, *, alpha, r_max, geom, v_max):
    """Kandidatenpaare zwischen Knoten. Reine Funktion.

    Rueckgabe: (i, j) int64, Indizes in die Knotentabelle, i = frueheres
    t_min. Keine Selbstkanten, jede ungerichtete Kante genau einmal.

    Gesucht wird auf (t_min, x_c, y_c) -- der Zeitpunkt, an dem ein
    Tracklet BEGINNT, und sein Zentroid. t_max bleibt aussen vor: zwei
    Bruchstuecke desselben Partikels beginnen nahezu gleichzeitig, wenn
    sie im selben Fenster liegen, und aufeinanderfolgend, wenn nicht.
    """
    tab = knoten.tabelle
    if len(knoten) < 2:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return neighborhood_pairs(tab["t_min"], tab["x_c"], tab["y_c"],
                              alpha, r_max, geom=geom, v_max=v_max)


def paar_labels(knoten, i, j) -> dict:
    """Label und Gueltigkeit je Paar. Reine Funktion.

    GUELTIG ist ein Paar nur, wenn BEIDE Knoten genau EIN GT-Partikel
    tragen. Das schliesst zwei Faelle aus, und beide aus demselben Grund
    -- es gibt keine Wahrheit, gegen die man messen koennte:

      n_gt == 0  reines Rauschen. Gemessen 41,9 % der Tracklets, davon
                 aber 99,8 % unter min_events; oberhalb bleiben 647 ueber
                 zehn Runs, mit Mittel 490 Events. Vermutlich Wolken beim
                 Ein- und Austritt am Bildrand -- dort entstehen kompakte
                 Ereignisse ohne Box (Journal 14.08.2026).
      n_gt >= 2  teilverschmolzen. Enthaelt A zwei Partikel und B eines
                 davon, ist "selbe Trajektorie?" weder ja noch nein.
                 Betrifft 1,2 bis 1,8 % der Tracklets.

    DIE MASKE GILT NUR FUER DEN LOSS, NICHT FUER DEN GRAPHEN. Im Einsatz
    laufen diese Paare mit: ein Knoten, den man weglaesst, ist ein Stueck
    Trajektorie, das niemand mehr verbindet.

    gt_anteil bleibt Diagnose und filtert nicht. Ein Mehrheitslabel unter
    0,9 heisst unsicher, nicht falsch, und die Aggregation ueber ein
    ganzes Tracklet macht es robust.

    Rueckgabe: {} ohne Ground Truth -- dann traegt das Artefakt kein
    Label, wie bei tracklet_features.
    """
    if not knoten.hat_gt:
        return {}
    n_gt = knoten.tabelle["n_gt"]
    gt = knoten.tabelle["gt_track_id"]
    eindeutig = n_gt == 1
    gueltig = eindeutig[i] & eindeutig[j]
    return {"label": ((gt[i] == gt[j]) & gueltig).astype(np.int8),
            "gueltig": gueltig}


def _gt_komponenten(knoten, i, j) -> dict:
    """Zerfaellt ein GT-Partikel im Kandidatengraphen? Reine Funktion.

    DIE OBERGRENZE VON STUFE 2, und deshalb die wichtigste Zahl dieser
    Stufe. Multicut ist transitiv: es muss NICHT jedes Paar desselben
    Partikels ein Kandidat sein, aber die Tracklets eines Partikels
    muessen im Kandidatengraphen ZUSAMMENHAENGEN. Tun sie es nicht, kann
    kein Netz sie mehr verbinden -- die Fragmentierung steht dann schon
    vor dem Training fest.

    Gerechnet auf dem Teilgraphen der Kanten, deren beide Enden dasselbe
    MEHRHEITS-Partikel tragen. Beteiligt ist jeder Knoten mit n_gt >= 1,
    teilverschmolzene eingeschlossen.

    ANDERE MASKE ALS BEIM LOSS, und das ist Absicht: dort geht es um ein
    LABEL, hier um ERREICHBARKEIT. Ein verschmolzenes Tracklet gehoert
    seinem Mehrheitspartikel und ist als Bruecke voll brauchbar; es
    herauszunehmen meldete Brueche, die der Kandidatengraph gar nicht hat.
    Dieselbe Unterscheidung wie zwischen cluster_metrics und
    merge_metrics in evaluate.py.

    Fuer das MINDERHEITSPARTIKEL eines verschmolzenen Tracklets bleibt die
    Kette trotzdem unterbrochen -- richtig so: der Knoten kann am Ende nur
    in EINEM Cluster landen, dieses Partikel ist also wirklich nicht
    vollstaendig rekonstruierbar. Eine Grenze von Stufe 1, die Stufe 2
    nicht aufheben kann.

    Diese Zahl rechtfertigt kandidaten.alpha. Waere die Reichweite zu
    klein, saehe man es hier und an keiner anderen Kennzahl.

    MITTEL UND MAXIMUM ALLEIN LUEGEN, wie ueberall bei schiefen
    Verteilungen: wenige zersplitterte Partikel heben das Mittel, waehrend
    der Median 1 bleibt. Deshalb stehen Median, p95 und die ZAHL der
    Zersplitterten daneben.
    """
    tab = knoten.tabelle
    if "n_gt" not in tab:
        return {}
    m = len(knoten)
    beteiligt = tab["n_gt"] >= 1
    k = np.flatnonzero(beteiligt)
    if len(k) == 0:
        return {}

    gt = tab["gt_track_id"]
    gleich = beteiligt[i] & beteiligt[j] & (gt[i] == gt[j])
    a = coo_array((np.ones(int(gleich.sum()), np.int8),
                   (i[gleich], j[gleich])), shape=(m, m))
    _, komp = connected_components(a, directed=False)

    # Verschiedene Komponenten je Partikel, ohne Schleife -- dieselbe
    # Paar-Zaehlmechanik wie in features.mehrheit.
    gu, gidx = np.unique(gt[k], return_inverse=True)
    kombi = np.unique(gidx.astype(np.int64) * m + komp[k])
    n_komp = np.bincount(kombi // m, minlength=len(gu))
    n_trk = np.bincount(gidx, minlength=len(gu))
    return {
        "gt_n_partikel": int(len(gu)),
        "gt_tracklets_je_partikel_median": float(np.median(n_trk)),
        "gt_komponenten_mittel": float(n_komp.mean()),
        "gt_komponenten_median": float(np.median(n_komp)),
        "gt_komponenten_p95": float(np.percentile(n_komp, 95)),
        "gt_komponenten_max": int(n_komp.max()),
        "gt_n_zersplittert": int((n_komp > 1).sum()),
        # 1,0 hiesse: Stufe 2 KANN jedes Partikel vollstaendig
        # rekonstruieren. Alles darunter ist eine Obergrenze, die das Netz
        # nicht mehr anheben kann.
        "gt_frac_zusammenhaengend": float((n_komp == 1).mean()),
    }


def _kennzahlen(knoten, i, j, lab, n_roh, n_ev_zugeordnet, alpha1) -> dict:
    """Kennzahlen eines Runs. Absolute Zahlen neben den Anteilen, damit
    sich Runs ohne Gewichtungsfehler zusammenfassen lassen."""
    tab = knoten.tabelle
    m = len(knoten)
    out = {
        "n_tracklets_roh": int(n_roh),
        "n_knoten": m,
        "knoten_frac": float(m / n_roh) if n_roh else float("nan"),
        "knoten_event_frac": (float(tab["n_on"].sum() / n_ev_zugeordnet)
                              if n_ev_zugeordnet else float("nan")),
        "n_on_median": float(np.median(tab["n_on"])) if m else float("nan"),
        "n_kandidaten": int(len(i)),
        # Je KNOTEN, nicht je Paar: eine Kante haengt an zwei Knoten.
        "kandidaten_je_knoten": float(2 * len(i) / m) if m else float("nan"),
    }
    if m:
        grad = np.bincount(np.concatenate([i, j]), minlength=m)
        # Ein Knoten ohne Kandidaten kann von Stufe 2 nie verbunden werden
        # -- er bleibt ein einzelnes Tracklet, egal was das Netz sagt.
        out["n_knoten_ohne_kandidat"] = int((grad == 0).sum())
        out["knoten_ohne_kandidat_frac"] = float((grad == 0).mean())
        out["grad_median"] = float(np.median(grad))
        out["grad_max"] = int(grad.max())
    if len(i):
        dt = np.abs(tab["t_min"][j] - tab["t_min"][i])
        dr = np.hypot(tab["x_c"][j] - tab["x_c"][i],
                      tab["y_c"][j] - tab["y_c"][i])
        out["paar_dt_median_ms"] = float(np.median(dt) / 1000.0)
        out["paar_dr_median_px"] = float(np.median(dr))
        # Naeherung fuer "aus demselben Stufe-1-Fenster": exakt waere der
        # Fensterindex, den die Knotentabelle nicht traegt. Die Zahl sagt,
        # welcher Anteil der Kandidaten Fragmentierung zuruecknimmt statt
        # Fenster zu verbinden.
        out["paar_frac_dt_unter_alpha1"] = float((dt < alpha1).mean())
    if lab:
        g = lab["gueltig"]
        n_g = int(g.sum())
        out["n_gueltig"] = n_g
        out["gueltig_frac"] = float(g.mean()) if len(g) else float("nan")
        out["ja_frac"] = (float(lab["label"][g].mean()) if n_g
                          else float("nan"))
        out["ratio_ja_nein"] = (float(lab["label"][g].sum()
                                      / max(n_g - int(lab["label"][g].sum()), 1))
                                if n_g else float("nan"))
        n_gt = knoten.tabelle["n_gt"]
        out["knoten_rausch_frac"] = float((n_gt == 0).mean())
        out["knoten_merge_frac"] = float((n_gt >= 2).mean())
        out["gt_anteil_lt90_frac"] = float(
            (knoten.tabelle["gt_anteil"][n_gt >= 1] < 0.9).mean())
        out.update(_gt_komponenten(knoten, i, j))
    return out


def kandidaten_fuer_run(cfg: KandidatenCfg, run_nr: int, verbose=True):
    """Einen Run laden, Knoten bauen, Kandidatenpaare bilden.

    Der Eventstrom kommt ueber lade_fenster mit Window(None, None) --
    derselbe Weg, den solve_run genommen hat, als (A) entstand. Damit ist
    die Reihenfolge strukturell dieselbe und nicht nur zugesichert. Das
    Fenster ist hier KEIN Parameter: das Artefakt deckt immer den ganzen
    Run ab, und was sich nicht einstellen laesst, kann auch nicht falsch
    eingestellt werden.

    DER VERTRAG zu (A) wird geprueft und nicht angenommen: die Zahl der
    ON-Events im Strom gegen die Laenge von tracklet_id. Ein Versatz um
    einen Eintrag waere still und verschoebe jedes Event auf den falschen
    Knoten.

    Rueckgabe: (daten, lauf, quelle).
    """
    nb, kd, sel = cfg.neighborhood, cfg.kandidaten, cfg.selection
    t_start = time.time()

    rsel = replace(sel, run_nr=run_nr)
    quelle = config.tracklet_file(nb, cfg.pairs, cfg.training, rsel,
                                  cfg.solver)
    if not quelle.exists():
        raise SystemExit(
            f"{quelle} fehlt. Erst scripts/build_tracklets.py mit denselben "
            "Parametern laufen lassen.")
    with np.load(quelle) as d:
        tid = d["tracklet_id"]

    gt_path = (Path(cfg.io.label_dir) / f"run{run_nr}_eventlabels.npz"
               if cfg.io.label_dir
               else config.label_file(sel.split, run_nr))
    ganz = lade_fenster(rsel, Window(t0=None, dt=None),
                        polarity=nb.polarity, gt_path=gt_path)
    t_laden = time.time() - t_start

    if len(ganz) != len(tid):
        raise SystemExit(
            f"VERTRAG VERLETZT: {len(ganz):,} ON-Events im Strom, "
            f"{len(tid):,} im Artefakt {quelle.name}. Andere Polaritaet "
            "oder anderes Artefakt.")

    rausch = (None if not ganz.hat_gt
              else np.asarray(ganz.status) == STATUS_NOISE)
    kn = knoten_bauen(ganz.t, ganz.x, ganz.y, tid,
                      min_events=kd.min_events,
                      track_id=ganz.track_id if ganz.hat_gt else None,
                      status_noise=rausch)
    if len(kn) == 0:
        raise SystemExit(
            f"run{run_nr}: kein Knoten ueber min_events={kd.min_events}.")

    ev, maske = stichprobe(kn, ganz.t, ganz.x, ganz.y,
                           n_events=kd.n_events, seed=kd.seed)
    # Zeit relativ zum eigenen Knoten -- siehe Modul-Docstring. Nur wo die
    # Maske gilt; Fuell-Token bleiben auf 0. Gerechnet wird in float64,
    # abgelegt in float32: danach liegen alle Werte unter alpha.
    ev[:, :, 0] = np.where(maske,
                           ev[:, :, 0] - kn.tabelle["t_min"][:, None], 0.0)
    ev = ev.astype(np.float32)

    i, j = kandidatenpaare(kn, alpha=kd.alpha, r_max=kd.r_max,
                           geom=kd.geom, v_max=kd.v_max)
    lab = paar_labels(kn, i, j)

    n_roh = int(len(np.unique(tid[tid >= 0])))
    lauf = _kennzahlen(kn, i, j, lab, n_roh, int((tid >= 0).sum()), nb.alpha)
    lauf.update({"run": run_nr, "split": sel.split,
                 "n_events_run": len(ganz),
                 "diag": {
                     "sec_laden": round(t_laden, 1),
                     "sec_total": round(time.time() - t_start, 1),
                     "max_rss_mb": round(
                         resource.getrusage(
                             resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}})

    daten = {**{k: v for k, v in kn.tabelle.items()},
             "ev": ev, "maske": maske,
             "paar_i": i.astype(np.int32), "paar_j": j.astype(np.int32),
             **lab}
    if verbose:
        print(f"run{run_nr}: {n_roh:,} Tracklets -> {len(kn):,} Knoten "
              f"({lauf['knoten_frac']*100:.1f} %, "
              f"{lauf['knoten_event_frac']*100:.2f} % der Events), "
              f"{len(i):,} Kandidaten "
              f"({lauf['kandidaten_je_knoten']:.1f} je Knoten, "
              f"{lauf.get('n_knoten_ohne_kandidat', 0):,} ohne)",
              flush=True)
        if lab:
            print(f"  Label: {lauf['gueltig_frac']*100:.1f} % gueltig, "
                  f"davon {lauf['ja_frac']*100:.2f} % JA "
                  f"({lauf['ratio_ja_nein']:.2f} : 1); "
                  f"dt median {lauf['paar_dt_median_ms']:.1f} ms, "
                  f"{lauf['paar_frac_dt_unter_alpha1']*100:.0f} % unter alpha1",
                  flush=True)
        if "gt_frac_zusammenhaengend" in lauf:
            print(f"  OBERGRENZE: {lauf['gt_frac_zusammenhaengend']*100:.1f} % "
                  f"der {lauf['gt_n_partikel']} Partikel haengen zusammen "
                  f"({lauf['gt_n_zersplittert']} zersplittert); Komponenten "
                  f"median {lauf['gt_komponenten_median']:.0f}, "
                  f"p95 {lauf['gt_komponenten_p95']:.1f}, "
                  f"max {lauf['gt_komponenten_max']}", flush=True)
    return daten, lauf, quelle


def schreibe_kandidaten(cfg: KandidatenCfg, run_nr, daten, lauf, quelle):
    """Artefakt [6a]: Knoten, Eventstichprobe und Kandidatenpaare.

    EINE Datei je Run, wie bei (A) und (B). Die Stichprobe ist der grosse
    Teil: 93.000 Knoten mal 32 Events mal drei float32 sind rund 36 MB,
    dieselbe Groessenordnung wie das Tracklet-Artefakt selbst.

    Komprimiert wie (A) und (B) -- der Gewinn auf den Koordinaten ist
    NICHT gemessen und vermutlich kleiner als dort, weil die Maske und die
    Indexarrays gut, die Koordinaten schlecht komprimieren.

    LESEN: ev[k] gehoert zu Zeile k der Knotentabelle, maske[k] sagt
    welche Token echt sind, und ev[k, :, 0] ist die Zeit RELATIV zu
    t_min[k].
    """
    rsel = replace(cfg.selection, run_nr=run_nr)
    out = (Path(cfg.io.out_path) if cfg.io.out_path
           else config.kandidaten_file(cfg.neighborhood, cfg.pairs,
                                       cfg.training, rsel, cfg.solver,
                                       cfg.kandidaten, create=True))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, **daten,
        meta=config.meta_json(
            stage="kandidaten", quelle=str(quelle),
            **identity_meta(neighborhood=cfg.neighborhood, pairs=cfg.pairs,
                            training=cfg.training, solver=cfg.solver,
                            kandidaten=cfg.kandidaten, selection=rsel),
            **{k: v for k, v in lauf.items() if k != "diag"}))
    return Path(out)