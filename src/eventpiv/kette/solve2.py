#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
solve2.py -- BAUSTEIN [6c]: den Stufe-2-Graphen EINES Runs loesen.

  kanten_kosten   Logits fuer alle Kandidatenpaare, stapelweise
  event_labels    Cluster je Knoten -> Cluster je EVENT
  solve2_run      einen Run laden, loesen, bewerten

EIN SOLVE JE RUN, nicht je Fenster. Das ist der Sinn der Stufe:
Trajektorien laufen ueber alle achthundert Fenster, und ein Ausschnitt
zerschnitte genau das, was hier zusammengefuegt werden soll. Stufe 1 muss
fensterweise loesen, Stufe 2 darf es nicht.

BEWERTET WIRD JE EVENT, nicht je Tracklet -- damit die Zahlen unveraendert
mit denen von Stufe 1 vergleichbar sind. Der Weg zurueck:

    Event --(A): tracklet_id--> globale Tracklet-ID
          --(searchsorted)-----> Zeile der Knotentabelle
          --(Solverausgabe)----> Cluster

Events ohne Tracklet und Events aus gefilterten Tracklets bekommen -1,
wie kantenlose Knoten bei Stufe 1; _kreuztabelle laesst sie heraus.

DIE STUFE-1-BASISLINIE FAELLT GRATIS AN: setzt man jedes Tracklet in
seinen eigenen Cluster, ist das genau die Partition VOR Stufe 2 --
eingeschraenkt auf dieselbe Knotenmenge. Beide Zahlensaetze nebeneinander
sind die Aussage der Arbeit, nicht die absolute Guete von Stufe 2 allein.

RAMA REPARAMETRISIERT DIE KOSTEN IM SELBEN PUFFER. Jede Statistik ueber
Kantenkosten wird VOR dem Solveraufruf entnommen.
"""
from __future__ import annotations

import json
import resource
import time
from dataclasses import replace

import numpy as np
import torch
from pathlib import Path

from eventpiv.bewerten.evaluate import (cluster_metrics, fa_hota_metrics,
                                        gt_raster_dt, idf1_metrics,
                                        idsw_metrics, merge_metrics,
                                        tracking_metrics)
from eventpiv.bewerten.metrics import pair_report
from eventpiv.kern import config
from eventpiv.kern.data import load_bbox
from eventpiv.kern.graph import polarity_mask
from eventpiv.kern.runconfig import Solve2Cfg, identity_meta
from eventpiv.kette.loeser import SOLVER_FUNKTIONEN
from eventpiv.kette.transformer import baue_tokens


def kanten_kosten(model, norm, daten, i, j, *, batch, device):
    """Logits fuer alle Kandidatenpaare. Rueckgabe: float32-Tensor (P,).

    Stapelweise, weil 3 Mio. Paare mal 65 Tokens nicht gleichzeitig auf die
    Karte passen -- der KNOTENtensor tut es, und genau dafuer nimmt
    baue_tokens Indizes.

    Der Logit geht UNVERAENDERT als Kantenkosten weiter: das Netz lernt mit
    BCEWithLogits die Log-Odds, die Nullstelle liegt damit bei p = 0,5, und
    RAMA liest das Vorzeichen. Dieselbe Begruendung wie in Stufe 1.
    """
    ev, maske, t_min, x_c, y_c = daten
    aus = torch.empty(len(i), dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        for k in range(0, len(i), batch):
            bi = torch.from_numpy(i[k:k + batch]).to(device)
            bj = torch.from_numpy(j[k:k + batch]).to(device)
            tok, tm = baue_tokens(ev, maske, bi, bj, t_min, x_c, y_c,
                                  alpha=norm["alpha"], r_max=norm["r_max"])
            aus[k:k + batch] = model(tok, tm).float()
    return aus


def event_labels(tid_ev, knoten_tid, node_labels):
    """Cluster je EVENT. Reine Funktion.

    tid_ev      globale Tracklet-ID je ON-Event (Artefakt (A))
    knoten_tid  tracklet_id je Knoten, AUFSTEIGEND (Artefakt [6a])
    node_labels Clusterausgabe des Solvers, -1 fuer kantenlose Knoten

    Events, deren Tracklet gefiltert wurde oder die zu keinem gehoeren,
    bekommen -1. Ein Treffer wird geprueft und nicht angenommen:
    searchsorted liefert auch fuer fremde Werte eine Position, und ohne den
    Vergleich landeten gefilterte Events still im falschen Cluster.
    """
    tid_ev = np.asarray(tid_ev)
    lab = np.full(len(tid_ev), -1, np.int32)
    if len(knoten_tid) == 0:
        return lab
    pos = np.clip(np.searchsorted(knoten_tid, tid_ev), 0, len(knoten_tid) - 1)
    treffer = knoten_tid[pos] == tid_ev
    lab[treffer] = np.asarray(node_labels)[pos[treffer]]
    return lab


def _kanten_kennzahlen(kosten, i, j, gt, n_gt) -> dict:
    """Was das NETZ auf den echten Kanten tut -- VOR dem Solveraufruf.

    Gegenstueck zu evaluate.edge_gt_stats eine Ebene hoeher. Bewertbar sind
    nur Kanten, deren beide Knoten genau ein GT-Partikel tragen; dieselbe
    Bedingung wie beim Training unter label_modus="streng".
    """
    aus = {"n_kanten": int(len(i)),
           "frac_neg": float((kosten < 0).mean()),
           "kosten_median": float(np.median(kosten)),
           "kosten_min": float(kosten.min()),
           "kosten_max": float(kosten.max())}
    if gt is None:
        return aus
    bewertbar = (n_gt[i] == 1) & (n_gt[j] == 1)
    aus["edge_n_bewertbar"] = int(bewertbar.sum())
    aus["edge_bewertbar_frac"] = float(bewertbar.mean())
    if bewertbar.any():
        ja = (gt[i[bewertbar]] == gt[j[bewertbar]]).astype(np.int8)
        aus.update({f"edge_{k}": v for k, v in
                    pair_report(ja, kosten[bewertbar]).items()})
    return aus


def solve2_run(cfg: Solve2Cfg, model, norm, run_nr, verbose=True):
    """Einen Run laden, loesen, bewerten. Rueckgabe: (labels_ev, lauf).

    Gelesen wird das Kandidaten-Artefakt [6a] plus (A) fuer die
    Event-Zuordnung und die Label-Datei fuer die Bewertung. Ohne
    Label-Datei entstehen Cluster, aber keine Guetezahlen -- das ist der
    Einsatzfall auf echten Messdaten und kein Fehler.
    """
    t0 = time.time()
    sel = replace(cfg.selection, run_nr=run_nr)
    nb, kd, tf, sv2, cp = (cfg.neighborhood, cfg.kandidaten,
                           cfg.transformer, cfg.solver2, cfg.compute)

    kpfad = config.kandidaten_file(nb, cfg.pairs, cfg.training, sel,
                                   cfg.solver, kd)
    if not kpfad.exists():
        raise SystemExit(
            f"{kpfad} fehlt.\n"
            f"`solver` waehlt die QUELLE (hier {cfg.solver.short()}), "
            f"`solver2` den Solver von Stufe 2 (hier {sv2.short()}).\n"
            "Sonst erst scripts/build_kandidaten.py mit denselben Gruppen.")
    with np.load(kpfad) as f:
        knoten_tid = f["tracklet_id"]
        i, j = f["paar_i"].astype(np.int64), f["paar_j"].astype(np.int64)
        daten = (torch.from_numpy(f["ev"]).to(cp.device),
                 torch.from_numpy(f["maske"]).to(cp.device),
                 torch.as_tensor(f["t_min"], dtype=torch.float32).to(cp.device),
                 torch.as_tensor(f["x_c"], dtype=torch.float32).to(cp.device),
                 torch.as_tensor(f["y_c"], dtype=torch.float32).to(cp.device))
        hat_gt = "gt_track_id" in f.files
        gt = f["gt_track_id"] if hat_gt else None
        n_gt = f["n_gt"] if hat_gt else None
    m = len(knoten_tid)
    if len(i) == 0:
        raise SystemExit(f"run{run_nr}: keine Kandidatenpaare.")

    kosten = kanten_kosten(model, norm, daten, i, j,
                           batch=tf.batch * 4, device=cp.device)
    if sv2.cost_offset:
        kosten = kosten + float(sv2.cost_offset)
    # VOR dem Solveraufruf: RAMA reparametrisiert denselben Puffer.
    kosten_np = kosten.detach().cpu().numpy()
    lauf = _kanten_kennzahlen(kosten_np, i, j, gt, n_gt)
    t_kosten = time.time() - t0

    node_labels = SOLVER_FUNKTIONEN[sv2.backend](
        i, j, kosten, m, cp, sv2.mode)
    lauf.update({
        "n_knoten": m,
        "n_cluster": int((np.unique(node_labels) >= 0).sum()),
        "n_knoten_ohne_cluster": int((node_labels < 0).sum()),
        "knoten_je_cluster": float(m / max(int((np.unique(node_labels) >= 0)
                                               .sum()), 1)),
    })

    # --- zurueck auf die Events
    tpfad = config.tracklet_file(nb, cfg.pairs, cfg.training, sel, cfg.solver)
    with np.load(tpfad) as d:
        tid_ev = d["tracklet_id"]
    lab_ev = event_labels(tid_ev, knoten_tid, node_labels)
    # Stufe-1-Basislinie: jedes Tracklet ein eigener Cluster, auf DERSELBEN
    # Knotenmenge. Der Unterschied der beiden Zahlensaetze ist das Ergebnis.
    lab_vorher = event_labels(tid_ev, knoten_tid, np.arange(m, dtype=np.int32))

    gtp = (Path(cfg.io.label_dir) / f"run{run_nr}_eventlabels.npz"
           if cfg.io.label_dir else config.label_file(sel.split, run_nr))
    if Path(gtp).exists():
        with np.load(gtp) as g:
            mask_on = polarity_mask(g["p"], nb.polarity)
            tid_gt, status = g["track_id"][mask_on], g["status"][mask_on]
            # Zeitstempel je ON-Event. Ohne sie sind IDSW und Frag nicht
            # definiert; cluster_metrics gewinnt daraus zusaetzlich
            # gt_visible_ms_median, das bisher nur in Stufe 1 anfiel.
            t_gt = g["t"][mask_on]
        if len(tid_gt) != len(tid_ev):
            raise SystemExit(
                f"VERTRAG VERLETZT: {len(tid_gt):,} ON-Events im Labelstrom, "
                f"{len(tid_ev):,} im Artefakt {tpfad.name}.")
        # Rasterweite fuer IDSW und Frag AUS DEM DATENSATZ, nicht aus einer
        # Konstanten und nicht aus der Konfiguration. Feiner als die
        # GT-Abtastung ist die Referenz nicht aufgeloest; darunter messen
        # beide Masse die Abtastdichte statt des benannten Phaenomens
        # (Journal 17.08.). Kein Konfigurationswert, weil delta die
        # PARTITION nicht beruehrt: er gehoerte weder in die Identitaet --
        # der Baum behauptete eine Abhaengigkeit, die es nicht gibt -- noch
        # daneben, wo zwei Werte dieselbe Datei still ueberschrieben.
        delta_gt = gt_raster_dt(load_bbox(sel.split, run_nr))
        for name, lab in (("stufe1", lab_vorher), ("stufe2", lab_ev)):
            lauf[name] = {**cluster_metrics(lab, tid_gt, status, t=t_gt),
                          **merge_metrics(lab, tid_gt, status),
                          **tracking_metrics(lab, tid_gt, status),
                          **idf1_metrics(lab, tid_gt, status),
                          **idsw_metrics(lab, tid_gt, status, t_gt,
                                         delta=delta_gt),
                          # Ohne Rasterung, anders als idsw_metrics: eine
                          # Rasterung muesste auch TPA, FNA und FPA
                          # betreffen, sonst haetten Zaehler und Nenner von
                          # F verschiedene Bezugsmengen -- und FA-HOTA waere
                          # nicht mehr mit hota vergleichbar.
                          **fa_hota_metrics(lab, tid_gt, status, t_gt)}

    lauf.update({
        "run": run_nr, "split": sel.split,
        "diag": {"sec_kosten": round(t_kosten, 1),
                 "sec_total": round(time.time() - t0, 1),
                 "max_rss_mb": round(
                     resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                     / 1024, 1)}})
    if verbose:
        print(f"run{run_nr}: {m:,} Knoten, {len(i):,} Kanten, "
              f"{lauf['frac_neg']*100:.1f} % negativ -> "
              f"{lauf['n_cluster']:,} Cluster "
              f"({lauf['knoten_je_cluster']:.1f} Tracklets je Cluster) "
              f"[{lauf['diag']['sec_total']:.0f}s]", flush=True)
        if "stufe2" in lauf:
            a, b = lauf["stufe1"], lauf["stufe2"]
            print(f"  Fragmentierung {a['frag_mean']:.2f} -> "
                  f"{b['frag_mean']:.2f}   |   Reinheit "
                  f"{a['purity_event_mean']:.4f} -> "
                  f"{b['purity_event_mean']:.4f}   |   Merge "
                  f"{a['merge_event_frac']*100:.2f} -> "
                  f"{b['merge_event_frac']*100:.2f} %", flush=True)
            print(f"  HOTA {a['hota']:.4f} -> {b['hota']:.4f}   |   "
                  f"AssRe {a['ass_re']:.4f} -> {b['ass_re']:.4f}   |   "
                  f"AssPr {a['ass_pr']:.4f} -> {b['ass_pr']:.4f}   |   "
                  f"DetA {a['det_a']:.4f} -> {b['det_a']:.4f}", flush=True)
            print(f"  IDSW {a['idsw']:,} -> {b['idsw']:,}   |   "
                  f"LHN-Frag {a['lhn_frag']:,} -> {b['lhn_frag']:,}   |   "
                  f"betroffene Partikel {b['idsw_ptkl_betroffen']} / "
                  f"{b['n_gt']}   [Raster {b['delta_us']:.0f} µs]",
                  flush=True)
            print(f"  FA-HOTA {a['fa_hota']:.4f} -> {b['fa_hota']:.4f}   |   "
                  f"FragA {a['hota_frag_a']:.5f} -> {b['hota_frag_a']:.5f}"
                  f"   |   Fragmente {b['hota_n_fragmente']:,}, Median "
                  f"{b['hota_fragment_len_median']:.0f} Events", flush=True)
    return lab_ev, lauf


def schreibe_solve2(cfg: Solve2Cfg, run_nr, lauf) -> Path:
    """Ergebnis [6c] als JSON. Die Cluster selbst werden NICHT abgelegt --
    sie sind aus Artefakt und Netz reproduzierbar, und die Visualisierung
    rechnet sie ohnehin neu."""
    sel = replace(cfg.selection, run_nr=run_nr)
    out = (Path(cfg.io.out_path) if cfg.io.out_path
           else config.solve2_file(cfg.neighborhood, cfg.pairs, cfg.training,
                                   cfg.solver, cfg.kandidaten,
                                   cfg.transformer, sel, cfg.solver2,
                                   create=True))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"meta": json.loads(config.meta_json(
             stage="solve2",
             **identity_meta(neighborhood=cfg.neighborhood, pairs=cfg.pairs,
                             training=cfg.training, solver=cfg.solver,
                             kandidaten=cfg.kandidaten,
                             transformer=cfg.transformer,
                             solver2=cfg.solver2, selection=sel))),
         "lauf": lauf}, indent=2, ensure_ascii=False), encoding="utf-8")
    return out