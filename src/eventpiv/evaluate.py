#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py -- Cluster-Qualitaet gegen die GT-track_id.

Zwei Kennzahlen, die IMMER ZUSAMMEN gelesen werden muessen:

  REINHEIT je Cluster: groesster GT-Anteil in diesem Cluster.
      1.0 = der Cluster enthaelt nur ein Partikel.
  FRAGMENTIERUNG je GT-Partikel: in wie viele Cluster zerfaellt es?
      1.0 = ideal, das Partikel ist genau ein Cluster.

Warum zusammen: Reinheit allein ist trivial zu maximieren -- lauter Singletons
sind perfekt rein. Genau das passierte mit dem dr_weight-Netz (Reinheit 0.999
bei Fragmentierung 5.38, BA_NN_Kantenkosten.md §3.7).

EINSCHRAENKUNG (dokumentiert, BA_NN §3.7): bewertet werden nur SAUBERE Events
(status == 0). Mehrdeutige -- also genau die schweren Ueberlappungsfaelle --
fliessen nicht ein. Die Bewertung ist damit OPTIMISTISCH. Fuer RELATIVE
Vergleiche zwischen Konfigurationen unkritisch (die Verzerrung trifft alle
gleich); die FINALE Zahl braucht eine ehrlichere Metrik.
"""
from __future__ import annotations

import numpy as np

from eventpiv.labels import STATUS_CLEAN


def cluster_metrics(labels, track_id, status, t=None) -> dict:
    """Vergleicht eine Partition gegen die GT. Reine Funktion.

    labels    : Cluster-ID je Event (RAMA-Ausgabe; < 0 = keinem Cluster
                zugeordnet, entsteht bei Knoten ohne Kante)
    track_id  : GT je Event
    status    : labels.STATUS_* je Event
    t         : optional Zeitstempel in µs. Nur fuer gt_visible_ms_median --
                ohne sie fehlt genau dieses Feld, alles andere bleibt.

    Rueckgabe: flaches dict. Leeres dict, wenn nichts zu bewerten ist.

    ZWEI REINHEITEN, und die Unterscheidung ist keine Spitzfindigkeit:
      purity_cluster_mean  Mittel UEBER CLUSTER -- ein Cluster aus 3 Events
                           zaehlt so viel wie einer aus 5000.
      purity_event_mean    Mittel UEBER EVENTS -- jedes Event zaehlt gleich.
    Bei vielen kleinen Fragmenten liegt die erste hoeher als die zweite. Fuer
    RELATIVE Vergleiche ist beides brauchbar (die Verzerrung trifft alle
    Konfigurationen gleich), fuer die FINALE Zahl ist die eventgewichtete die
    ehrlichere. Der frueher hier stehende Alias purity_mean ist am 29.07.2026
    entfallen; bestehende solves/*.json wurden neu gerechnet.

    purity_event_mean laesst sich direkt aus groesste/groessen bilden:
    sum(groesste)/sum(groessen) ist der Anteil der Events, die in ihrem
    Cluster zur Mehrheits-GT gehoeren. Keine zweite Rechnung noetig.

    EINSCHRAENKUNG (unveraendert, BA_NN §3.7): bewertet werden nur SAUBERE
    Events (status == 0). Auch die gt_-Felder gelten damit nur fuer diese
    Teilmenge -- gt_visible_ms_median unterschaetzt die echte Sichtbarkeit,
    wenn ein Partikel am Rand mehrdeutige Events hat. Bewusst so, weil eine
    zweite Maske die Ausrichtung dupliziert haette, und genau dort liegt die
    Falle (align_gt_window).
    """
    labels = np.asarray(labels)
    clean = np.asarray(status) == STATUS_CLEAN
    lab = labels[clean]
    gt = np.asarray(track_id)[clean]
    tt = np.asarray(t)[clean] if t is not None else None

    gueltig = lab >= 0            # RAMA vergibt -1 fuer kantenlose Knoten
    lab, gt = lab[gueltig], gt[gueltig]
    if tt is not None:
        tt = tt[gueltig]
    if len(lab) == 0:
        return {}

    lab_u, lab_idx = np.unique(lab, return_inverse=True)
    gt_u, gt_idx = np.unique(gt, return_inverse=True)

    # (Cluster, GT)-Kombinationen zaehlen
    key = lab_idx.astype(np.int64) * len(gt_u) + gt_idx
    kombi, anzahl = np.unique(key, return_counts=True)
    k_lab = kombi // len(gt_u)
    k_gt = kombi % len(gt_u)

    # Reinheit: groesste GT-Gruppe je Cluster / Clustergroesse
    groesste = np.zeros(len(lab_u), dtype=np.int64)
    np.maximum.at(groesste, k_lab, anzahl)
    groessen = np.bincount(lab_idx, minlength=len(lab_u))
    reinheit = groesste / groessen

    # Fragmentierung: Zahl verschiedener Cluster je GT-Partikel
    frag = np.bincount(k_gt, minlength=len(gt_u))
    gt_groessen = np.bincount(gt_idx, minlength=len(gt_u))

    out = {
        "n_events": int(len(lab)),
        "n_clusters": int(len(lab_u)),
        "n_gt": int(len(gt_u)),
        "purity_cluster_mean": float(reinheit.mean()),
        "purity_event_mean": float(groesste.sum() / groessen.sum()),
        "purity_lt90": int((reinheit < 0.9).sum()),
        "purity_lt90_frac": float((reinheit < 0.9).mean()),
        "cluster_size_min": int(groessen.min()),
        "cluster_size_median": float(np.median(groessen)),
        "cluster_size_max": int(groessen.max()),
        "n_singletons": int((groessen == 1).sum()),
        "frag_mean": float(frag.mean()),
        "frag_max": int(frag.max()),
        "gt_events_per_particle_median": float(np.median(gt_groessen)),
    }
    if tt is not None:
        # Sichtbarkeitsspanne je GT-Partikel, vektorisiert ueber
        # np.maximum.at / np.minimum.at -- eine Schleife ueber die Partikel
        # waere bei groesseren Fenstern der Engpass.
        tmax = np.full(len(gt_u), -np.inf)
        tmin = np.full(len(gt_u), np.inf)
        np.maximum.at(tmax, gt_idx, tt)
        np.minimum.at(tmin, gt_idx, tt)
        out["gt_visible_ms_median"] = float(np.median(tmax - tmin) / 1000.0)
    return out

def align_gt_window(gt_t, event_t):
    """Findet den Index-Ausschnitt der Label-Datei, der zum Eventfenster passt.

    Warum nicht per Zeitmaske: bei mehreren Events mit demselben Zeitstempel
    am Fensterrand liefert (t >= t0) & (t <= t1) mehr Eintraege als das
    geladene Fenster hat -- die Vorgaengerversion hat solche Runs still
    uebersprungen. Da die Label-Datei denselben Eventstream in derselben
    Reihenfolge enthaelt, ist der Ausschnitt zusammenhaengend: Startindex
    suchen, n Eintraege nehmen, Zeitstempel gegenpruefen.

    Rueckgabe: slice. Wirft ValueError, wenn die Zeitstempel nicht passen
    (dann stammt die Label-Datei nicht aus diesem Run oder nur aus einem
    Teilfenster).
    """
    gt_t = np.asarray(gt_t)
    event_t = np.asarray(event_t)
    n = len(event_t)
    if n == 0:
        return slice(0, 0)

    start = int(np.searchsorted(gt_t, event_t[0], side="left"))
    sl = slice(start, start + n)
    ausschnitt = gt_t[sl]
    if len(ausschnitt) != n or not np.array_equal(
            ausschnitt.astype(np.int64), event_t.astype(np.int64)):
        raise ValueError(
            f"GT passt nicht zum Eventfenster (Start {start}, {n} Events). "
            "Wurde die Label-Datei ueber den GANZEN Run erzeugt?")
    return sl


