#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py -- Cluster-Qualitaet gegen die GT-track_id.

Zwei Kennzahlen, die IMMER ZUSAMMEN gelesen werden muessen:

  REINHEIT je Cluster: groesster GT-Anteil in diesem Cluster.
      1.0 = der Cluster enthaelt nur ein Partikel.
  FRAGMENTIERUNG je GT-Partikel: in wie viele Cluster zerfaellt es?
      1.0 = ideal, das Partikel ist genau ein Cluster.

Warum zusammen: Reinheit allein ist trivial zu maximieren -- lauter
Singletons sind perfekt rein. Genau das lieferte das dr_weight-Netz:
Reinheit 0.999 bei Fragmentierung 5.38.

EINSCHRAENKUNG: bewertet werden nur SAUBERE Events
(status == 0). Mehrdeutige -- also genau die schweren Ueberlappungsfaelle --
fliessen nicht ein. Die Bewertung ist damit OPTIMISTISCH. Fuer RELATIVE
Vergleiche zwischen Konfigurationen unkritisch (die Verzerrung trifft alle
gleich); die FINALE Zahl braucht eine ehrlichere Metrik.
"""
from __future__ import annotations

import numpy as np

from eventpiv.graph import edge_deltas
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
    RELATIVE Vergleiche ist beides brauchbar, fuer die FINALE Zahl ist die
    eventgewichtete die ehrlichere.

    purity_event_mean entsteht direkt aus groesste/groessen:
    sum(groesste)/sum(groessen) ist der Anteil der Events, die in ihrem
    Cluster zur Mehrheits-GT gehoeren -- keine zweite Rechnung noetig.

    EINSCHRAENKUNG: bewertet werden nur SAUBERE Events (status == 0). Auch
    die gt_-Felder gelten damit nur fuer diese Teilmenge --
    gt_visible_ms_median unterschaetzt die Sichtbarkeit, wenn ein Partikel am
    Rand mehrdeutige Events hat. Bewusst so: eine zweite Maske wuerde die
    Ausrichtung duplizieren, und genau dort liegt die Falle
    (align_gt_window).
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

    Warum nicht per Zeitmaske: viele Events teilen sich einen Zeitstempel
    (rund vier je µs), und dann liefert (t >= t0) & (t <= t1) am Fensterrand
    mehr Eintraege als das geladene Fenster hat. Da die Label-Datei denselben
    Eventstrom in derselben Reihenfolge enthaelt, ist der Ausschnitt
    ZUSAMMENHAENGEND: Startindex suchen, n Eintraege nehmen, Zeitstempel
    gegenpruefen.

    Ein GEFILTERTER Eventstrom (ON-only) faellt hier auf: er ist eine
    Teilfolge, die Positionen laufen sofort auseinander, der Vergleich
    schlaegt fehl. Die Meldung zeigt dann allerdings auf die falsche
    Ursache. Richtig bleibt: erst auf dem ungefilterten Strom ausrichten,
    dann dieselbe Maske auf beide Seiten.

    Rueckgabe: slice. ValueError, wenn die Zeitstempel nicht passen.
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


def edge_gt_stats(i, j, t, x, y, track_id, status) -> dict:
    """Statistik ueber die KANTEN eines Fensters, gegen die GT.

    Gegenstueck zu cluster_metrics: dort wird die AUSGABE des Solvers
    bewertet, hier seine EINGABE. edge_ja_frac ist der Anteil der Kanten,
    die zwei Events desselben Partikels verbinden -- die Klassenverteilung,
    die das Netz im Einsatz tatsaechlich vorfindet.

    Wozu: pred_neg_frac auf dem Holdout und frac_neg im Einsatz weichen
    voneinander ab, und ohne edge_ja_frac liesse sich nicht trennen, ob das
    an der Klassenverteilung liegt oder an den Merkmalen.

    NUR Kanten mit ZWEI sauberen Endpunkten sind bewertbar -- dieselbe
    Bedingung wie beim Sammeln des Paar-Datensatzes, damit die Zahlen
    vergleichbar sind. edge_bewertbar_frac sagt, auf welchem Anteil der
    Kanten die Aussage ruht.

    dt_mean und dr_mean laufen dagegen ueber ALLE Kanten -- das ist, was das
    Netz im Einsatz sieht. Mit data/mu aus dem Training nur naeherungsweise
    vergleichbar, weil dort ausschliesslich sauber-sauber-Paare stehen.

    Legt vier Arrays der Kantenzahl in float64 an. Wird NACH dem RAMA-Aufruf
    gerechnet und liegt deshalb nicht in der Speicherspitze.
    """
    i, j = np.asarray(i), np.asarray(j)
    if len(i) == 0:
        return {}
    dt, dx, dy = edge_deltas(t, x, y, i, j)
    out = {"dt_mean": float(dt.mean()),
           "dr_mean": float(np.sqrt(dx * dx + dy * dy).mean())}

    clean = np.asarray(status) == STATUS_CLEAN
    beide = clean[i] & clean[j]
    n_bew = int(beide.sum())
    out["edge_n_bewertbar"] = n_bew
    out["edge_bewertbar_frac"] = float(n_bew / len(i))
    if n_bew:
        tid = np.asarray(track_id)
        out["edge_ja_frac"] = float(
            (tid[i[beide]] == tid[j[beide]]).mean())
    return out