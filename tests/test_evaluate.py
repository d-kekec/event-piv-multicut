#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Invariantentests fuer evaluate.

Kernstueck: cluster_metrics_naiv als offensichtlich korrekte Referenz gegen
die vektorisierte Fassung -- dasselbe Muster wie der Brute-Force-Test in
test_graph.py. Zusaetzlich Faelle mit von Hand ausgerechnetem Ergebnis.
"""
import numpy as np
import pytest

from eventpiv.evaluate import align_gt_window, cluster_metrics
from eventpiv.labels import STATUS_AMBIG, STATUS_CLEAN, STATUS_NOISE


def cluster_metrics_naiv(labels, track_id, status, t=None):
    """Referenz: direkte Schleifen, keine Tricks.

    Deckt bewusst ALLE Felder ab, nicht nur die urspruenglichen: sonst
    laufen neue Kennzahlen ungeprueft mit, und genau dort sitzen die Fehler
    (purity_event_mean nimmt hier den anderen Rechenweg -- Summe ueber
    Reinheit x Groesse statt Summe der Mehrheitsanteile).
    """
    clean = np.asarray(status) == STATUS_CLEAN
    lab = np.asarray(labels)[clean]
    gt = np.asarray(track_id)[clean]
    tt = np.asarray(t)[clean] if t is not None else None
    ok = lab >= 0
    lab, gt = lab[ok], gt[ok]
    if tt is not None:
        tt = tt[ok]
    if len(lab) == 0:
        return {}

    reinheit, groessen = [], []
    for c in np.unique(lab):
        _, cnt = np.unique(gt[lab == c], return_counts=True)
        reinheit.append(cnt.max() / cnt.sum())
        groessen.append(cnt.sum())
    frag = np.array([len(np.unique(lab[gt == p])) for p in np.unique(gt)])
    gt_gr = [int((gt == p).sum()) for p in np.unique(gt)]
    reinheit, groessen = np.array(reinheit), np.array(groessen)

    out = {"n_events": int(len(lab)),
           "n_clusters": int(len(np.unique(lab))),
           "n_gt": int(len(np.unique(gt))),
           "purity_cluster_mean": float(reinheit.mean()),
           "purity_event_mean": float((reinheit * groessen).sum()
                                      / groessen.sum()),
           "purity_lt90": int((reinheit < 0.9).sum()),
           "purity_lt90_frac": float((reinheit < 0.9).mean()),
           "cluster_size_min": int(groessen.min()),
           "cluster_size_median": float(np.median(groessen)),
           "cluster_size_max": int(groessen.max()),
           "n_singletons": int((groessen == 1).sum()),
           "frag_mean": float(frag.mean()),
           "frag_max": int(frag.max()),
           "gt_events_per_particle_median": float(np.median(gt_gr))}
    if tt is not None:
        sp = [tt[gt == p].max() - tt[gt == p].min() for p in np.unique(gt)]
        out["gt_visible_ms_median"] = float(np.median(sp) / 1000.0)
    return out


def test_perfekte_partition():
    """Zwei Partikel, zwei Cluster, exakt deckungsgleich."""
    lab = np.array([0, 0, 0, 1, 1, 1])
    gt = np.array([7, 7, 7, 9, 9, 9])
    st = np.full(6, STATUS_CLEAN, np.uint8)
    m = cluster_metrics(lab, gt, st)
    assert m["purity_cluster_mean"] == 1.0
    assert m["purity_event_mean"] == 1.0
    assert m["frag_mean"] == 1.0
    assert m["n_clusters"] == m["n_gt"] == 2
    assert m["n_singletons"] == 0


def test_fragmentierung_von_hand():
    """Ein Partikel in DREI Cluster zerlegt -> Frag 3.0, Reinheit bleibt 1.0.

    Zeigt, warum beide Zahlen zusammengehoeren: perfekte Reinheit trotz
    voelliger Zerstueckelung. BEIDE Reinheiten sind hier 1.0 -- die
    Gewichtung hilft gegen Verschmelzung, nicht gegen Zerteilung.
    """
    lab = np.array([0, 0, 1, 1, 2, 2])
    gt = np.full(6, 5)
    st = np.full(6, STATUS_CLEAN, np.uint8)
    m = cluster_metrics(lab, gt, st)
    assert m["purity_cluster_mean"] == 1.0
    assert m["purity_event_mean"] == 1.0
    assert m["frag_mean"] == 3.0 and m["frag_max"] == 3


def test_reinheit_von_hand():
    """Ein Cluster mit 3x Partikel 1 und 1x Partikel 2 -> Reinheit 0.75."""
    lab = np.zeros(4, np.int64)
    gt = np.array([1, 1, 1, 2])
    st = np.full(4, STATUS_CLEAN, np.uint8)
    m = cluster_metrics(lab, gt, st)
    assert m["purity_cluster_mean"] == pytest.approx(0.75)
    assert m["purity_event_mean"] == pytest.approx(0.75)   # ein Cluster
    assert m["purity_lt90"] == 1
    assert m["purity_lt90_frac"] == pytest.approx(1.0)
    assert m["frag_mean"] == 1.0        # beide GT jeweils in EINEM Cluster


def test_nur_saubere_events_zaehlen():
    """Mehrdeutige und Rausch-Events duerfen die Metrik nicht beeinflussen."""
    lab = np.array([0, 0, 0, 0])
    gt = np.array([1, 1, 99, 99])
    st = np.array([STATUS_CLEAN, STATUS_CLEAN, STATUS_AMBIG, STATUS_NOISE],
                  dtype=np.uint8)
    m = cluster_metrics(lab, gt, st)
    assert m["n_events"] == 2 and m["n_gt"] == 1
    assert m["purity_cluster_mean"] == 1.0


def test_negative_label_werden_ignoriert():
    """RAMA vergibt -1 fuer Knoten ohne Kante -- die gehoeren keinem Cluster."""
    lab = np.array([-1, 0, 0])
    gt = np.array([1, 1, 1])
    st = np.full(3, STATUS_CLEAN, np.uint8)
    assert cluster_metrics(lab, gt, st)["n_events"] == 2


@pytest.mark.parametrize("seed", range(8))
def test_gegen_naive_referenz(seed):
    """Zufaellige Partitionen: vektorisiert == naiv, in ALLEN Feldern."""
    rng = np.random.default_rng(seed)
    n = 400
    lab = rng.integers(-1, 25, n)
    gt = rng.integers(1, 12, n)
    st = rng.choice([STATUS_CLEAN, STATUS_AMBIG, STATUS_NOISE], n,
                    p=[0.9, 0.05, 0.05]).astype(np.uint8)
    t = np.sort(rng.integers(0, 20_000, n)).astype(np.int64)
    a = cluster_metrics(lab, gt, st, t=t)
    b = cluster_metrics_naiv(lab, gt, st, t=t)
    assert set(a) == set(b), "Feld ungeprueft: " + str(set(a) ^ set(b))
    for k in b:
        assert a[k] == pytest.approx(b[k]), f"Abweichung in {k}"


def test_leere_bewertung():
    st = np.full(3, STATUS_NOISE, np.uint8)
    assert cluster_metrics(np.zeros(3), np.ones(3), st) == {}


def test_align_findet_das_fenster():
    gt_t = np.arange(0, 1000, 10)
    ev_t = np.arange(300, 400, 10)
    sl = align_gt_window(gt_t, ev_t)
    np.testing.assert_array_equal(gt_t[sl], ev_t)


def test_align_mit_doppelten_zeitstempeln():
    """Genau der Fall, an dem die alte Zeitmasken-Loesung scheiterte."""
    gt_t = np.array([10, 20, 20, 20, 30, 40])
    ev_t = np.array([20, 20])            # nur ZWEI der drei 20er geladen
    sl = align_gt_window(gt_t, ev_t)
    assert sl == slice(1, 3)


def test_align_meldet_unpassende_gt():
    with pytest.raises(ValueError):
        align_gt_window(np.array([0, 10, 20]), np.array([5, 15]))


def test_die_beiden_reinheiten_laufen_auseinander():
    """DER Fall, fuer den purity_event_mean existiert.

    Ein grosser reiner Cluster und ein kleiner schmutziger: clustergewichtet
    zaehlen beide gleich, eventgewichtet nicht. Von Hand:
      Cluster 0: 8 Events, alle GT 1               -> 1.00
      Cluster 1: 2 Events, 1x GT 1 + 1x GT 2       -> 0.50
      cluster: (1.00+0.50)/2 = 0.75
      event  : (8+1)/10       = 0.90
    """
    lab = np.array([0] * 8 + [1, 1])
    gt = np.array([1] * 9 + [2])
    st = np.full(10, STATUS_CLEAN, np.uint8)
    m = cluster_metrics(lab, gt, st)
    assert m["purity_cluster_mean"] == pytest.approx(0.75)
    assert m["purity_event_mean"] == pytest.approx(0.90)
    assert m["cluster_size_min"] == 2 and m["cluster_size_max"] == 8


def test_sichtbarkeitsspanne_von_hand():
    """gt_visible_ms_median: Spanne je Partikel, Median darueber.

    GT 1: t 0..4000 µs -> 4 ms ; GT 2: t 1000..9000 µs -> 8 ms
    Median von [4, 8] = 6.0
    """
    lab = np.array([0, 0, 0, 1, 1, 1])
    gt = np.array([1, 1, 1, 2, 2, 2])
    st = np.full(6, STATUS_CLEAN, np.uint8)
    t = np.array([0, 2000, 4000, 1000, 5000, 9000], dtype=np.int64)
    m = cluster_metrics(lab, gt, st, t=t)
    assert m["gt_visible_ms_median"] == pytest.approx(6.0)
    assert "gt_visible_ms_median" not in cluster_metrics(lab, gt, st)