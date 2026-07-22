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


def cluster_metrics_naiv(labels, track_id, status):
    """Referenz: direkte Schleifen, keine Tricks."""
    clean = np.asarray(status) == STATUS_CLEAN
    lab = np.asarray(labels)[clean]
    gt = np.asarray(track_id)[clean]
    ok = lab >= 0
    lab, gt = lab[ok], gt[ok]
    if len(lab) == 0:
        return {}
    reinheit = []
    for c in np.unique(lab):
        g = gt[lab == c]
        _, cnt = np.unique(g, return_counts=True)
        reinheit.append(cnt.max() / cnt.sum())
    frag = [len(np.unique(lab[gt == p])) for p in np.unique(gt)]
    reinheit, frag = np.array(reinheit), np.array(frag)
    return {"n_events": int(len(lab)),
            "n_clusters": int(len(np.unique(lab))),
            "n_gt": int(len(np.unique(gt))),
            "purity_mean": float(reinheit.mean()),
            "purity_lt90": int((reinheit < 0.9).sum()),
            "frag_mean": float(frag.mean()),
            "frag_max": int(frag.max())}


def test_perfekte_partition():
    """Zwei Partikel, zwei Cluster, exakt deckungsgleich."""
    lab = np.array([0, 0, 0, 1, 1, 1])
    gt = np.array([7, 7, 7, 9, 9, 9])
    st = np.full(6, STATUS_CLEAN, np.uint8)
    m = cluster_metrics(lab, gt, st)
    assert m["purity_mean"] == 1.0
    assert m["frag_mean"] == 1.0
    assert m["n_clusters"] == m["n_gt"] == 2


def test_fragmentierung_von_hand():
    """Ein Partikel in DREI Cluster zerlegt -> Frag 3.0, Reinheit bleibt 1.0.

    Zeigt, warum beide Zahlen zusammengehoeren: perfekte Reinheit trotz
    voelliger Zerstueckelung.
    """
    lab = np.array([0, 0, 1, 1, 2, 2])
    gt = np.full(6, 5)
    st = np.full(6, STATUS_CLEAN, np.uint8)
    m = cluster_metrics(lab, gt, st)
    assert m["purity_mean"] == 1.0
    assert m["frag_mean"] == 3.0 and m["frag_max"] == 3


def test_reinheit_von_hand():
    """Ein Cluster mit 3x Partikel 1 und 1x Partikel 2 -> Reinheit 0.75."""
    lab = np.zeros(4, np.int64)
    gt = np.array([1, 1, 1, 2])
    st = np.full(4, STATUS_CLEAN, np.uint8)
    m = cluster_metrics(lab, gt, st)
    assert m["purity_mean"] == pytest.approx(0.75)
    assert m["purity_lt90"] == 1
    assert m["frag_mean"] == 1.0        # beide GT jeweils in EINEM Cluster


def test_nur_saubere_events_zaehlen():
    """Mehrdeutige und Rausch-Events duerfen die Metrik nicht beeinflussen."""
    lab = np.array([0, 0, 0, 0])
    gt = np.array([1, 1, 99, 99])
    st = np.array([STATUS_CLEAN, STATUS_CLEAN, STATUS_AMBIG, STATUS_NOISE],
                  dtype=np.uint8)
    m = cluster_metrics(lab, gt, st)
    assert m["n_events"] == 2 and m["n_gt"] == 1
    assert m["purity_mean"] == 1.0


def test_negative_label_werden_ignoriert():
    """RAMA vergibt -1 fuer Knoten ohne Kante -- die gehoeren keinem Cluster."""
    lab = np.array([-1, 0, 0])
    gt = np.array([1, 1, 1])
    st = np.full(3, STATUS_CLEAN, np.uint8)
    assert cluster_metrics(lab, gt, st)["n_events"] == 2


@pytest.mark.parametrize("seed", range(8))
def test_gegen_naive_referenz(seed):
    """Zufaellige Partitionen: vektorisiert == naiv."""
    rng = np.random.default_rng(seed)
    n = 400
    lab = rng.integers(-1, 25, n)
    gt = rng.integers(1, 12, n)
    st = rng.choice([STATUS_CLEAN, STATUS_AMBIG, STATUS_NOISE], n,
                    p=[0.9, 0.05, 0.05]).astype(np.uint8)
    a, b = cluster_metrics(lab, gt, st), cluster_metrics_naiv(lab, gt, st)
    assert set(a) >= set(b)
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
