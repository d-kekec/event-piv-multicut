#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kandidatenpaare und Paar-Labels.

Ohne Datensatz: beide Funktionen sind rein. Die Suche geht gegen eine
offensichtlich korrekte O(n^2)-Referenz, wie test_graph.py es fuer die
Eventebene vormacht -- hier eine Ebene hoeher, auf Zentroiden.
"""
import numpy as np
import pytest

from eventpiv.kette.kandidaten import kandidatenpaare, paar_labels
from eventpiv.kette.knoten import Knoten

ALPHA, R_MAX, V_MAX = 30_000.0, 15.0, 1.02e-4


def _knoten(t_min, x_c, y_c, n_gt=None, gt=None):
    """Knotentabelle von Hand. reihenfolge/grenzen bleiben leer -- die
    Kandidatensuche fasst sie nicht an."""
    m = len(t_min)
    tab = {"tracklet_id": np.arange(m, dtype=np.int64),
           "n_on": np.full(m, 50, np.int64),
           "x_c": np.asarray(x_c, np.float64),
           "y_c": np.asarray(y_c, np.float64),
           "t_min": np.asarray(t_min, np.float64),
           "t_max": np.asarray(t_min, np.float64) + 10_000.0}
    if n_gt is not None:
        tab["n_gt"] = np.asarray(n_gt, np.int32)
        tab["gt_track_id"] = np.asarray(gt, np.int64)
        tab["gt_anteil"] = np.ones(m)
    return Knoten(tab, np.empty(0, np.int64), np.zeros(m + 1, np.int64))


def _referenz(t, x, y, alpha, r_max, v_max):
    """Alle Paare im Hyperboloid, als doppelte Schleife."""
    aus = set()
    for a in range(len(t)):
        for b in range(a + 1, len(t)):
            dt = abs(t[a] - t[b])
            dr2 = (x[a] - x[b])**2 + (y[a] - y[b])**2
            if dt <= alpha and dr2 <= r_max**2 + (v_max * dt)**2:
                aus.add((a, b) if t[a] <= t[b] else (b, a))
    return aus


def test_suche_gegen_schleifenreferenz():
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 200_000, 60))
    x = rng.uniform(0, 120, 60)
    y = rng.uniform(0, 120, 60)
    kn = _knoten(t, x, y)

    i, j = kandidatenpaare(kn, alpha=ALPHA, r_max=R_MAX,
                           geom="hyperboloid", v_max=V_MAX)
    assert set(zip(i.tolist(), j.tolist())) == _referenz(
        t, x, y, ALPHA, R_MAX, V_MAX)


def test_kanonische_ordnung_und_keine_doppelten():
    """Der Kontrakt aus graph.py IST die Ordnung aus dem Entwurf: das
    Tracklet mit kleinerem t_min ist Nummer 1. Damit ist c_ij = c_ji
    strukturell und nicht durch Mittelung ueber den Tausch."""
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 100_000, 40))
    kn = _knoten(t, rng.uniform(0, 60, 40), rng.uniform(0, 60, 40))
    i, j = kandidatenpaare(kn, alpha=ALPHA, r_max=R_MAX,
                           geom="hyperboloid", v_max=V_MAX)

    assert (kn.tabelle["t_min"][i] <= kn.tabelle["t_min"][j]).all()
    assert (i != j).all()
    paare = list(zip(i.tolist(), j.tolist()))
    assert len(paare) == len(set(paare))
    assert not (set(paare) & {(b, a) for a, b in paare})


def test_gleiches_t_min_ordnet_nach_tracklet_id():
    """Zwei Bruchstuecke desselben Fensters koennen dieselbe erste
    Eventzeit haben. Dann muss der kleinere Zeilenindex vorn stehen -- die
    Tabelle ist nach tracklet_id sortiert, also ist das dritte Kriterium
    aus dem Entwurf erfuellt, ohne dass es jemand rechnet."""
    kn = _knoten([5000.0, 5000.0], [10.0, 12.0], [10.0, 10.0])
    i, j = kandidatenpaare(kn, alpha=ALPHA, r_max=R_MAX,
                           geom="hyperboloid", v_max=V_MAX)
    assert (i, j) == (0, 1) or (i.tolist(), j.tolist()) == ([0], [1])


def test_weniger_als_zwei_knoten():
    """Kein Sonderfall im Aufrufer, und neighborhood_pairs bekaeme sonst
    eine Instanz, auf der query_pairs nichts zu tun hat."""
    for m in (0, 1):
        kn = _knoten(np.zeros(m), np.zeros(m), np.zeros(m))
        i, j = kandidatenpaare(kn, alpha=ALPHA, r_max=R_MAX,
                               geom="hyperboloid", v_max=V_MAX)
        assert len(i) == 0 and len(j) == 0


def test_label_nur_bei_eindeutigem_gt_auf_beiden_seiten():
    """Rauschknoten (n_gt = 0) und teilverschmolzene (n_gt >= 2) haben kein
    definiertes Label. Die Maske gilt fuer den LOSS -- im Graphen laufen
    beide mit."""
    kn = _knoten([0.0, 1000.0, 2000.0, 3000.0],
                 [10.0, 11.0, 12.0, 13.0], [10.0, 10.0, 10.0, 10.0],
                 n_gt=[1, 1, 0, 2], gt=[7, 7, -1, 9])
    i = np.array([0, 0, 0, 1, 1, 2], np.int64)
    j = np.array([1, 2, 3, 2, 3, 3], np.int64)

    lab = paar_labels(kn, i, j)
    np.testing.assert_array_equal(
        lab["gueltig"], [True, False, False, False, False, False])
    # Nur das gueltige Paar traegt eine Aussage, und beide zeigen auf 7.
    assert lab["label"][0] == 1
    assert not lab["label"][1:].any()


def test_label_nein_bei_verschiedenen_partikeln():
    kn = _knoten([0.0, 1000.0], [10.0, 11.0], [10.0, 10.0],
                 n_gt=[1, 1], gt=[7, 8])
    lab = paar_labels(kn, np.array([0]), np.array([1]))
    assert lab["gueltig"][0] and lab["label"][0] == 0


def test_ohne_gt_kein_label():
    """Echte Messdaten haben keine Boxen -- dann traegt das Artefakt kein
    Label und alles andere bleibt, wie bei tracklet_features."""
    kn = _knoten([0.0, 1000.0], [10.0, 11.0], [10.0, 10.0])
    assert paar_labels(kn, np.array([0]), np.array([1])) == {}


def test_gt_komponenten_erkennt_den_zerfall():
    """Vier Tracklets eines Partikels, aber die Kandidatenkette reisst in
    der Mitte: zwei Komponenten, und Stufe 2 kann das Partikel nie mehr
    vollstaendig rekonstruieren."""
    from eventpiv.kette.kandidaten import _gt_komponenten

    kn = _knoten([0.0, 1000.0, 90_000.0, 91_000.0],
                 [10.0, 11.0, 12.0, 13.0], [10.0] * 4,
                 n_gt=[1, 1, 1, 1], gt=[7, 7, 7, 7])
    # Nur innerhalb der beiden Zeitgruppen gibt es Kandidaten.
    i, j = np.array([0, 2]), np.array([1, 3])
    z = _gt_komponenten(kn, i, j)
    assert z["gt_n_partikel"] == 1
    assert z["gt_komponenten_mittel"] == 2.0
    assert z["gt_frac_zusammenhaengend"] == 0.0

    # Eine Bruecke genuegt, und das Partikel haengt zusammen.
    z2 = _gt_komponenten(kn, np.array([0, 1, 2]), np.array([1, 2, 3]))
    assert z2["gt_komponenten_mittel"] == 1.0
    assert z2["gt_frac_zusammenhaengend"] == 1.0


def test_verschmolzenes_tracklet_bleibt_bruecke():
    """Beim LOSS ist n_gt >= 2 ausgeschlossen, bei der ERREICHBARKEIT
    nicht. Ein verschmolzenes Tracklet gehoert seinem Mehrheitspartikel
    und traegt dessen Kette weiter -- es herauszunehmen meldete einen
    Bruch, den der Kandidatengraph nicht hat."""
    from eventpiv.kette.kandidaten import _gt_komponenten

    kn = _knoten([0.0, 1000.0, 2000.0], [10.0, 11.0, 12.0], [10.0] * 3,
                 n_gt=[1, 2, 1], gt=[7, 7, 7])
    i, j = np.array([0, 1]), np.array([1, 2])

    z = _gt_komponenten(kn, i, j)
    assert z["gt_komponenten_mittel"] == 1.0
    assert z["gt_n_zersplittert"] == 0
    # Der Loss sieht dieselben Paare weiterhin als ungueltig.
    assert not paar_labels(kn, i, j)["gueltig"].any()