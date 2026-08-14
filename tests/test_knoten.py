#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Knotentabelle und Eventstichprobe.

Ohne Datensatz und ohne Framework: beide Funktionen sind rein. Gegen eine
offensichtlich korrekte Schleifenfassung geprueft, wie test_graph.py und
test_features.py es fuer die vektorisierten Stellen vormachen.
"""
import numpy as np
import pytest

from eventpiv.kette.knoten import knoten_bauen, stichprobe


def _strom():
    """Vier Tracklets mit 1, 3, 5 und 8 Events, dazu zwei kantenlose.

    t aufsteigend, wie im echten Strom -- knoten_bauen setzt das fuer die
    Zeitordnung innerhalb eines Knotens voraus.
    """
    tid = np.array([7, -1, 7, 4, 4, 4, 9, 9, 9, 9, 9,
                    2, 9, 9, 9, -1, 4, 7, 7, 7], np.int64)
    n = len(tid)
    t = np.arange(n, dtype=np.float64) * 100.0
    x = np.arange(n, dtype=np.float64)
    y = np.arange(n, dtype=np.float64) * 2.0
    return t, x, y, tid


def _referenz(t, x, y, tid, min_events):
    """Dieselbe Rechnung als Schleife. Bewusst langsam und offensichtlich."""
    aus = {}
    for k in sorted(set(int(v) for v in tid if v >= 0)):
        m = tid == k
        if int(m.sum()) < min_events:
            continue
        aus[k] = {"n_on": int(m.sum()),
                  "x_c": float(x[m].mean()), "y_c": float(y[m].mean()),
                  "t_min": float(t[m].min()), "t_max": float(t[m].max()),
                  "idx": np.flatnonzero(m)}
    return aus


@pytest.mark.parametrize("min_events", [1, 2, 4, 6])
def test_tabelle_gegen_schleifenreferenz(min_events):
    t, x, y, tid = _strom()
    kn = knoten_bauen(t, x, y, tid, min_events=min_events)
    ref = _referenz(t, x, y, tid, min_events)

    assert len(kn) == len(ref)
    np.testing.assert_array_equal(kn.tabelle["tracklet_id"],
                                  np.array(sorted(ref)))
    for i, k in enumerate(sorted(ref)):
        for feld in ("n_on", "x_c", "y_c", "t_min", "t_max"):
            assert kn.tabelle[feld][i] == pytest.approx(ref[k][feld]), feld
        lo, hi = kn.grenzen[i], kn.grenzen[i + 1]
        np.testing.assert_array_equal(kn.reihenfolge[lo:hi], ref[k]["idx"])


def test_kantenlose_events_fallen_heraus():
    """tracklet_id = -1 heisst KEIN Tracklet, nicht Tracklet 0."""
    t, x, y, tid = _strom()
    kn = knoten_bauen(t, x, y, tid, min_events=1)
    assert -1 not in kn.tabelle["tracklet_id"]
    assert not np.isin(np.flatnonzero(tid < 0), kn.reihenfolge).any()


def test_leerer_strom_und_alles_gefiltert():
    """Kein Sonderfall im Aufrufer: beides liefert eine leere Tabelle."""
    t, x, y, tid = _strom()
    assert len(knoten_bauen(t, x, y, np.full(len(tid), -1),
                            min_events=1)) == 0
    assert len(knoten_bauen(t, x, y, tid, min_events=100)) == 0


def test_gt_label_ignoriert_rauschen():
    """Rauschevents zaehlen beim Mehrheitsentscheid nicht mit -- sonst
    bekaeme ein Tracklet aus zwei echten und drei Rauschevents das
    Rausch-Label."""
    t, x, y, tid = _strom()
    gt = np.where(tid == 9, 5, 3).astype(np.int64)
    rausch = tid == 9
    gt[rausch] = 99                      # Rauschen behauptet ein anderes GT
    kn = knoten_bauen(t, x, y, tid, min_events=1,
                      track_id=gt, status_noise=rausch)
    assert kn.hat_gt
    i = int(np.flatnonzero(kn.tabelle["tracklet_id"] == 9)[0])
    # Nur Rauschevents -> kein GT-Partikel, wie bei mehrheit vereinbart.
    assert kn.tabelle["n_gt"][i] == 0
    assert kn.tabelle["gt_track_id"][i] == -1


@pytest.mark.parametrize("n_events", [1, 3, 4, 32])
def test_stichprobe_zieht_ohne_zuruecklegen(n_events):
    t, x, y, tid = _strom()
    kn = knoten_bauen(t, x, y, tid, min_events=1)
    ev, maske = stichprobe(kn, t, x, y, n_events=n_events, seed=0)

    assert ev.shape == (len(kn), n_events, 3)
    # float64 ist hier kein Zufall: absolute Zeitstempel bis 8e6 liegen
    # bei Faktor 2 unter der float32-Grenze von 2^24 = 16.777.216.
    assert ev.dtype == np.float64
    np.testing.assert_array_equal(
        maske.sum(1), np.minimum(kn.tabelle["n_on"], n_events))

    for i in range(len(kn)):
        lo, hi = kn.grenzen[i], kn.grenzen[i + 1]
        erlaubt = set(t[kn.reihenfolge[lo:hi]].tolist())
        gezogen = ev[i, maske[i], 0].astype(np.float64).tolist()
        assert len(gezogen) == len(set(gezogen)), "Event doppelt gezogen"
        assert set(gezogen) <= erlaubt, "fremdes Event in der Stichprobe"


def test_fuelltoken_stehen_hinten_und_sind_null():
    """Die Maske ist die einzige Unterscheidung -- steht ein Fuell-Token
    mitten in der Zeile, greift src_key_padding_mask ins Leere."""
    t, x, y, tid = _strom()
    kn = knoten_bauen(t, x, y, tid, min_events=1)
    ev, maske = stichprobe(kn, t, x, y, n_events=6, seed=0)
    for i in range(len(kn)):
        k = int(maske[i].sum())
        assert maske[i, :k].all() and not maske[i, k:].any()
        assert (ev[i, k:] == 0).all()


def test_stichprobe_ist_reproduzierbar():
    """Gleicher Seed, gleiche Auswahl -- sonst traegt derselbe Knoten in
    zwei Laeufen verschiedene Events und nichts zeigt es an."""
    t, x, y, tid = _strom()
    kn = knoten_bauen(t, x, y, tid, min_events=1)
    a = stichprobe(kn, t, x, y, n_events=4, seed=0)
    b = stichprobe(kn, t, x, y, n_events=4, seed=0)
    c = stichprobe(kn, t, x, y, n_events=4, seed=1)
    np.testing.assert_array_equal(a[0], b[0])
    assert not np.array_equal(a[0], c[0]), "Seed ohne Wirkung"