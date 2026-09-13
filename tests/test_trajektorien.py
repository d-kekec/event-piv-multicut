#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gruppierung der Events nach Trajektorie.

Ohne Datensatz: gruppiere und bahn sind rein. Gegen eine offensichtlich
korrekte Schleifenfassung geprueft, wie test_knoten.py es vormacht.
"""
import numpy as np
import pytest

from eventpiv.kette.trajektorien import bahn, gruppiere


def _strom():
    """Zwoelf ON-Events, drei Trajektorien, drei ohne Zuordnung."""
    tid = np.array([0, 2, -1, 0, 1, 1, -1, 2, 0, 1, -1, 2], np.int32)
    n = len(tid)
    t = np.arange(n, dtype=np.float64) * 100.0
    x = np.arange(n, dtype=np.float64)
    y = np.arange(n, dtype=np.float64) * 2.0
    return t, x, y, tid


def _referenz(tid):
    return {int(k): np.flatnonzero(tid == k)
            for k in sorted(set(int(v) for v in tid if v >= 0))}


def test_gruppierung_gegen_schleifenreferenz():
    t, x, y, tid = _strom()
    tr = gruppiere(tid, len(t))
    ref = _referenz(tid)

    assert len(tr) == len(ref)
    np.testing.assert_array_equal(tr.cluster_id, np.array(sorted(ref)))
    for i, k in enumerate(sorted(ref)):
        lo, hi = tr.grenzen[i], tr.grenzen[i + 1]
        np.testing.assert_array_equal(tr.reihenfolge[lo:hi], ref[k])


def test_zeitordnung_bleibt_innerhalb_einer_bahn():
    """Der Kalman-Filter bildet sein dt aus aufeinanderfolgenden
    Messungen -- eine unsortierte Bahn ergaebe negative Zeitschritte."""
    t, x, y, tid = _strom()
    tr = gruppiere(tid, len(t))
    for i in range(len(tr)):
        tb, _, _ = bahn(tr, i, t, x, y)
        assert (np.diff(tb) > 0).all()


def test_nicht_zugeordnete_events_fallen_heraus():
    t, x, y, tid = _strom()
    tr = gruppiere(tid, len(t))
    assert -1 not in tr.cluster_id
    assert not np.isin(np.flatnonzero(tid < 0), tr.reihenfolge).any()
    assert len(tr.reihenfolge) == int((tid >= 0).sum())


def test_jedes_zugeordnete_event_genau_einmal():
    t, x, y, tid = _strom()
    tr = gruppiere(tid, len(t))
    np.testing.assert_array_equal(np.sort(tr.reihenfolge),
                                  np.flatnonzero(tid >= 0))


def test_bahn_liefert_die_richtigen_werte():
    t, x, y, tid = _strom()
    tr = gruppiere(tid, len(t))
    i = int(np.flatnonzero(tr.cluster_id == 1)[0])
    tb, xb, yb = bahn(tr, i, t, x, y)
    erwartet = np.flatnonzero(tid == 1)
    np.testing.assert_array_equal(tb, t[erwartet])
    np.testing.assert_array_equal(xb, x[erwartet])
    np.testing.assert_array_equal(yb, y[erwartet])


def test_alles_ohne_zuordnung_gibt_leer():
    t, x, y, tid = _strom()
    tr = gruppiere(np.full(len(tid), -1, np.int32), len(t))
    assert len(tr) == 0
    assert len(tr.reihenfolge) == 0


def test_laengenvertrag_bricht_ab():
    """Ein Versatz um einen Eintrag ordnete jedes Event der falschen Bahn
    zu und faellt an keiner Zahl auf."""
    t, x, y, tid = _strom()
    with pytest.raises(ValueError, match="VERTRAG VERLETZT"):
        gruppiere(tid, len(t) + 1)


def test_luecken_in_den_ids_sind_zulaessig():
    """kompakt nummeriert lueckenlos, aber das ist keine Zusicherung des
    Solvers -- cluster_id traegt die Bruecke zurueck."""
    tid = np.array([5, 900, 5, -1, 900], np.int32)
    tr = gruppiere(tid, len(tid))
    np.testing.assert_array_equal(tr.cluster_id, [5, 900])
    assert len(tr) == 2


def _gerade(n=2000, dt=50.0, v=(0.05, -0.02), start=(100.0, 200.0)):
    """Exakte Gerade, kein Rauschen. t in µs, v in px/ms."""
    t = np.arange(n, dtype=np.float64) * dt
    x = start[0] + v[0] * t / 1e3
    y = start[1] + v[1] * t / 1e3
    return t, x, y


def test_kalman_trifft_die_exakte_gerade():
    """Ohne Rauschen muss der Filter die Bahn treffen -- Position auf
    Bruchteile eines Pixels, Geschwindigkeit auf wenige Promille."""
    from eventpiv.kette.trajektorien import kalman_bahn

    t, x, y = _gerade()
    ta, xa, ya, ua, va = kalman_bahn(t, x, y, r_mess=8.0, q_c=20.0,
                                     dt_aus=10_000.0)
    assert len(ta) > 5
    soll_x = 100.0 + 0.05 * ta / 1e3
    soll_y = 200.0 - 0.02 * ta / 1e3
    assert np.abs(xa - soll_x).max() < 0.05
    assert np.abs(ya - soll_y).max() < 0.05
    # Geschwindigkeit intern px/s, Rueckgabe px/µs.
    np.testing.assert_allclose(ua, 0.05e-3, rtol=2e-2)
    np.testing.assert_allclose(va, -0.02e-3, rtol=2e-2)


def test_kalman_gleiche_zeiten_dt_aus_und_t_aus():
    """dt_aus und ein gleichwertiges t_aus muessen dasselbe liefern --
    der Ausgabeweg darf das Ergebnis nicht beruehren."""
    from eventpiv.kette.trajektorien import kalman_bahn

    t, x, y = _gerade()
    a = kalman_bahn(t, x, y, r_mess=8.0, q_c=20.0, dt_aus=10_000.0)
    b = kalman_bahn(t, x, y, r_mess=8.0, q_c=20.0, t_aus=a[0])
    for va, vb in zip(a, b):
        np.testing.assert_allclose(va, vb, rtol=1e-12, atol=1e-12)


def test_kalman_vertraegt_doppelte_zeitstempel():
    """Rund 2,8 % der Events teilen sich einen Stempel. Ohne die
    h > 0-Bedingung entstuende dort eine Vorhersage ueber dt = 0."""
    from eventpiv.kette.trajektorien import kalman_bahn

    t, x, y = _gerade(n=600)
    t[300:310] = t[300]
    ta, xa, ya, ua, va = kalman_bahn(t, x, y, r_mess=8.0, q_c=20.0,
                                     dt_aus=10_000.0)
    assert np.isfinite(xa).all() and np.isfinite(ua).all()


def test_kalman_fordert_genau_einen_ausgabeweg():
    from eventpiv.kette.trajektorien import kalman_bahn

    t, x, y = _gerade(n=100)
    with pytest.raises(ValueError, match="genau eines"):
        kalman_bahn(t, x, y, r_mess=8.0, q_c=20.0)
    with pytest.raises(ValueError, match="genau eines"):
        kalman_bahn(t, x, y, r_mess=8.0, q_c=20.0, dt_aus=1e4, t_aus=t)
    with pytest.raises(ValueError, match="aufsteigend"):
        kalman_bahn(t, x, y, r_mess=8.0, q_c=20.0, t_aus=t[::-1])


def test_kalman_bricht_bei_zu_wenigen_events_ab():
    from eventpiv.kette.trajektorien import kalman_bahn

    t, x, y = _gerade(n=5)
    with pytest.raises(ValueError, match="mindestens"):
        kalman_bahn(t, x, y, r_mess=8.0, q_c=20.0, dt_aus=1e3)