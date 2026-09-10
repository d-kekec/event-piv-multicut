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