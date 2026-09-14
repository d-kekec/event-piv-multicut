#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kennzahlen der Bahn-Ebene.

Ohne Datensatz: alle drei Funktionen sind rein. Geprueft wird, was still
falsch sein kann -- welche Punkte in den Median gehen, welches Vorzeichen
du_signiert traegt, und dass die Reinheitsspalten dieselbe Bahn anders
zaehlen.
"""
import numpy as np
import pytest

from eventpiv.bewerten.evaluate_trajektorien import (bahn_metriken,
                                                     gt_bahnen,
                                                     trajektorien_metriken)


def _boxen(n=40, dt=16667.0, v=(0.05, -0.02), ids=(7,)):
    """Boxen auf dem GT-Raster, exakte Gerade. (x,y) ist die ECKE."""
    zeilen = []
    for k in ids:
        t = np.arange(n) * dt
        for j in range(n):
            zeilen.append((t[j], 100.0 + v[0] * t[j] / 1e3 - 5.5,
                           200.0 + v[1] * t[j] / 1e3 - 5.5,
                           11.0, 11.0, k))
    return np.array(zeilen, dtype=[("t", "<i8"), ("x", "<f4"),
                                   ("y", "<f4"), ("w", "<f4"),
                                   ("h", "<f4"), ("track_id", "<u4")])


def test_gt_bahnen_trifft_mitte_und_geschwindigkeit():
    """(x,y) ist die Ecke, die Mitte also x + w/2. Ein Vorzeichenfehler
    dort verschoebe jede Kennzahl um 5,5 px."""
    g = gt_bahnen(_boxen(), gt_fenster=5)
    assert set(g) == {7}
    t, cx, cy, gu, gv = g[7]
    np.testing.assert_allclose(cx, 100.0 + 0.05 * t / 1e3, atol=1e-4)
    np.testing.assert_allclose(cy, 200.0 - 0.02 * t / 1e3, atol=1e-4)
    ok = np.isfinite(gu)
    np.testing.assert_allclose(gu[ok], 0.05e-3, rtol=1e-5)
    np.testing.assert_allclose(gv[ok], -0.02e-3, rtol=1e-5)


def test_gt_bahnen_laesst_kurze_partikel_aus():
    """Unter 2*gt_fenster+1 Boxen waere die Geschwindigkeit ueberwiegend
    Rasterrauschen."""
    assert gt_bahnen(_boxen(n=8), gt_fenster=5) == {}
    assert set(gt_bahnen(_boxen(n=8), gt_fenster=3)) == {7}


def test_bahn_metriken_vorzeichen_von_du():
    """du_signiert ist |u_fit| - |u_gt|: positiv heisst ZU SCHNELL."""
    n = 10
    e = np.ones(n)
    z = bahn_metriken(x=100 * e, y=200 * e, ux=0.06e-3 * e, uy=0.0 * e,
                      cx=100 * e, cy=200 * e, gu=0.05e-3 * e, gv=0.0 * e)
    assert z["du_signiert"] == pytest.approx(0.01, rel=1e-6)
    assert z["medae_xy"] == pytest.approx(0.0)
    assert z["n_punkte"] == n


def test_bahn_metriken_ohne_auswertbaren_punkt():
    n = 5
    e = np.ones(n)
    assert bahn_metriken(e, e, e, e, e, e, np.full(n, np.nan),
                         np.full(n, np.nan)) is None


def test_nan_punkte_gehen_nicht_in_den_median():
    """Die GT-Geschwindigkeit ist an den Bahnenden NaN. Ein NaN im Median
    machte jede Kennzahl zu NaN, ohne dass es auffiele."""
    n = 6
    e = np.ones(n)
    gu = np.array([np.nan, 0.05e-3, 0.05e-3, 0.05e-3, 0.05e-3, np.nan])
    z = bahn_metriken(100 * e, 200 * e, 0.05e-3 * e, 0.0 * e,
                      100 * e, 200 * e, gu, np.zeros(n))
    assert z["n_punkte"] == 4
    assert np.isfinite(z["medae_u"])


def test_rein_und_alle_werden_getrennt_gezaehlt():
    """Der Abstand zwischen beiden misst, was ein Clusterfehler kostet --
    nicht der Fit."""
    bb = _boxen(ids=(7, 9))
    t = np.arange(40) * 16667.0
    tt = np.concatenate([t, t])
    x = np.concatenate([100.0 + 0.05 * t / 1e3,
                        100.0 + 0.05 * t / 1e3 + 3.0])   # Bahn 1 daneben
    y = np.concatenate([200.0 - 0.02 * t / 1e3,
                        200.0 - 0.02 * t / 1e3])
    ux = np.full(80, 0.05e-3)
    uy = np.full(80, -0.02e-3)
    tid = np.concatenate([np.zeros(40, np.int32), np.ones(40, np.int32)])

    out = trajektorien_metriken((tt, x, y, ux, uy, tid), bb,
                                {0: (7, 1.0), 1: (9, 0.60)}, v_norm=0.102)
    assert out["n_bahnen_bewertet"] == 2
    assert out["n_bahnen_rein"] == 1
    # Die verschmolzene Bahn liegt 3 px daneben und hebt den Median.
    assert out["medae_xy_rein"] < out["medae_xy"]


def test_ohne_zuordnung_keine_bahn():
    bb = _boxen()
    t = np.arange(40) * 16667.0
    e = np.ones(40)
    out = trajektorien_metriken(
        (t, 100 * e, 200 * e, 0.05e-3 * e, 0.0 * e, np.zeros(40, np.int32)),
        bb, {}, v_norm=0.102)
    assert out == {"n_bahnen_bewertet": 0}