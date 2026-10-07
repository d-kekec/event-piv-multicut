#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kennzahlen der Bahn-Ebene.

Ohne Datensatz: alle Funktionen sind rein. Geprueft wird, was still
falsch sein kann -- welche Punkte in den Median gehen, welches Vorzeichen
du_signiert traegt, dass die Reinheitsspalten dieselbe Bahn anders
zaehlen, und fuer die STELLA-Fassung Zuordnung, Ausreisserfilter,
Abdeckung und Pooling.
"""
import numpy as np
import pytest

from eventpiv.bewerten.evaluate_trajektorien import (bahn_metriken,
                                                     gt_bahnen,
                                                     stella_fehler,
                                                     stella_kennzahlen,
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
    """Die verschmolzene Bahn zaehlt nur unter "alle"."""
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


# --- STELLA §4.1 ------------------------------------------------------
def _boxen_ms(bahnen, w=4.0):
    """Boxen auf einem 1-ms-Raster wie STELLA, Bewegung entlang x.
    bahnen: {track_id: (n, x0, y0, vx)}, vx in px/ms. (x,y) ist die Ecke."""
    zeilen = []
    for k, (n, x0, y0, vx) in bahnen.items():
        for j in range(n):
            zeilen.append((j * 1000, x0 + vx * j - w / 2, y0 - w / 2,
                           w, w, k))
    return np.array(zeilen, dtype=[("t", "<i8"), ("x", "<f4"),
                                   ("y", "<f4"), ("w", "<f4"),
                                   ("h", "<f4"), ("track_id", "<u4")])


def _fit_ms(bahnen):
    """Fitspalten auf demselben Raster. bahnen: [(tid, j0, n, x0, y0, vx)],
    die Bahn beginnt bei j0 ms; u in px/µs wie aus fitte_run."""
    t, x, y, ux, uy, tid = [], [], [], [], [], []
    for k, j0, n, x0, y0, vx in bahnen:
        j = np.arange(j0, j0 + n, dtype=np.float64)
        t.append(j * 1000.0)
        x.append(x0 + vx * j)
        y.append(np.full(n, y0))
        ux.append(np.full(n, vx * 1e-3))
        uy.append(np.zeros(n))
        tid.append(np.full(n, k, np.int32))
    return tuple(np.concatenate(v) for v in (t, x, y, ux, uy, tid))


def test_stella_ordnet_unter_r_zuordnung_zu():
    """Unter 5 px zugeordnet, darueber nicht -- dann fehlt der Punkt in
    jeder MedAE und zaehlt gegen t_covered."""
    bb = _boxen_ms({1: (40, 100.0, 200.0, 0.5)})
    nah = stella_fehler(_fit_ms([(0, 0, 40, 100.0, 203.0, 0.5)]), bb)
    np.testing.assert_allclose(nah["d_xy"], 3.0, atol=1e-4)
    assert stella_kennzahlen(nah)["t_covered"] == pytest.approx(1.0)
    fern = stella_kennzahlen(
        stella_fehler(_fit_ms([(0, 0, 40, 100.0, 206.0, 0.5)]), bb))
    assert fern["n_punkte_zugeordnet"] == 0
    assert fern["t_covered"] == 0.0
    assert np.isnan(fern["medae_xy"])


def test_stella_waehlt_das_naechste_partikel():
    """Zwei Partikel 8 px auseinander: der Bahnpunkt gehoert dem naeheren,
    und nur dessen Punkte gelten als abgedeckt."""
    bb = _boxen_ms({1: (40, 100.0, 200.0, 0.5), 2: (40, 100.0, 208.0, 0.5)})
    f = stella_fehler(_fit_ms([(0, 0, 40, 100.0, 206.0, 0.5)]), bb)
    np.testing.assert_allclose(f["d_xy"], 2.0, atol=1e-4)
    assert (f["n_gt_gedeckt"], f["n_gt_punkte"]) == (40, 80)


def test_stella_verwirft_die_ganze_bahn_mit_ausreisser():
    """Ein Punkt ueber dem 2,5-Fachen des Medians von |u| nimmt die ganze
    Bahn heraus, auch ihre unauffaelligen Punkte."""
    bb = _boxen_ms({1: (40, 100.0, 200.0, 0.5), 2: (40, 100.0, 300.0, 0.5),
                    3: (40, 100.0, 400.0, 0.5)})
    spalten = _fit_ms([(0, 0, 40, 100.0, 200.0, 0.5),
                       (1, 0, 40, 100.0, 300.0, 0.5),
                       (2, 0, 40, 100.0, 401.0, 0.5)])
    spalten[3][100] = 2.0e-3                  # Bahn 2: einmal 2 px/ms
    k = stella_kennzahlen(stella_fehler(spalten, bb))
    assert k["verworfen_frac"] == pytest.approx(1 / 3)
    assert k["n_tracks"] == 2
    assert k["medae_xy"] == pytest.approx(0.0, abs=1e-4)
    assert k["t_covered"] == pytest.approx(2 / 3)


def test_stella_poolt_ueber_punkte_nicht_ueber_bahnen():
    """30 Punkte mit 1 px, 10 mit 3 px: der Median ueber alle Punkte ist 1,
    der Median der beiden Bahn-Mediane laege bei 2."""
    bb = _boxen_ms({1: (40, 100.0, 200.0, 0.5), 2: (40, 100.0, 300.0, 0.5)})
    k = stella_kennzahlen(stella_fehler(
        _fit_ms([(0, 0, 30, 100.0, 201.0, 0.5),
                 (1, 0, 10, 100.0, 303.0, 0.5)]), bb))
    assert k["medae_xy"] == pytest.approx(1.0, abs=1e-4)
    assert k["p25_xy"] == pytest.approx(1.0, abs=1e-4)
    assert k["p75_xy"] > k["medae_xy"]


def test_stella_t_covered_und_n_tracks():
    """Zwei Bruchstuecke ueber einem Partikel: 35 von 40 GT-Punkten
    abgedeckt, zwei Bahnen fuer eine GT-Bahn."""
    bb = _boxen_ms({1: (40, 100.0, 200.0, 0.5)})
    k = stella_kennzahlen(stella_fehler(
        _fit_ms([(0, 0, 20, 100.0, 200.0, 0.5),
                 (1, 25, 15, 100.0, 200.0, 0.5)]), bb))
    assert k["t_covered"] == pytest.approx(35 / 40)
    assert k["n_tracks_je_gt"] == pytest.approx(2.0)


def test_stella_kurze_gt_bahn_zaehlt_im_ort_nicht_in_u():
    """Unter 2*gt_fenster+1 Boxen traegt die GT keine Geschwindigkeit: der
    Ort wird bewertet, die Geschwindigkeit nicht, die Bahn zaehlt mit."""
    bb = _boxen_ms({1: (8, 100.0, 200.0, 0.5)})
    f = stella_fehler(_fit_ms([(0, 0, 8, 100.0, 201.0, 0.5)]), bb)
    assert (len(f["d_xy"]), len(f["d_u"]), f["n_gt"]) == (8, 0, 1)


def test_stella_laengsfehler_traegt_das_vorzeichen():
    """Voraus positiv, hinterher negativ, quer null. MedAE_u sieht keinen
    der drei Faelle -- die Geschwindigkeit stimmt jedes Mal."""
    bb = _boxen_ms({1: (40, 100.0, 200.0, 0.5)})
    for dx, dy, soll in ((1.0, 0.0, 1.0), (-1.0, 0.0, -1.0), (0.0, 1.0, 0.0)):
        k = stella_kennzahlen(stella_fehler(
            _fit_ms([(0, 0, 40, 100.0 + dx, 200.0 + dy, 0.5)]), bb))
        assert k["laengs_median"] == pytest.approx(soll, abs=1e-3)
        assert k["medae_u"] == pytest.approx(0.0, abs=1e-6)


def test_stella_poolt_ueber_runs():
    """Zwei Runs: die Zaehler addieren sich, der Median laeuft ueber alle
    Punkte beider Runs."""
    bb = _boxen_ms({1: (40, 100.0, 200.0, 0.5)})
    a = stella_fehler(_fit_ms([(0, 0, 40, 100.0, 201.0, 0.5)]), bb)
    b = stella_fehler(_fit_ms([(0, 0, 10, 100.0, 203.0, 0.5)]), bb)
    k = stella_kennzahlen([a, b])
    assert k["medae_xy"] == pytest.approx(1.0, abs=1e-4)
    assert (k["n_gt"], k["n_tracks"]) == (2, 2)
    assert k["t_covered"] == pytest.approx(50 / 80)


def test_stella_ungruppierte_ids_brechen_ab():
    """Stuende eine Bahn in zwei Bloecken, zaehlte sie doppelt, und der
    Ausreisserfilter naehme nur den Block mit dem Ausreisser heraus."""
    bb = _boxen_ms({1: (40, 100.0, 200.0, 0.5)})
    spalten = _fit_ms([(0, 0, 10, 100.0, 200.0, 0.5),
                       (1, 10, 10, 100.0, 200.0, 0.5),
                       (0, 20, 10, 100.0, 200.0, 0.5)])
    with pytest.raises(ValueError, match="gruppiert"):
        stella_fehler(spalten, bb)