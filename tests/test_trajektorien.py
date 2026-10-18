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
    np.testing.assert_array_equal(tr.trajektorie_id, np.array(sorted(ref)))
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
    assert -1 not in tr.trajektorie_id
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
    i = int(np.flatnonzero(tr.trajektorie_id == 1)[0])
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
    Solvers -- trajektorie_id traegt die Bruecke zurueck."""
    tid = np.array([5, 900, 5, -1, 900], np.int32)
    tr = gruppiere(tid, len(tid))
    np.testing.assert_array_equal(tr.trajektorie_id, [5, 900])
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


def test_versatz_korrektur_schiebt_entgegen_der_bewegung():
    """Der Betrag haengt an |u|, die Richtung an ihrem Vorzeichen."""
    from eventpiv.kette.trajektorien import versatz_korrektur

    x = np.array([100.0, 100.0])
    y = np.array([200.0, 200.0])
    u = np.array([0.05e-3, -0.05e-3])      # px/µs, also 0,05 px/ms
    v = np.zeros(2)
    xk, yk = versatz_korrektur(x, y, u, v, c0=0.88, c1=29.1)
    schub = 0.88 + 29.1 * 0.05
    np.testing.assert_allclose(xk, [100.0 - schub, 100.0 + schub])
    np.testing.assert_allclose(yk, y)


def test_versatz_korrektur_ist_abschaltbar():
    from eventpiv.kette.trajektorien import versatz_korrektur

    x, y = np.array([1.0, 2.0]), np.array([3.0, 4.0])
    u, v = np.array([1e-3, 0.0]), np.array([0.0, 2e-3])
    xk, yk = versatz_korrektur(x, y, u, v, c0=0.0, c1=0.0)
    np.testing.assert_array_equal(xk, x)
    np.testing.assert_array_equal(yk, y)


def test_fitte_run_laesst_kurze_bahnen_aus():
    """Der Filter braucht init_us allein zum Einschwingen."""
    from eventpiv.kette.trajektorien import fitte_run, gruppiere

    n = 4000
    t = np.arange(n, dtype=np.float64) * 100.0        # 0..400 ms
    x = 100.0 + 0.05 * t / 1e3
    y = 200.0 - 0.02 * t / 1e3
    lang = np.zeros(n, np.int32)
    lang[t > 300_000.0] = 1                            # zweite Bahn: 100 ms
    tr = gruppiere(lang, n)

    tf, xf, yf, uf, vf, idf = fitte_run(
        tr, t, x, y, r_mess=8.0, q_c=20.0, c0=0.0, c1=0.0,
        dt_aus=10_000.0, min_dauer_us=200_000.0)
    assert set(np.unique(idf).tolist()) == {0}
    assert len(tf) == len(xf) == len(yf) == len(uf) == len(vf) == len(idf)
    assert (np.diff(tf) > 0).all()


def test_fitte_run_ohne_taugliche_bahn():
    from eventpiv.kette.trajektorien import fitte_run, gruppiere

    t, x, y = _gerade(n=200, dt=50.0)                  # 10 ms
    tr = gruppiere(np.zeros(200, np.int32), 200)
    aus = fitte_run(tr, t, x, y, r_mess=8.0, q_c=20.0, c0=0.0, c1=0.0,
                    dt_aus=1e3, min_dauer_us=1e6)
    assert all(len(a) == 0 for a in aus)


def test_glaetter_trifft_die_exakte_gerade_ebenfalls():
    """Auf einer rauschfreien Geraden darf der Glaetter nichts
    verschlechtern -- die Bahn ist bereits optimal."""
    from eventpiv.kette.trajektorien import kalman_bahn

    t, x, y = _gerade()
    ta, xa, ya, ua, va = kalman_bahn(t, x, y, r_mess=8.0, q_c=20.0,
                                     dt_aus=10_000.0, glaetten=True)
    soll_x = 100.0 + 0.05 * ta / 1e3
    assert np.abs(xa - soll_x).max() < 0.05
    np.testing.assert_allclose(ua, 0.05e-3, rtol=2e-2)


def test_glaetter_verbessert_den_bahnanfang():
    """Der kausale Filter hat am Anfang keine Vergangenheit. Der Glaetter
    nutzt dort die Zukunft und muss naeher an der Wahrheit liegen."""
    from eventpiv.kette.trajektorien import kalman_bahn

    rng = np.random.default_rng(0)
    t, x, y = _gerade(n=8000, dt=30.0)
    xr = x + rng.normal(0, 2.8, len(t))
    yr = y + rng.normal(0, 2.8, len(t))

    _, xa, _, _, _ = kalman_bahn(t, xr, yr, r_mess=8.0, q_c=20.0,
                                 dt_aus=5_000.0, glaetten=False)
    tb, xb, _, _, _ = kalman_bahn(t, xr, yr, r_mess=8.0, q_c=20.0,
                                  dt_aus=5_000.0, glaetten=True)
    soll = 100.0 + 0.05 * tb / 1e3
    k = max(3, len(tb) // 5)            # erstes Fuenftel
    assert (np.abs(xb[:k] - soll[:k]).mean()
            < np.abs(xa[:k] - soll[:k]).mean())


def test_a_unbekannt_erreicht_den_filter():
    """Die Streuung der Startbeschleunigung wirkt: auf einer stark
    gekruemmten Bahn trifft ein grosszuegiger Wert den Bahnanfang, ein zu
    kleiner haelt die Beschleunigung zu lange bei null. Die Vorgabe
    bleibt 100 -- ohne das Feld rechnet der Filter wie bisher."""
    from eventpiv.kette.trajektorien import kalman_bahn

    n = 20_000
    t = np.arange(n, dtype=np.float64) * 10.0          # 200 ms
    ts = t * 1e-6
    x = 100.0 + 1000.0 * ts + 0.5 * 5000.0 * ts ** 2   # a = 5000 px/s^2
    y = np.full(n, 200.0)
    kw = dict(r_mess=3.0, q_c=1e6, dt_aus=1_000.0, init_us=5_000.0,
              glaetten=True)
    klein = kalman_bahn(t, x, y, a_unbekannt=100.0, **kw)
    gross = kalman_bahn(t, x, y, a_unbekannt=1e4, **kw)
    soll = (1000.0 + 5000.0 * klein[0] * 1e-6) * 1e-6  # px/µs
    k = 20                                             # erste 20 ms
    f_klein = np.abs(klein[3][:k] - soll[:k]).mean()
    f_gross = np.abs(gross[3][:k] - soll[:k]).mean()
    assert f_gross < f_klein / 10

    vorgabe = kalman_bahn(t, x, y, **kw)
    for a, b in zip(vorgabe, klein):
        np.testing.assert_array_equal(a, b)


def test_fit_slug_trennt_glaetter():
    """Zwei Fit-Arbeitspunkte muessen nebeneinander liegen koennen."""
    from eventpiv.kern.runconfig import Fit, slug

    assert slug(Fit(glaetten=True)) != slug(Fit(glaetten=False))
    assert "_rts" in Fit(glaetten=True).short()
    assert "_rts" not in Fit(glaetten=False).short()


def test_fit_slug_trennt_beta():
    """Fits mit und ohne Querkorrektur muessen nebeneinander liegen
    koennen; ein negatives beta schoebe in Bewegungsrichtung."""
    from eventpiv.kern.runconfig import Fit, slug

    assert slug(Fit(beta=1.31)) != slug(Fit(beta=0.0))
    assert Fit(beta=1.31, spline_eps=0.0).short().endswith("_b1.31")
    assert "_b" not in Fit(beta=0.0).short()
    with pytest.raises(ValueError, match="beta"):
        Fit(beta=-1.0)


def _wolke(n=6000, dt=50.0, v=(0.48, 0.36), sq=1.5, sl=0.9, vor=2.0,
           seed=0):
    """Eventwolke um eine gerade Bahn, schraeg zu den Achsen: vor px
    voraus, Streuung sl laengs und sq quer. t in µs, v in px/ms.
    Laengs wird vor quer gezogen -- gleicher seed, gleiche Querwerte."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) * dt
    b = np.hypot(*v)
    ex, ey = v[0] / b, v[1] / b
    laengs = vor + rng.normal(0.0, sl, n)
    quer = rng.normal(0.0, sq, n)
    x = 100.0 + v[0] * t / 1e3 + laengs * ex - quer * ey
    y = 200.0 + v[1] * t / 1e3 + laengs * ey + quer * ex
    return t, x, y


def test_quer_streuung_misst_nur_quer():
    """Gegen die exakte Bahn: der Wert ist sigma_quer, und Laengsversatz
    wie Laengsstreuung gehen nicht ein."""
    from eventpiv.kette.trajektorien import quer_streuung

    ta = np.arange(0.0, 299_000.0, 1_000.0)
    fit = (ta, 100.0 + 0.48 * ta / 1e3, 200.0 + 0.36 * ta / 1e3,
           np.full(len(ta), 0.48e-3), np.full(len(ta), 0.36e-3))
    a = quer_streuung(*_wolke(sl=0.9, vor=2.0), *fit)
    b = quer_streuung(*_wolke(sl=2.5, vor=0.0), *fit)
    assert abs(a - 1.5) < 0.05
    assert np.isclose(a, b, rtol=1e-9)


def test_quer_streuung_ohne_auswertbare_events_ist_nan():
    from eventpiv.kette.trajektorien import quer_streuung

    t, x, y = np.array([0.0, 100.0]), np.zeros(2), np.zeros(2)
    ta = np.array([5_000.0, 6_000.0])
    eins, null = np.full(2, 1e-3), np.zeros(2)
    assert np.isnan(quer_streuung(t, x, y, ta, null, null, eins, null))
    assert np.isnan(quer_streuung(t, x, y, ta[:0], null[:0], null[:0],
                                  eins[:0], null[:0]))


def test_fitte_run_beta_schiebt_um_beta_mal_querstreuung():
    """beta = 0 ist der alte Weg, Bit fuer Bit. beta > 0 schiebt jede
    Bahn um beta*sigma_quer gegen die Bewegung, und der Fit schluckt die
    Querstreuung nicht."""
    from eventpiv.kette.trajektorien import (fitte_run, gruppiere,
                                             quer_streuung)

    t, x, y = _wolke()
    tr = gruppiere(np.zeros(len(t), np.int32), len(t))
    kw = dict(r_mess=3.0, q_c=1e7, c0=0.0, c1=0.0, dt_aus=1_000.0,
              min_dauer_us=10_000.0, init_us=5_000.0, glaetten=True,
              a_unbekannt=1e4)
    vorher = fitte_run(tr, t, x, y, **kw)
    beta0 = fitte_run(tr, t, x, y, beta=0.0, **kw)
    for p, q in zip(vorher, beta0):
        np.testing.assert_array_equal(p, q)

    mit = fitte_run(tr, t, x, y, beta=1.3, **kw)
    sq = quer_streuung(t, x, y, *vorher[:5])
    assert abs(sq - 1.5) < 0.05
    b = np.hypot(vorher[3], vorher[4])
    np.testing.assert_allclose(vorher[1] - mit[1], 1.3 * sq * vorher[3] / b,
                               rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(vorher[2] - mit[2], 1.3 * sq * vorher[4] / b,
                               rtol=1e-9, atol=1e-12)


def test_fit_slug_trennt_spline():
    """Fits mit und ohne Spline muessen nebeneinander liegen koennen; eine
    negative Abweichung gibt es nicht."""
    from eventpiv.kern.runconfig import Fit, slug

    assert slug(Fit(spline_eps=0.02)) != slug(Fit(spline_eps=0.0))
    assert Fit(spline_eps=0.02).short().endswith("_s0.02")
    assert "_s" not in Fit(spline_eps=0.0).short()
    with pytest.raises(ValueError, match="spline_eps"):
        Fit(spline_eps=-0.01)


def test_spline_bahn_haelt_eps_und_u_ist_die_ableitung():
    """Der Spline weicht im quadratischen Mittel hoechstens eps ab, und u
    ist die Ableitung des ausgegebenen Orts. Am Rand rechnet np.gradient
    einseitig, verglichen wird deshalb innen."""
    from eventpiv.kette.trajektorien import spline_bahn

    t = np.arange(400, dtype=np.float64) * 1_000.0     # 1-ms-Raster
    ph = 0.004 * t / 1e3                               # 0,6 px/ms auf r = 150
    x, y = 300.0 + 150.0 * np.cos(ph), 200.0 + 150.0 * np.sin(ph)
    xs, ys, us, vs = spline_bahn(t, x, y, eps=0.02)
    for a, b in ((xs, x), (ys, y)):
        assert np.sqrt(np.mean((a - b) ** 2)) <= 0.02 * 1.001
    np.testing.assert_allclose(us[1:-1], np.gradient(xs, t)[1:-1], atol=1e-8)
    np.testing.assert_allclose(vs[1:-1], np.gradient(ys, t)[1:-1], atol=1e-8)


def _bogen(n=8000, dt=50.0, r=100.0, v=0.5, vor=2.0, sq=1.0, sl=0.7,
           seed=0):
    """Eventwolke um eine Kreisbahn, vor px voraus, Streuung sl laengs und
    sq quer. Dazu die Mitte als Funktion der Zeit: Ort in px, u in px/µs.
    t in µs, v in px/ms."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) * dt

    def mitte(tt):
        ph = v / r * tt / 1e3
        return (300.0 + r * np.cos(ph), 200.0 + r * np.sin(ph),
                -v * np.sin(ph) * 1e-3, v * np.cos(ph) * 1e-3)

    mx, my, mu, mv = mitte(t)
    ex, ey = mu / np.hypot(mu, mv), mv / np.hypot(mu, mv)
    laengs = vor + rng.normal(0.0, sl, n)
    quer = rng.normal(0.0, sq, n)
    return (t, mx + laengs * ex - quer * ey, my + laengs * ey + quer * ex,
            mitte)


_KW_BOGEN = dict(r_mess=3.0, q_c=1e7, c0=2.0, c1=0.0, dt_aus=1_000.0,
                 min_dauer_us=10_000.0, init_us=5_000.0, glaetten=True,
                 a_unbekannt=1e4)


def test_fitte_run_spline_eps_null_ist_der_alte_weg():
    from eventpiv.kette.trajektorien import fitte_run, gruppiere

    t, x, y, _ = _bogen()
    tr = gruppiere(np.zeros(len(t), np.int32), len(t))
    vorher = fitte_run(tr, t, x, y, **_KW_BOGEN)
    null = fitte_run(tr, t, x, y, spline_eps=0.0, **_KW_BOGEN)
    for p, q in zip(vorher, null):
        np.testing.assert_array_equal(p, q)


def test_fitte_run_spline_u_trifft_die_mitte_auf_dem_bogen():
    """Die Korrektur schiebt den Ort um c0 gegen die Bewegung, das u des
    Filters bleibt das der Wolke. Auf dem Bogen weicht es dadurch um
    c0*omega von der Mitte ab; die Ableitung des korrigierten Orts nicht."""
    from eventpiv.kette.trajektorien import fitte_run, gruppiere

    t, x, y, mitte = _bogen()
    tr = gruppiere(np.zeros(len(t), np.int32), len(t))
    ohne = fitte_run(tr, t, x, y, **_KW_BOGEN)
    mit = fitte_run(tr, t, x, y, spline_eps=0.02, **_KW_BOGEN)
    np.testing.assert_array_equal(ohne[0], mit[0])
    _, _, mu, mv = mitte(mit[0])
    f_ohne = np.median(np.hypot(ohne[3] - mu, ohne[4] - mv))
    f_mit = np.median(np.hypot(mit[3] - mu, mit[4] - mv))
    assert f_mit < f_ohne / 4


def test_fitte_run_spline_laesst_zu_kurze_bahnen_aus():
    """Ein kubischer Spline braucht mehr als drei Stuetzpunkte; ohne
    Spline bleibt dieselbe Bahn."""
    from eventpiv.kette.trajektorien import fitte_run, gruppiere

    t, x, y = _gerade(n=30, dt=100.0)                  # 2,9 ms
    tr = gruppiere(np.zeros(30, np.int32), 30)
    kw = dict(r_mess=8.0, q_c=20.0, c0=0.0, c1=0.0, dt_aus=1_000.0,
              min_dauer_us=1_000.0)
    assert len(fitte_run(tr, t, x, y, **kw)[0]) == 3
    assert all(len(a) == 0
               for a in fitte_run(tr, t, x, y, spline_eps=0.02, **kw))