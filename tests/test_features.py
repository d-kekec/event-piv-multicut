#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Invariantentests fuer features.

Wie bei test_graph.py: eine offensichtlich korrekte O(n^2)-Referenz gegen
die vektorisierte Umsetzung. Kein Datensatz noetig, keine Referenzdatei --
die Aussagen bleiben gueltig, wenn sich Parameter oder Implementierung
aendern.
"""
import numpy as np
import pytest

from eventpiv.kette.features import (off_zuordnung,
                               tracklet_features)


# --- Referenz -----------------------------------------------------------

def brute_force_zuordnung(t_on, x_on, y_on, tid, t_off, x_off, y_off, s, k):
    """O(n*m)-Referenz: k naechste ON-Events, Mehrheit, bei Gleichstand
    der naechste. Genau die Regel aus dem Docstring, ohne KDTree."""
    aus, kon = [], []
    for a in range(len(t_off)):
        d = ((x_on - x_off[a])**2 + (y_on - y_off[a])**2
             + (s * (t_on - t_off[a]))**2)
        nah = np.argsort(d, kind="stable")[:k]
        stimmen = [tid[b] for b in nah]
        # Erstes Auftreten entscheidet den Gleichstand -- nah ist nach
        # Abstand sortiert.
        bester = max(dict.fromkeys(stimmen), key=stimmen.count)
        aus.append(bester)
        kon.append(stimmen.count(bester) / len(stimmen))
    return np.array(aus), np.array(kon)


@pytest.fixture
def zwei_partikel():
    """Zwei Wolken, 19 px auseinander -- der gemessene Partikelabstand.

    Je Partikel eine ON-Sichel und eine OFF-Sichel 6 px dahinter, wie im
    Datensatz. Beide bewegen sich, damit die Zeit ueberhaupt eine Rolle
    spielen KANN.
    """
    rng = np.random.default_rng(11)
    t = np.sort(rng.uniform(0, 10_000, 120))
    v = 0.0008                                   # px/µs
    teil = np.arange(120) % 2                    # Partikel 0 / 1
    x_on = 50.0 + 19.0 * teil + v * t + rng.normal(0, 1.5, 120)
    y_on = 50.0 + rng.normal(0, 1.5, 120)
    x_off = x_on - 6.0
    y_off = y_on.copy()
    return t, x_on, y_on, x_off, y_off, teil.astype(np.int64)


# --- off_zuordnung ------------------------------------------------------

@pytest.mark.parametrize("s", [0.0005, 0.1, 1.0])
@pytest.mark.parametrize("k", [4, 8])
def test_zuordnung_gegen_brute_force(zwei_partikel, s, k):
    t, xo, yo, xf, yf, tid = zwei_partikel
    a, kon = off_zuordnung(t, xo, yo, tid, t, xf, yf, s=s, k=k)
    ref_a, ref_kon = brute_force_zuordnung(t, xo, yo, tid, t, xf, yf, s, k)
    np.testing.assert_array_equal(a, ref_a)
    np.testing.assert_allclose(kon, ref_kon)


def test_konsens_liegt_zwischen_ein_k_tel_und_eins(zwei_partikel):
    """Der Sieger hat mindestens eine und hoechstens alle k Stimmen."""
    t, xo, yo, xf, yf, tid = zwei_partikel
    _, kon = off_zuordnung(t, xo, yo, tid, t, xf, yf, s=0.1, k=8)
    assert ((kon >= 1 / 8 - 1e-12) & (kon <= 1.0 + 1e-12)).all()


def test_ein_einziges_tracklet_ergibt_einstimmigkeit(zwei_partikel):
    t, xo, yo, xf, yf, _ = zwei_partikel
    eins = np.full(len(t), 7, np.int64)
    a, kon = off_zuordnung(t, xo, yo, eins, t, xf, yf, s=0.1, k=8)
    assert (a == 7).all() and np.allclose(kon, 1.0)


def test_s_ist_kein_kosmetischer_parameter(zwei_partikel):
    """Der Kern der Regel: s entscheidet, ob raeumliche oder zeitliche
    Naehe gewinnt. Waeren beide Ergebnisse gleich, waere die Messreihe
    vom 09.08. sinnlos gewesen."""
    t, xo, yo, xf, yf, tid = zwei_partikel
    klein, _ = off_zuordnung(t, xo, yo, tid, t, xf, yf, s=0.0005, k=8)
    gross, _ = off_zuordnung(t, xo, yo, tid, t, xf, yf, s=0.1, k=8)
    assert (klein != gross).any()


def test_grosses_s_ordnet_das_eigene_partikel_zu(zwei_partikel):
    """Bei s = 0,1 entspricht 1 px rund 10 µs. Die OFF-Sichel liegt zum
    SELBEN Zeitpunkt 6 px hinter der eigenen ON-Sichel, das Nachbar-
    partikel 19 px entfernt -- die Zuordnung muss also fast immer
    stimmen. Der gemessene Richtungsfehler von 1,2 deg haengt daran."""
    t, xo, yo, xf, yf, tid = zwei_partikel
    a, _ = off_zuordnung(t, xo, yo, tid, t, xf, yf, s=0.1, k=8)
    assert (a == tid).mean() > 0.9


def test_leere_eingaben():
    leer = np.empty(0)
    a, kon = off_zuordnung(leer, leer, leer, np.empty(0, np.int64),
                           leer, leer, leer, s=0.1)
    assert len(a) == 0 and len(kon) == 0
    # OFF vorhanden, ON nicht -> -1, keine Ausnahme
    a, kon = off_zuordnung(leer, leer, leer, np.empty(0, np.int64),
                           np.array([1.0]), np.array([1.0]),
                           np.array([1.0]), s=0.1)
    assert a.tolist() == [-1]


def test_k_groesser_als_die_eventzahl(zwei_partikel):
    """kk = min(k, len(t_on)) -- sonst faellt query auf Fuellwerte zurueck
    und idx enthaelt Indizes ausserhalb des Arrays."""
    t, xo, yo, xf, yf, tid = zwei_partikel
    a, kon = off_zuordnung(t[:3], xo[:3], yo[:3], tid[:3],
                           t[:5], xf[:5], yf[:5], s=0.1, k=32)
    assert len(a) == 5 and (a >= 0).all()


# --- tracklet_features --------------------------------------------------

@pytest.fixture
def kleines_fenster():
    """Drei Tracklets, eines davon mit nur zwei Events; ein Event ohne
    Tracklet (-1). OFF-Zuordnung von Hand, damit die Erwartung nachrechenbar
    bleibt."""
    tid = np.array([0, 0, 0, 1, 1, 2, 2, -1], np.int64)
    t = np.array([10., 20., 30., 11., 21., 12., 22., 40.])
    x = np.array([1., 2., 3., 10., 12., 20., 20., 99.])
    y = np.array([1., 1., 1., 5., 5., 7., 9., 99.])
    off_t = np.array([10., 20., 11.])
    off_x = np.array([0., 1., 8.])
    off_y = np.array([1., 1., 5.])
    off_tl = np.array([0, 0, 1], np.int64)
    off_k = np.array([1.0, 0.75, 1.0])
    return tid, t, x, y, off_t, off_x, off_y, off_tl, off_k


def test_zentroide_und_huelle(kleines_fenster):
    tid, t, x, y, ot, ox, oy, otl, ok = kleines_fenster
    f = tracklet_features(t, x, y, tid, ot, ox, oy, otl, ok, fenster_idx=3)
    np.testing.assert_array_equal(f["tracklet_id"], [0, 1, 2])
    np.testing.assert_allclose(f["x_on"], [2.0, 11.0, 20.0])
    np.testing.assert_allclose(f["y_on"], [1.0, 5.0, 8.0])
    np.testing.assert_array_equal(f["n_on"], [3, 2, 2])
    np.testing.assert_allclose(f["x_min"], [1.0, 10.0, 20.0])
    np.testing.assert_allclose(f["t_max"], [30.0, 21.0, 22.0])
    assert (f["fenster_idx"] == 3).all()


def test_event_ohne_tracklet_faellt_heraus(kleines_fenster):
    """tracklet_id = -1 heisst kantenloser Knoten. Das Event bei (99, 99)
    wuerde jeden Zentroid und jede Huelle zerstoeren."""
    tid, t, x, y, ot, ox, oy, otl, ok = kleines_fenster
    f = tracklet_features(t, x, y, tid, ot, ox, oy, otl, ok, fenster_idx=0)
    assert -1 not in f["tracklet_id"].tolist()
    assert f["x_max"].max() < 90.0


def test_eigenwerte_gegen_numpy(kleines_fenster):
    """lam1/lam2 aus dem Verschiebungssatz gegen np.cov. bias=True, weil
    durch n geteilt wird und nicht durch n-1."""
    tid, t, x, y, ot, ox, oy, otl, ok = kleines_fenster
    f = tracklet_features(t, x, y, tid, ot, ox, oy, otl, ok, fenster_idx=0)
    m = tid == 0
    ew = np.linalg.eigvalsh(np.cov(np.vstack([x[m], y[m]]), bias=True))
    np.testing.assert_allclose([f["lam2"][0], f["lam1"][0]], ew, atol=1e-9)


def test_lam1_nie_kleiner_als_lam2_und_nie_negativ(zwei_partikel):
    t, xo, yo, xf, yf, tid = zwei_partikel
    f = tracklet_features(t, xo, yo, tid, t, xf, yf,
                          np.zeros(len(t), np.int64),
                          np.ones(len(t)), fenster_idx=0)
    assert (f["lam1"] >= f["lam2"]).all()
    assert (f["lam2"] >= -1e-9).all()


def test_kollineare_wolke_hat_lam2_null():
    """Eine Gerade hat keine Ausdehnung quer dazu."""
    tid = np.zeros(5, np.int64)
    t = np.arange(5.0)
    x = np.arange(5.0)
    y = 2.0 * np.arange(5.0)
    f = tracklet_features(t, x, y, tid, np.empty(0), np.empty(0),
                          np.empty(0), np.empty(0, np.int64), np.empty(0),
                          fenster_idx=0)
    assert f["lam2"][0] == pytest.approx(0.0, abs=1e-9)


def test_zu_wenige_punkte_geben_nan():
    """Zwei Punkte spannen keine Flaeche auf -- lam2 waere exakt null und
    der Quotient unendlich."""
    f = tracklet_features(np.array([1., 2.]), np.array([1., 5.]),
                          np.array([1., 1.]), np.zeros(2, np.int64),
                          np.empty(0), np.empty(0), np.empty(0),
                          np.empty(0, np.int64), np.empty(0), fenster_idx=0)
    assert np.isnan(f["lam1"][0]) and np.isnan(f["lam2"][0])


def test_tracklet_ohne_off_events_bekommt_nan(kleines_fenster):
    """Tracklet 2 hat keine OFF-Zuordnung. NAN statt 0.0: ein Zentroid bei
    (0, 0) saehe wie eine Messung aus, und D zeigte quer durchs Bild."""
    tid, t, x, y, ot, ox, oy, otl, ok = kleines_fenster
    f = tracklet_features(t, x, y, tid, ot, ox, oy, otl, ok, fenster_idx=0)
    assert f["n_off"].tolist() == [2, 1, 0]
    assert np.isnan(f["x_off"][2]) and np.isnan(f["off_konsens"][2])
    np.testing.assert_allclose(f["x_off"][:2], [0.5, 8.0])


def test_off_zuordnung_auf_fremde_id_wird_verworfen(kleines_fenster):
    """off_tracklet kann IDs tragen, die in diesem Fenster nicht vorkommen
    -- etwa nach einem Filter. Sie duerfen keinen Zentroid verfaelschen."""
    tid, t, x, y, ot, ox, oy, _, ok = kleines_fenster
    fremd = np.array([0, 0, 999], np.int64)
    f = tracklet_features(t, x, y, tid, ot, ox, oy, fremd, ok, fenster_idx=0)
    assert f["n_off"].tolist() == [2, 0, 0]


def test_gt_felder_nur_mit_gt(kleines_fenster):
    tid, t, x, y, ot, ox, oy, otl, ok = kleines_fenster
    f = tracklet_features(t, x, y, tid, ot, ox, oy, otl, ok, fenster_idx=0)
    assert "gt_track_id" not in f


def test_gt_mehrheit_und_anteil(kleines_fenster):
    """Tracklet 0 ist verschmolzen: zwei Events von Partikel 5, eines von
    7. gt_anteil = 2/3 ist genau die Zahl, an der die Labelfrage von
    Stufe 2 haengt (TODO §C, teilverschmolzene Paare)."""
    tid, t, x, y, ot, ox, oy, otl, ok = kleines_fenster
    gt = np.array([5, 5, 7, 3, 3, 4, 4, 0], np.int64)
    rausch = np.zeros(8, bool)
    f = tracklet_features(t, x, y, tid, ot, ox, oy, otl, ok, fenster_idx=0,
                          track_id=gt, status_noise=rausch)
    np.testing.assert_array_equal(f["gt_track_id"], [5, 3, 4])
    np.testing.assert_allclose(f["gt_anteil"], [2 / 3, 1.0, 1.0])
    np.testing.assert_array_equal(f["n_gt"], [2, 1, 1])


def test_rauschen_zaehlt_beim_gt_label_nicht_mit(kleines_fenster):
    """Rauschen traegt track_id = -1. Zaehlte es mit, waere ein Tracklet
    mit viel Rauschen 'Partikel -1'."""
    tid, t, x, y, ot, ox, oy, otl, ok = kleines_fenster
    gt = np.array([-1, -1, 7, 3, 3, 4, 4, 0], np.int64)
    rausch = np.array([1, 1, 0, 0, 0, 0, 0, 0], bool)
    f = tracklet_features(t, x, y, tid, ot, ox, oy, otl, ok, fenster_idx=0,
                          track_id=gt, status_noise=rausch)
    assert f["gt_track_id"][0] == 7 and f["n_gt"][0] == 1


def test_alle_felder_gleich_lang_und_nach_id_sortiert(zwei_partikel):
    """Die Tabelle wird spaltenweise gespeichert -- eine abweichende Laenge
    verschoebe stillschweigend jede Zeile danach."""
    t, xo, yo, xf, yf, tid = zwei_partikel
    f = tracklet_features(t, xo, yo, tid, t, xf, yf,
                          np.zeros(len(t), np.int64), np.ones(len(t)),
                          fenster_idx=0)
    n = len(f["tracklet_id"])
    assert all(len(v) == n for v in f.values())
    assert (np.diff(f["tracklet_id"]) > 0).all()


def test_leeres_fenster_gibt_leeres_dict():
    leer = np.empty(0)
    assert tracklet_features(leer, leer, leer, np.empty(0, np.int64),
                             leer, leer, leer, np.empty(0, np.int64), leer,
                             fenster_idx=0) == {}


def test_features_slug_haengt_an_beiden_feldern():
    """Zwei Parametersaetze duerfen nicht auf demselben Pfad landen --
    dieselbe Falle wie bei Labeling (Journal 08.08.)."""
    from eventpiv.kern.runconfig import Features, slug
    assert slug(Features()) != slug(Features(off_s=0.15))
    assert slug(Features()) != slug(Features(off_k=16))


def test_features_default_ist_der_gemessene_wert():
    """off_s = 0,1 ist das Optimum aus der Messreihe vom 09.08. auf run76
    und run32. Faellt der Test, wurde der Wert ohne Messung geaendert."""
    from eventpiv.kern.runconfig import Features
    assert Features().off_s == 0.1 and Features().off_k == 8