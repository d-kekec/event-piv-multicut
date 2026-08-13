#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ist die Fixture ihrer Aufgabe gewachsen?

Kein Test des Produktivcodes, sondern der Testgrundlage. Eine Fixture, die
still etwas anderes erzeugt als beabsichtigt, macht jeden Test daraufhin
wertlos -- und das faellt nirgends auf.
"""
import numpy as np
import pytest

import mini_dataset as md
from eventpiv.kern.data import load_events
from eventpiv.kern.graph import neighborhood_pairs
from eventpiv.kern.status import STATUS_AMBIG, STATUS_CLEAN, STATUS_NOISE
from eventpiv.labels import assign_track_ids
from eventpiv.psee_io.psee_loader import PSEELoader


def _gelabelt(mini_datensatz, run_nr):
    ev = load_events("train", run_nr)
    bbox = np.load(mini_datensatz.data / "train" / f"run{run_nr}_bbox.npy")
    tid, st = assign_track_ids(ev["t"], ev["x"], ev["y"], bbox)
    return ev, tid, st


def test_rundreise_durch_das_dat_format(mini_datensatz):
    """Was geschrieben wurde, liest PSEELoader exakt zurueck.

    Die eigentliche Pruefung des selbstgebauten Kopfes: schlaegt sie fehl,
    ist das Format falsch verstanden und nicht der Code kaputt.
    """
    bahnen = md.SZENEN[1][1]()
    t, x, y, p = md.events_array(bahnen, seed=1)
    xi = np.clip(np.rint(x), 0, md.BREITE - 1).astype(np.int64)
    yi = np.clip(np.rint(y), 0, md.HOEHE - 1).astype(np.int64)

    ld = PSEELoader(str(mini_datensatz.data / "train" / "run1.dat"))
    ev = ld.load_n_events(ld.event_count())

    assert ld.event_count() == len(t)
    np.testing.assert_array_equal(ev["t"].astype(np.int64), t.astype(np.int64))
    np.testing.assert_array_equal(ev["x"].astype(np.int64), xi)
    np.testing.assert_array_equal(ev["y"].astype(np.int64), yi)
    np.testing.assert_array_equal(ev["p"].astype(np.int64), p.astype(np.int64))


def test_fenster_ist_ausschnitt_des_ganzen_stroms(mini_datensatz):
    """load_events mit t0/dt gegen den vollen Strom.

    Deckt den Pfad seek_time + load_delta_t ab -- genau die Zeile, die ein
    Doku-Commit am 04.08. entfernt hat, ohne dass pytest es merkte.
    """
    voll = load_events("train", 1)
    fenster = load_events("train", 1, t0=200_000, dt=20_000)
    m = (voll["t"] >= 200_000) & (voll["t"] < 220_000)
    np.testing.assert_array_equal(fenster["t"], voll["t"][m])
    np.testing.assert_array_equal(fenster["x"], voll["x"][m])


def test_zeit_ist_sortiert(mini_datensatz):
    """collect_pairs bricht auf unsortiertem t ab -- hier faellt es frueher auf."""
    t = load_events("train", 1)["t"]
    assert (np.diff(t.astype(np.int64)) >= 0).all()


def test_polaritaetsversatz_ist_die_bewegungsrichtung(mini_datensatz):
    """|D| zwischen OFF- und ON-Zentroid betraegt rund 6 px.

    Die Groesse, aus der Stufe 2 Richtung und Betrag der Geschwindigkeit
    zieht. Ohne sie traefe ON-only in Phase 2 auf eine Struktur, die der
    echte Datensatz nicht hat.
    """
    ev, tid, st = _gelabelt(mini_datensatz, 3)     # sauberer Run
    m = (st == STATUS_CLEAN) & (tid == 1)
    on, off = m & (ev["p"] == 1), m & (ev["p"] == 0)
    d = np.hypot(ev["x"][on].mean() - ev["x"][off].mean(),
                 ev["y"][on].mean() - ev["y"][off].mean())
    assert 5.0 < d < 7.0, f"|D| = {d:.2f} px"


@pytest.mark.parametrize("run_nr", [1, 2])
def test_kreuzung_erzeugt_alle_drei_statusklassen(mini_datensatz, run_nr):
    _, tid, st = _gelabelt(mini_datensatz, run_nr)
    assert set(np.unique(st)) == {STATUS_CLEAN, STATUS_AMBIG, STATUS_NOISE}
    np.testing.assert_array_equal(tid == -1, st == STATUS_NOISE)


@pytest.mark.parametrize("run_nr", [3, 4])
def test_sauberer_run_hat_keine_mehrdeutigen_events(mini_datensatz, run_nr):
    """r_ambig kann nur feuern, wenn zwei Boxzentren naeher als
    2*r_assign + r_ambig = 18 px liegen. Hier sind es ueber 250 px."""
    _, _, st = _gelabelt(mini_datensatz, run_nr)
    assert not (st == STATUS_AMBIG).any()


def test_rauschanteil_liegt_bei_fuenf_prozent(mini_datensatz):
    """Wie am echten Datensatz gemessen (5,04 % auf run3)."""
    _, _, st = _gelabelt(mini_datensatz, 3)
    assert 0.04 < (st == STATUS_NOISE).mean() < 0.06


def test_nur_die_kreuzung_liefert_nein_paare(mini_datensatz):
    """Die physikalische Aussage, auf der die Sammelstrategie ruht.

    Weit getrennte Partikel erzeugen zwischen sich GAR KEINE Paare -- der
    Abstand ist groesser als r_max. Ein sauberer Run liefert deshalb null
    NEIN, nicht wenige. Hier als Eigenschaft der FIXTURE geprueft: taete
    sie es nicht, waere sie fuer Stufe 1 unbrauchbar.
    """
    def nein(run_nr):
        ev, tid, st = _gelabelt(mini_datensatz, run_nr)
        i, j = neighborhood_pairs(ev["t"].astype(np.float64),
                                  ev["x"].astype(np.float64),
                                  ev["y"].astype(np.float64),
                                  alpha=10_000, r_max=5.0, geom="cylinder")
        beide = (st[i] == STATUS_CLEAN) & (st[j] == STATUS_CLEAN)
        return int((tid[i[beide]] != tid[j[beide]]).sum())

    assert nein(3) == 0
    assert nein(1) > 100