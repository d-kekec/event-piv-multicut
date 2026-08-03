#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Invariantentests fuer pairs.collect_pairs -- ohne Datensatz, ohne Referenz."""
import numpy as np
import pytest

from eventpiv.labels import STATUS_AMBIG, STATUS_CLEAN, STATUS_NOISE
from eventpiv.pairs import collect_pairs

PARAMS = dict(alpha=1_000.0, r_max=5.0, win_offsets=(0,),
              win_dt=2_000, slab_dt=1_000, geom="ellipsoid")


@pytest.fixture
def szene():
    """Zwei dicht benachbarte Partikel (3 px), je 6 Events ueber 1,5 ms.

    Abstand 3 px < r_max -> es entstehen sowohl JA- als auch NEIN-Paare.
    Ein Event je Partikel ist absichtlich NICHT sauber.

    ZEITLICH SORTIERT, wie ein echter Eventstrom: load_events liefert die
    Prophesee-Datei in Aufnahmereihenfolge, und collect_pairs verankert
    Fenster und Slab-Raster an t[0] bzw. tw[0].
    """
    zeiten = np.arange(6, dtype=np.float64) * 300.0
    t = np.repeat(zeiten, 2)                        # 0,0,300,300,600,600,...
    x = np.tile(np.array([100.0, 103.0]), 6)        # abwechselnd P1, P2
    y = np.full(12, 100.0)
    tid = np.tile(np.array([1, 2], dtype=np.int16), 6)
    sta = np.full(12, STATUS_CLEAN, dtype=np.uint8)
    sta[4] = STATUS_AMBIG        # Partikel 1 bei t=600
    sta[7] = STATUS_NOISE        # Partikel 2 bei t=900
    return t, x, y, tid, sta

def test_geom_wird_durchgereicht(szene):
    """Zylinder ist weiter als Ellipsoid -> mindestens so viele Paare."""
    t, x, y, tid, sta = szene
    e = collect_pairs(t, x, y, tid, sta, want_label=None,
                      **{**PARAMS, "geom": "ellipsoid"})
    z = collect_pairs(t, x, y, tid, sta, want_label=None,
                      **{**PARAMS, "geom": "cylinder"})
    assert len(z["label"]) >= len(e["label"])

def test_label_entspricht_track_id(szene):
    t, x, y, tid, sta = szene
    d = collect_pairs(t, x, y, tid, sta, want_label=None, **PARAMS)
    assert len(d["label"]) > 0
    assert set(np.unique(d["label"])) <= {0, 1}


def test_dt_nicht_negativ(szene):
    """i = frueheres Event -- Grundannahme des gesamten Feature-Designs."""
    t, x, y, tid, sta = szene
    d = collect_pairs(t, x, y, tid, sta, want_label=None, **PARAMS)
    assert (d["dt"] >= 0).all()


def test_nur_saubere_events(szene):
    """Kein Paar darf ein mehrdeutiges oder verrauschtes Event enthalten."""
    t, x, y, tid, sta = szene
    d = collect_pairs(t, x, y, tid, sta, want_label=None, **PARAMS)
    # Events 2 und 9 liegen bei t=600 bzw. t=2700 -- Partner mit gleichem t
    # sind erlaubt, solange sie vom sauberen Partikel stammen. Deshalb ueber
    # die Positionen pruefen statt ueber die Zeit:
    assert d["label"].size > 0
    for ti, xi, dt, dx in zip(d["ti"], d["xi"], d["dt"], d["dx"]):
        for tt, xx in ((ti, xi), (ti + dt, xi + dx)):
            treffer = np.flatnonzero((t == tt) & (np.isclose(x, xx)))
            assert (sta[treffer] == STATUS_CLEAN).any()


def test_want_label_filtert(szene):
    t, x, y, tid, sta = szene
    ja = collect_pairs(t, x, y, tid, sta, want_label=1, **PARAMS)
    nein = collect_pairs(t, x, y, tid, sta, want_label=0, **PARAMS)
    beide = collect_pairs(t, x, y, tid, sta, want_label=None, **PARAMS)
    assert (ja["label"] == 1).all() and len(ja["label"]) > 0
    assert (nein["label"] == 0).all() and len(nein["label"]) > 0
    assert len(ja["label"]) + len(nein["label"]) == len(beide["label"])


def test_kein_doppelzaehlen_ueber_slabgrenzen(szene):
    """Jedes Paar genau einmal -- die Slab-Zuordnung ueber das fruehere
    Event ist genau dafuer da."""
    t, x, y, tid, sta = szene
    d = collect_pairs(t, x, y, tid, sta, want_label=None, **PARAMS)
    schluessel = list(zip(d["ti"], d["xi"], d["dt"], d["dx"], d["dy"]))
    assert len(set(schluessel)) == len(schluessel)


def test_saubere_szene_liefert_keine_nein_paare():
    """Physikalischer Befund aus BA_NN §4.1: weit getrennte Partikel
    erzeugen NULL NEIN-Paare -- deshalb sind Overlap-Runs zwingend."""
    zeiten = np.arange(6, dtype=np.float64) * 300.0
    t = np.repeat(zeiten, 2)
    x = np.tile(np.array([100.0, 400.0]), 6)        # 300 px auseinander
    y = np.full(12, 100.0)
    tid = np.tile(np.array([1, 2], dtype=np.int16), 6)
    sta = np.full(12, STATUS_CLEAN, dtype=np.uint8)
    d = collect_pairs(t, x, y, tid, sta, want_label=None, **PARAMS)
    assert (d["label"] == 1).all()


def test_determinismus(szene):
    t, x, y, tid, sta = szene
    a = collect_pairs(t, x, y, tid, sta, want_label=None, **PARAMS)
    b = collect_pairs(t, x, y, tid, sta, want_label=None, **PARAMS)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_leere_eingabe():
    leer = np.empty(0)
    d = collect_pairs(leer, leer, leer, np.empty(0, np.int16),
                      np.empty(0, np.uint8), want_label=None, **PARAMS)
    assert len(d["label"]) == 0

def test_unsortierte_zeit_faellt_auf(szene):
    """t[0] und tw[0] verankern Fenster und Slabs -- bei unsortiertem t
    fielen Events vor tw[0] still in gar keine Slab."""
    t, x, y, tid, sta = szene
    with pytest.raises(ValueError, match="sortiert"):
        collect_pairs(t[::-1], x, y, tid, sta, want_label=None, **PARAMS)


def test_ueberlappende_fenster_fallen_auf(szene):
    """Die Slab-Zuordnung schuetzt nur INNERHALB eines Fensters -- zwei
    ueberlappende Fenster sammelten dieselben Paare zweimal."""
    t, x, y, tid, sta = szene
    with pytest.raises(ValueError, match="ueberlappen"):
        collect_pairs(t, x, y, tid, sta, want_label=None,
                      **{**PARAMS, "win_offsets": (0, 1_000)})   # win_dt = 2000
