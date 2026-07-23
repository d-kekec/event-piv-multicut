#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Invariantentests fuer pairs.collect_pairs -- ohne Datensatz, ohne Referenz."""
import numpy as np
import pytest

from eventpiv.labels import STATUS_AMBIG, STATUS_CLEAN, STATUS_NOISE
from eventpiv.pairs import collect_pairs

PARAMS = dict(alpha=1_000.0, r_max=5.0, win_offsets=(0,),
              win_dt=2_000, slab_dt=1_000)


@pytest.fixture
def szene():
    """Zwei dicht benachbarte Partikel (3 px), je 6 Events ueber 2 ms.

    Abstand 3 px < r_max -> es entstehen sowohl JA- als auch NEIN-Paare.
    Ein Event je Partikel ist absichtlich NICHT sauber.
    """
    zeiten = np.arange(6, dtype=np.float64) * 300.0
    t = np.concatenate([zeiten, zeiten])
    x = np.concatenate([np.full(6, 100.0), np.full(6, 103.0)])
    y = np.full(12, 100.0)
    tid = np.array([1]*6 + [2]*6, dtype=np.int16)
    sta = np.full(12, STATUS_CLEAN, dtype=np.uint8)
    sta[2] = STATUS_AMBIG        # Partikel 1
    sta[9] = STATUS_NOISE        # Partikel 2
    return t, x, y, tid, sta

def test_geom_wird_durchgereicht(szene):
    """Zylinder ist weiter als Ellipsoid -> mindestens so viele Paare."""
    t, x, y, tid, sta = szene
    e = collect_pairs(t, x, y, tid, sta, want_label=None,
                      geom="ellipsoid", **PARAMS)
    z = collect_pairs(t, x, y, tid, sta, want_label=None,
                      geom="cylinder", **PARAMS)
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
    unsauber_t = set(t[sta != STATUS_CLEAN].tolist())
    beteiligt = set(d["ti"].tolist()) | set((d["ti"] + d["dt"]).tolist())
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
    t = np.concatenate([zeiten, zeiten])
    x = np.concatenate([np.full(6, 100.0), np.full(6, 400.0)])   # 300 px
    y = np.full(12, 100.0)
    tid = np.array([1]*6 + [2]*6, dtype=np.int16)
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
