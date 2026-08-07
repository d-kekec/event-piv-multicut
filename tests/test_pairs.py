#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Invariantentests fuer pairs.collect_pairs -- ohne Datensatz, ohne Referenz."""
import numpy as np
import pytest

from eventpiv.labels import STATUS_AMBIG, STATUS_CLEAN, STATUS_NOISE
from eventpiv.pairs import collect_pairs

# ja_frac=1.0 in den meisten Tests: sie pruefen die Fensterlogik, nicht das
# Ziehen, und sollen deterministisch sein. rng wird trotzdem gebraucht.
PARAMS = dict(alpha=1_000.0, r_max=5.0, geom="ellipsoid", ja_frac=1.0,
              rng=np.random.default_rng(0))


@pytest.fixture
def szene():
    """Zwei dicht benachbarte Partikel (3 px), je 6 Events ueber 1,5 ms.

    Abstand 3 px < r_max -> es entstehen sowohl JA- als auch NEIN-Paare.
    Ein Event je Partikel ist absichtlich NICHT sauber.

    ZEITLICH SORTIERT, wie ein echter Eventstrom: load_events liefert die
    Prophesee-Datei in Aufnahmereihenfolge, und collect_pairs verankert die
    Fenstergrenzen an t[0].

    Bei alpha = 1000 ergibt das ZWEI Fenster: t < 1000 und t >= 1000.
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
    e, _ = collect_pairs(t, x, y, tid, sta,
                         **{**PARAMS, "geom": "ellipsoid"})
    z, _ = collect_pairs(t, x, y, tid, sta,
                         **{**PARAMS, "geom": "cylinder"})
    assert len(z["label"]) >= len(e["label"])

def test_label_entspricht_track_id(szene):
    t, x, y, tid, sta = szene
    d, _ = collect_pairs(t, x, y, tid, sta, **PARAMS)
    assert len(d["label"]) > 0
    assert set(np.unique(d["label"])) <= {0, 1}


def test_dt_nicht_negativ(szene):
    """i = frueheres Event -- Grundannahme des gesamten Feature-Designs."""
    t, x, y, tid, sta = szene
    d, _ = collect_pairs(t, x, y, tid, sta, **PARAMS)
    assert (d["dt"] >= 0).all()


def test_nur_saubere_events(szene):
    """Kein Paar darf ein mehrdeutiges oder verrauschtes Event enthalten."""
    t, x, y, tid, sta = szene
    d, _ = collect_pairs(t, x, y, tid, sta, **PARAMS)
    # Die unsauberen Events liegen bei t=600 (P1) und t=900 (P2). Partner mit
    # demselben t sind erlaubt, solange sie vom sauberen Partikel stammen --
    # deshalb ueber die Positionen pruefen statt ueber die Zeit:
    assert d["label"].size > 0
    for ti, xi, dt, dx in zip(d["ti"], d["xi"], d["dt"], d["dx"]):
        for tt, xx in ((ti, xi), (ti + dt, xi + dx)):
            treffer = np.flatnonzero((t == tt) & (np.isclose(x, xx)))
            assert (sta[treffer] == STATUS_CLEAN).any()


def test_kein_doppelzaehlen(szene):
    """Jedes Paar genau einmal. Frueher leistete das die Slab-Zuordnung
    ueber das fruehere Event; jetzt folgt es daraus, dass die Fenster
    disjunkt sind und der rechte Rand NICHT erweitert wird."""
    t, x, y, tid, sta = szene
    d, _ = collect_pairs(t, x, y, tid, sta, **PARAMS)
    schluessel = list(zip(d["ti"], d["xi"], d["dt"], d["dx"], d["dy"]))
    assert len(set(schluessel)) == len(schluessel)


def test_saubere_szene_liefert_keine_nein_paare():
    """Physikalisch: weit getrennte Partikel liegen ausserhalb von r_max und
    erzeugen zwischen sich gar keine Paare -- deshalb sind Overlap-Runs die
    EINZIGE NEIN-Quelle."""
    zeiten = np.arange(6, dtype=np.float64) * 300.0
    t = np.repeat(zeiten, 2)
    x = np.tile(np.array([100.0, 400.0]), 6)        # 300 px auseinander
    y = np.full(12, 100.0)
    tid = np.tile(np.array([1, 2], dtype=np.int16), 6)
    sta = np.full(12, STATUS_CLEAN, dtype=np.uint8)
    d, _ = collect_pairs(t, x, y, tid, sta, **PARAMS)
    # Laengenpruefung zuerst: .all() auf einem leeren Array ist True, der
    # Test waere sonst auch dann gruen, wenn gar nichts gesammelt wurde.
    assert len(d["label"]) > 0
    assert (d["label"] == 1).all()

def test_determinismus(szene):
    """Zwei Aufrufe, derselbe rng -- und trotzdem gleiches Ergebnis.

    Bei ja_frac=1.0 ist die Ziehung `rng.random(n) < 1.0` immer wahr,
    unabhaengig vom Zustand des Generators. Der Test prueft also die
    Sammellogik, nicht die Reproduzierbarkeit der Ziehung; die haengt am
    seed und wird in build_pair_dataset gesetzt.
    """
    t, x, y, tid, sta = szene
    a, za = collect_pairs(t, x, y, tid, sta, **PARAMS)
    b, zb = collect_pairs(t, x, y, tid, sta, **PARAMS)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    assert za == zb


def test_leere_eingabe():
    leer = np.empty(0)
    d, z = collect_pairs(leer, leer, leer, np.empty(0, np.int16),
                         np.empty(0, np.uint8), **PARAMS)
    assert len(d["label"]) == 0
    assert z["n_fenster"] == 0

def test_unsortierte_zeit_faellt_auf(szene):
    """Die Fenstergrenzen entstehen ueber searchsorted -- bei unsortiertem
    t schnitte das an beliebigen Stellen."""
    t, x, y, tid, sta = szene
    with pytest.raises(ValueError, match="sortiert"):
        collect_pairs(t[::-1], x, y, tid, sta, **PARAMS)


def test_keine_kante_ueber_die_fenstergrenze():
    """Der Kern der Umstellung: harter Schnitt bei alpha.

    Zwei Events, 2 µs auseinander und 1 px entfernt -- geometrisch klare
    Nachbarn. Sie liegen aber links und rechts der Fenstergrenze bei
    t[0]+alpha, und damit darf es das Paar nicht geben. Im Einsatz von
    Stufe 1 kann es nicht entstehen; entstuende es hier, lernte das Netz
    eine Kante, die es nie zu bewerten bekommt.

    Das Gegenstueck steht in der zweiten Zusicherung: INNERHALB des
    Fensters muss das Paar mit dt = 999 sehr wohl entstehen. Ohne sie waere
    der Test auch gruen, wenn gar nichts mehr gesammelt wuerde.
    """
    t = np.array([0.0, 999.0, 1001.0, 1500.0])
    x = np.array([100.0, 100.0, 101.0, 200.0])
    y = np.full(4, 100.0)
    tid = np.array([1, 1, 1, 1], dtype=np.int16)
    sta = np.zeros(4, np.uint8)
    d, _ = collect_pairs(t, x, y, tid, sta, **PARAMS)   # alpha = 1000
    paare = set(zip(d["ti"].tolist(), (d["ti"] + d["dt"]).tolist()))
    assert (999.0, 1001.0) not in paare
    assert (0.0, 999.0) in paare, "Paar INNERHALB des Fensters fehlt"


def test_ja_frac_null_laesst_nur_nein_uebrig(szene):
    """NEIN wird vollstaendig behalten, JA anteilig -- ja_frac=0 ist der
    Grenzfall, der beides zugleich zeigt."""
    t, x, y, tid, sta = szene
    d, z = collect_pairs(t, x, y, tid, sta, **{**PARAMS, "ja_frac": 0.0})
    assert len(d["label"]) > 0
    assert (d["label"] == 0).all()
    assert int((d["label"] == 0).sum()) == z["n_roh_nein"]

def test_zaehler_haelt_die_rohen_zahlen(szene):
    """Ohne die Rohzahlen waere nach dem Duennen nicht mehr rekonstruierbar,
    welches Klassenverhaeltnis das Netz im EINSATZ vorfindet."""
    t, x, y, tid, sta = szene
    voll, z_voll = collect_pairs(t, x, y, tid, sta, **PARAMS)
    _, z_duenn = collect_pairs(t, x, y, tid, sta, **{**PARAMS, "ja_frac": 0.0})
    assert z_voll["n_roh_ja"] == z_duenn["n_roh_ja"] > 0
    assert z_voll["n_roh_nein"] == z_duenn["n_roh_nein"] > 0
    assert int((voll["label"] == 1).sum()) == z_voll["n_roh_ja"]


def test_fenster_werden_gezaehlt(szene):
    """Die Szene laeuft ueber 1,5 ms bei alpha = 1 ms -> zwei Fenster."""
    _, z = collect_pairs(*szene, **PARAMS)
    assert z["n_fenster"] == 2