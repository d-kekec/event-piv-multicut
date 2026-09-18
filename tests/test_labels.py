#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Invariantentests fuer labels.assign_track_ids.

Laufen OHNE Datensatz (der liegt nicht im Repo) auf einem synthetischen
Mini-Fixture mit von Hand bekannter Zuordnung. Geprueft werden Eigenschaften,
die unabhaengig von der Parameterwahl gelten muessen -- ein Vergleich gegen
eine Referenzdatei waere bei jeder gewollten Aenderung wertlos.
"""
import numpy as np
import pytest

from eventpiv.kern.status import STATUS_AMBIG, STATUS_CLEAN, STATUS_NOISE
from eventpiv.kette.labels import assign_track_ids



@pytest.fixture
def scene(make_bbox):
    """Zwei Partikel, weit getrennt (Zentren 100.5 und 300.5), ein Rasterpunkt.

    Events von Hand gesetzt:
      0: exakt auf Zentrum A            -> sauber, track 7
      1: 5 px neben A (typische Wolke)  -> sauber, track 7
      2: auf Zentrum B                  -> sauber, track 9
      3: 50 px im Nichts                -> Rausch
      4: genau mittig zwischen A und B  -> Rausch (viel zu weit von beiden)
    """
    bbox = make_bbox([(1000, 95.0, 95.0, 7), (1000, 295.0, 295.0, 9)])
    t = np.array([1000, 1000, 1000, 1000, 1000], dtype=np.float64)
    x = np.array([100.5, 105.5, 300.5, 700.0, 200.5])
    y = np.array([100.5, 100.5, 300.5, 700.0, 200.5])
    return t, x, y, bbox


def test_formatzusicherungen(scene):
    t, x, y, bbox = scene
    tid, st = assign_track_ids(t, x, y, bbox)
    assert len(tid) == len(st) == len(t)
    assert tid.dtype == np.int32 and st.dtype == np.uint8
    assert set(np.unique(st)) <= {STATUS_CLEAN, STATUS_AMBIG, STATUS_NOISE}


def test_track_id_genau_dann_wenn_rauschen(scene):
    """Kernzusicherung des Formats: -1 <=> RAUSCH.

    Frueher galt -1 <=> nicht sauber. Mehrdeutige Events tragen jetzt die
    track_id ihres naechsten Zentrums: status entscheidet, was ins TRAINING
    geht, track_id sagt nur, WOHIN ein Event gehoert. Ohne diese Trennung
    ist die Merge-Rate blind, weil verschmolzene Bereiche ueberwiegend aus
    mehrdeutigen Events bestehen.
    """
    t, x, y, bbox = scene
    tid, st = assign_track_ids(t, x, y, bbox)
    np.testing.assert_array_equal(tid == -1, st == STATUS_NOISE)


def test_saubere_events_liegen_innerhalb_r_assign(scene):
    """Prueft die Kernlogik direkt, ohne Referenzdatei."""
    t, x, y, bbox = scene
    r_assign = 8.0
    tid, st = assign_track_ids(t, x, y, bbox, r_assign=r_assign)
    centers = {int(b["track_id"]): (b["x"] + 5.5, b["y"] + 5.5) for b in bbox}
    for i in np.flatnonzero(st == STATUS_CLEAN):
        cx, cy = centers[int(tid[i])]
        assert np.hypot(x[i] - cx, y[i] - cy) <= r_assign

def test_zeitliche_zuordnung_zum_naechsten_rasterpunkt(make_bbox):
    """Zwei Rasterpunkte, dasselbe Partikel bewegt sich dazwischen.

    Das Event liegt zeitlich naeher an t=2000 -> muss gegen die DORTIGE
    Boxposition gepruefft werden, nicht gegen die bei t=1000.
    """
    bbox = make_bbox([(1000, 95.0, 95.0, 4),      # Zentrum (100.5, 100.5)
                      (2000, 195.0, 95.0, 4)])    # Zentrum (200.5, 100.5)
    # t=1900 -> naeher an 2000; raeumlich auf dem SPAETEN Zentrum
    t = np.array([1900.0]); x = np.array([200.5]); y = np.array([100.5])
    tid, st = assign_track_ids(t, x, y, bbox)
    assert st[0] == STATUS_CLEAN and tid[0] == 4

    # dasselbe Event raeumlich auf dem FRUEHEN Zentrum -> zu weit weg
    x2 = np.array([100.5])
    _, st2 = assign_track_ids(t, x2, y, bbox)
    assert st2[0] == STATUS_NOISE

def test_bekannte_zuordnung(scene):
    t, x, y, bbox = scene
    tid, st = assign_track_ids(t, x, y, bbox)
    assert list(st) == [STATUS_CLEAN, STATUS_CLEAN, STATUS_CLEAN,
                        STATUS_NOISE, STATUS_NOISE]
    assert tid[0] == tid[1] == 7 and tid[2] == 9


def test_boxmitte_kommt_aus_der_w_spalte():
    """Polydisperse Boxen: die Mitte folgt der ZEILE, nicht einer Konstanten.

    Box mit w = h = 4 an der Ecke (98, 98) -- Zentrum also (100, 100). Ein
    Event genau dort ist sauber. Mit einer Konstanten von 11 laege das
    angenommene Zentrum bei (103.5, 103.5), also 4,95 px daneben, und bei
    r_assign = 3 fiele das Event auf RAUSCH.

    make_bbox wird hier NICHT benutzt: die Fixture setzt w = h = 11 fest,
    und genau davon soll dieser Test unabhaengig sein.
    """
    from mini_dataset import BBOX_DTYPE

    b = np.zeros(1, dtype=BBOX_DTYPE)
    b[0] = (1000, 98.0, 98.0, 4.0, 4.0, 0, 5, 1.0)
    t = np.array([1000.0]); x = np.array([100.0]); y = np.array([100.0])
    tid, st = assign_track_ids(t, x, y, b, r_assign=3.0)
    assert st[0] == STATUS_CLEAN and tid[0] == 5


def test_monotonie_r_assign(scene):
    """Sehr grosses r_assign => es darf keinen Rausch-Status mehr geben."""
    t, x, y, bbox = scene
    _, st = assign_track_ids(t, x, y, bbox, r_assign=1e9)
    assert not (st == STATUS_NOISE).any()


def test_monotonie_r_ambig(scene):
    """r_ambig = 0 => nie mehrdeutig (d2-d1 >= 0 immer)."""
    t, x, y, bbox = scene
    _, st = assign_track_ids(t, x, y, bbox, r_ambig=0.0)
    assert not (st == STATUS_AMBIG).any()


def test_mehrdeutig_wird_erkannt(make_bbox):
    """Zwei Boxen dicht beieinander, Event exakt mittig -> MEHRDEUTIG.

    Der Abstand zu beiden Zentren ist hier exakt gleich (3.0 px). WELCHE
    der beiden IDs cKDTree bei Gleichstand liefert, ist nicht zugesichert
    -- geprueft wird deshalb nur, dass ueberhaupt zugeordnet wird.
    """
    bbox = make_bbox([(1000, 95.0, 95.0, 1), (1000, 101.0, 95.0, 2)])
    t = np.array([1000.0]); x = np.array([103.5]); y = np.array([100.5])
    tid, st = assign_track_ids(t, x, y, bbox, r_assign=8.0, r_ambig=2.0)
    assert st[0] == STATUS_AMBIG and tid[0] in (1, 2)


def test_mehrdeutig_bekommt_naechstes_zentrum(make_bbox):
    """Mehrdeutig, aber mit eindeutigem NAECHSTEN -- dessen ID muss stehen.

    Zentren bei 100.5 und 106.5, Event bei 103.0: d1 = 2.5 zu Partikel 1,
    d2 = 3.5 zu Partikel 2. Differenz 1.0 < r_ambig = 2.0 -> MEHRDEUTIG,
    naechstes Zentrum aber eindeutig Partikel 1.
    """
    bbox = make_bbox([(1000, 95.0, 95.0, 1), (1000, 101.0, 95.0, 2)])
    t = np.array([1000.0]); x = np.array([103.0]); y = np.array([100.5])
    tid, st = assign_track_ids(t, x, y, bbox, r_assign=8.0, r_ambig=2.0)
    assert st[0] == STATUS_AMBIG and tid[0] == 1


def test_determinismus(scene):
    t, x, y, bbox = scene
    a = assign_track_ids(t, x, y, bbox)
    b = assign_track_ids(t, x, y, bbox)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_leere_eingabe(make_bbox):
    bbox = make_bbox([(1000, 95.0, 95.0, 1)])
    tid, st = assign_track_ids(np.array([]), np.array([]), np.array([]), bbox)
    assert len(tid) == 0 and len(st) == 0
