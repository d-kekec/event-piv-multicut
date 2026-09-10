#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Der Weg von Knoten-Clustern zurueck auf die Events.

Die stillste Stelle der Stufe: searchsorted liefert auch fuer fremde Werte
eine Position, und ohne Trefferpruefung landeten gefilterte Events im
falschen Cluster -- die Zahlen saehen plausibel aus.
"""
import numpy as np

from eventpiv.kette.solve2 import event_labels


def test_zuordnung_und_gefilterte_events():
    # Knoten 4, 9, 12; Tracklet 7 wurde gefiltert, -1 hatte nie eins.
    knoten_tid = np.array([4, 9, 12], np.int64)
    node_labels = np.array([0, 0, 1], np.int32)
    tid_ev = np.array([4, 4, 7, 9, -1, 12, 12, 99], np.int64)

    lab = event_labels(tid_ev, knoten_tid, node_labels)
    np.testing.assert_array_equal(lab, [0, 0, -1, 0, -1, 1, 1, -1])


def test_kantenloser_knoten_bleibt_minus_eins():
    """RAMA schreibt kantenlose Knoten nicht und laesst -1 stehen. Das darf
    nicht als Cluster 0 bei den Events ankommen."""
    lab = event_labels(np.array([4, 9]), np.array([4, 9], np.int64),
                       np.array([-1, 3], np.int32))
    np.testing.assert_array_equal(lab, [-1, 3])


def test_leere_knotenmenge():
    lab = event_labels(np.array([1, 2, 3]), np.empty(0, np.int64),
                       np.empty(0, np.int32))
    np.testing.assert_array_equal(lab, [-1, -1, -1])


def test_basislinie_gibt_jedem_tracklet_einen_eigenen_cluster():
    """Die Stufe-1-Partition faellt aus derselben Funktion, mit arange als
    Knotenlabels -- eingeschraenkt auf dieselbe Knotenmenge."""
    knoten_tid = np.array([4, 9, 12], np.int64)
    tid_ev = np.array([4, 9, 9, 12, 7], np.int64)
    lab = event_labels(tid_ev, knoten_tid, np.arange(3, dtype=np.int32))
    np.testing.assert_array_equal(lab, [0, 1, 1, 2, -1])


def test_kompakt_nummeriert_lueckenlos_und_haelt_minus_eins():
    """RAMA nummeriert nicht lueckenlos -- der Wertebereich folgt der
    Knotenzahl. Fuer das Artefakt zaehlt nur die Zugehoerigkeit."""
    from eventpiv.kette.solve2 import kompakt

    lab = kompakt(np.array([7, 7, 91_000, -1, 3], np.int32))
    np.testing.assert_array_equal(lab, [1, 1, 2, -1, 0])
    assert lab.dtype == np.int32


def test_kompakt_ohne_cluster():
    from eventpiv.kette.solve2 import kompakt

    lab = kompakt(np.full(4, -1, np.int32))
    np.testing.assert_array_equal(lab, [-1, -1, -1, -1])


def test_kompakt_aendert_die_partition_nicht():
    """Umnummerieren darf Knoten nicht anders gruppieren."""
    from eventpiv.kette.solve2 import kompakt

    roh = np.array([5, 900, 5, -1, 900, 12], np.int32)
    neu = kompakt(roh)
    for a in range(len(roh)):
        for b in range(len(roh)):
            if roh[a] >= 0 and roh[b] >= 0:
                assert (roh[a] == roh[b]) == (neu[a] == neu[b])