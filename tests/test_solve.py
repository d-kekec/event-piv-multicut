#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Der CC-Vergleichssolver. Fuer solve.py gab es bisher keine Testdatei.

Geprueft wird die Baseline, nicht RAMA: RAMA braucht eine GPU, und was es
auf der bekannten Instanz liefert, steht in RAMA_Anbindung.md §5.1.
"""
import numpy as np
import pytest

from eventpiv.runconfig import Solver
from eventpiv.solve import SOLVER_FUNKTIONEN, solve_cc


def test_formatzusicherungen():
    lab = solve_cc(np.array([0, 1]), np.array([1, 2]),
                   np.array([1.0, 1.0]), num_nodes=3)
    assert lab.dtype == np.int32 and len(lab) == 3


def test_negative_kanten_trennen():
    """Zwei Dreiergruppen, verbunden durch eine einzige negative Kante."""
    i = np.array([0, 1, 2, 3, 4, 3])
    j = np.array([1, 2, 3, 4, 5, 5])
    c = np.array([5.0, 5.0, -8.0, 4.0, 3.0, 2.0])
    lab = solve_cc(i, j, c, num_nodes=6)
    assert lab[0] == lab[1] == lab[2]
    assert lab[3] == lab[4] == lab[5]
    assert lab[0] != lab[3]


def test_stimmt_mit_rama_auf_der_bekannten_instanz_ueberein():
    """Die 6-Knoten-Instanz aus RAMA_Anbindung.md §5.1.

    RAMA liefert dort [0,0,0,1,1,1], beweisbar optimal. CC liefert dieselbe
    Partition -- die negative Kante im frustrierten Dreieck faellt weg, das
    Dreieck bleibt trotzdem zusammenhaengend. Der Test haelt fest, dass die
    Baseline nicht generell schlechter ist; sie ist es an einer bestimmten
    Stelle, und die zeigt der naechste Test.
    """
    i = np.array([0, 1, 0, 2, 3, 4, 3])
    j = np.array([1, 2, 2, 3, 4, 5, 5])
    c = np.array([5.0, 5.0, -1.0, -8.0, 4.0, 3.0, 2.0])
    lab = solve_cc(i, j, c, num_nodes=6)
    assert len(set(lab.tolist())) == 2
    assert lab[0] == lab[1] == lab[2] and lab[3] == lab[4] == lab[5]


def test_eine_positive_kante_verschmilzt_zwei_wolken():
    """Die Kernaussage der Baseline, und der Grund fuer ihre Existenz.

    Zwei Gruppen, zwischen denen zehn Kanten liegen: neun stark negativ,
    eine schwach positiv. Die Bilanz spricht klar fuer den Schnitt, und
    Multicut duerfte die positive Kante mitschneiden. CC kann das nicht --
    es kennt nur die Kantenpraeferenz, nie die Bilanz.
    """
    i = np.array([0, 1] + [0] * 10)
    j = np.array([1, 2] + [3] * 10)          # alles gegen Knoten 3
    c = np.array([5.0, 5.0] + [-9.0] * 9 + [0.5])
    lab = solve_cc(i, j, c, num_nodes=4)
    assert lab[0] == lab[3], "CC muesste hier verschmelzen"


def test_kantenlose_knoten_bekommen_minus_eins():
    """Gleiche Buchhaltung wie RAMA (dort ueber sanitize_graph), sonst sind
    n_clusters_rama und n_nodes_unassigned nicht vergleichbar."""
    lab = solve_cc(np.array([0]), np.array([1]), np.array([1.0]), num_nodes=4)
    assert lab[0] == lab[1] and lab[0] >= 0
    assert lab[2] == -1 and lab[3] == -1


def test_knoten_mit_nur_negativen_kanten_bleibt_zugeordnet():
    """Er ist allein, aber nicht unzugeordnet -- der Unterschied zaehlt."""
    lab = solve_cc(np.array([0]), np.array([1]), np.array([-1.0]), num_nodes=2)
    assert lab[0] >= 0 and lab[1] >= 0 and lab[0] != lab[1]


def test_kosten_null_bleiben_erhalten():
    """Dieselbe Lesart wie solve/frac_neg: 0 zaehlt als nicht-negativ."""
    lab = solve_cc(np.array([0]), np.array([1]), np.array([0.0]), num_nodes=2)
    assert lab[0] == lab[1]


def test_leere_instanz_faellt_auf():
    with pytest.raises(ValueError):
        solve_cc(np.empty(0, np.int64), np.empty(0, np.int64),
                 np.empty(0, np.float32), num_nodes=5)


def test_backend_waehlt_die_funktion():
    assert set(SOLVER_FUNKTIONEN) == {"rama", "cc"}
    assert SOLVER_FUNKTIONEN["cc"] is solve_cc


def test_unbekanntes_backend_faellt_zur_konfigurationszeit_auf():
    with pytest.raises(ValueError):
        Solver(backend="unionfind")


def test_vorgabe_aendert_den_dateinamen_nicht():
    """Der Anker muss unter demselben Pfad erreichbar bleiben."""
    assert Solver().short() == "P"
    assert Solver(backend="cc").short() == "P_cc"