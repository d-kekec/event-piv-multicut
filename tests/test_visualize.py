#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Invariantentests fuer die Farbvergabe in visualize.py.

Die Farbwahl entscheidet, WAS man auf der Animation sieht -- eine falsche
Zuordnung faellt visuell nicht auf, sondern fuehrt zu einer falschen
Interpretation. Getestet werden nur die reinen Funktionen; das Zeichnen
braucht Pillow-Ausgabe und bleibt aussen vor.
"""
import numpy as np
import pytest

from eventpiv.kern.status import STATUS_AMBIG, STATUS_CLEAN
from eventpiv.zeigen.visualize import (GREY, color_lookup, distinct_colors,
                                majority_gt_colors)


def ist_grau(farbe):
    return np.array_equal(np.asarray(farbe), GREY)


def test_distinct_colors_liefert_verschiedene_farben():
    """Der Goldene-Winkel-Trick: aufeinanderfolgende Indizes landen weit
    auseinander im Farbkreis, benachbarte Cluster also nie gleich."""
    farben = distinct_colors(20)
    assert farben.shape == (20, 3)
    assert len({tuple(f) for f in farben}) == 20


def test_distinct_colors_bei_null():
    """max(n, 1) im Code -- ein leeres Array waere ein Formfehler."""
    assert distinct_colors(0).shape == (1, 3)


def test_minus_eins_wird_grau():
    """RAMA vergibt -1 fuer kantenlose Knoten. Das ist KEIN Cluster."""
    uniq, colors = color_lookup(np.array([-1, -1, 0, 0, 0]))
    assert ist_grau(colors[np.searchsorted(uniq, -1)])
    assert not ist_grau(colors[np.searchsorted(uniq, 0)])


def test_kleine_cluster_werden_grau():
    """min_size trennt Rauschen von Partikeln -- darunter grau."""
    ids = np.array([0] * 10 + [1, 1] + [2])
    uniq, colors = color_lookup(ids, min_size=3)
    assert not ist_grau(colors[np.searchsorted(uniq, 0)])
    assert ist_grau(colors[np.searchsorted(uniq, 1)])
    assert ist_grau(colors[np.searchsorted(uniq, 2)])


def test_groesster_cluster_bekommt_die_erste_farbe():
    """Nach Anzahl sortiert: die groessten Cluster bekommen die
    kraeftigsten Farben, unabhaengig von ihrer ID."""
    gross_id_1 = color_lookup(np.array([0] * 2 + [1] * 9))
    uniq, colors = gross_id_1
    erste = distinct_colors(2)[0]
    np.testing.assert_array_equal(colors[np.searchsorted(uniq, 1)], erste)


def test_majority_faerbt_nach_mehrheits_gt():
    """Ein Cluster bekommt die Farbe SEINES Partikels -- so wird das
    Cluster-Panel mit dem GT-Panel vergleichbar."""
    labels = np.array([0, 0, 0, 1, 1, 1])
    gt = np.array([7, 7, 9, 9, 9, 9])          # Cluster 0 mehrheitlich 7
    st = np.full(6, STATUS_CLEAN, np.uint8)

    lab_uniq, lab_colors = majority_gt_colors(labels, gt, st)
    gt_uniq, gt_colors = color_lookup(gt)

    np.testing.assert_array_equal(
        lab_colors[np.searchsorted(lab_uniq, 0)],
        gt_colors[np.searchsorted(gt_uniq, 7)])
    np.testing.assert_array_equal(
        lab_colors[np.searchsorted(lab_uniq, 1)],
        gt_colors[np.searchsorted(gt_uniq, 9)])


def test_majority_faerbt_verschmolzene_cluster_einheitlich():
    """DER Fall, fuer den color_by='gt' gebaut ist: zwei GT-Partikel in
    EINEM Cluster bekommen eine Farbe -- der Reinheitsfehler wird sichtbar,
    weil im GT-Panel daneben zwei Farben stehen."""
    labels = np.zeros(6, np.int64)
    gt = np.array([7, 7, 7, 9, 9, 9])
    st = np.full(6, STATUS_CLEAN, np.uint8)
    lab_uniq, lab_colors = majority_gt_colors(labels, gt, st)
    assert len(lab_uniq) == 1 and not ist_grau(lab_colors[0])


def test_majority_ignoriert_minus_eins():
    """Ohne diese Ausnahme gewaenne ein einzelnes sauberes Event die
    Mehrheitsabstimmung im -1-Eimer und faerbte den gesamten Rauschanteil."""
    labels = np.array([-1, -1, -1, 0, 0])
    gt = np.array([7, 7, 7, 9, 9])
    st = np.full(5, STATUS_CLEAN, np.uint8)
    lab_uniq, lab_colors = majority_gt_colors(labels, gt, st)
    assert ist_grau(lab_colors[np.searchsorted(lab_uniq, -1)])


def test_majority_zaehlt_nur_saubere_events():
    """Mehrdeutige Events duerfen die Mehrheit nicht kippen -- sonst
    faerbte die Animation nach etwas, das die Metrik gar nicht bewertet."""
    labels = np.zeros(5, np.int64)
    gt = np.array([7, 7, 9, 9, 9])
    st = np.array([STATUS_CLEAN, STATUS_CLEAN,
                   STATUS_AMBIG, STATUS_AMBIG, STATUS_AMBIG], np.uint8)
    lab_uniq, lab_colors = majority_gt_colors(labels, gt, st)
    gt_uniq, gt_colors = color_lookup(gt[st == STATUS_CLEAN])
    # Nur die sauberen Events zaehlen -> Mehrheit ist 7, nicht 9.
    np.testing.assert_array_equal(
        lab_colors[np.searchsorted(lab_uniq, 0)],
        gt_colors[np.searchsorted(gt_uniq, 7)])


def test_determinismus():
    ids = np.array([0, 0, 1, 2, 2, 2, -1])
    a = color_lookup(ids)
    b = color_lookup(ids)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])