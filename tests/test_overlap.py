#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Invariantentests fuer overlap.py -- synthetische bbox, ohne Datensatz."""
import numpy as np

from eventpiv.bewerten.overlap import classify, overlap_metrics
from eventpiv.kern.runconfig import Overlap

# Default-Gruppe: box_w=11.0, cross_thresh=1.4, mean_clean_max=0.05.
# Die Tests pruefen die Kennzahlen, nicht die Schwellenwahl -- deshalb
# ueberall dieselbe Instanz.
OV = Overlap()


def test_weit_getrennte_boxen(make_bbox):
    """Zwei Tracks, 100 px auseinander -> nie unter 11 px."""
    bbox = make_bbox([(t, 100.0, 100.0, 1) for t in (0, 16667, 33334)] +
                     [(t, 200.0, 100.0, 2) for t in (0, 16667, 33334)])
    m = overlap_metrics(bbox, box_w=OV.box_w)
    assert m["tracks"] == 2 and m["n_raster"] == 3
    assert m["min_dist"] == 100.0
    assert m["frac11"] == 0.0 and m["mean11"] == 0.0


def test_durchgehende_ueberlappung(make_bbox):
    """Zwei Tracks im Abstand 5 px -> frac11 == 1.0."""
    bbox = make_bbox([(t, 100.0, 100.0, 1) for t in (0, 16667)] +
                     [(t, 105.0, 100.0, 2) for t in (0, 16667)])
    m = overlap_metrics(bbox, box_w=OV.box_w)
    assert m["frac11"] == 1.0
    assert m["min_dist"] == 5.0
    assert m["mean11"] == 1.0            # genau EIN Paar je Zeitpunkt


def test_mean11_zaehlt_paare_nicht_boxen(make_bbox):
    """Drei nahe Boxen ergeben 3 PAARE, nicht 3 Boxen -- query_pairs zaehlt
    Paare."""
    bbox = make_bbox([(0, 100.0, 100.0, 1), (0, 104.0, 100.0, 2),
                      (0, 108.0, 100.0, 3)])
    assert overlap_metrics(bbox, box_w=OV.box_w)["mean11"] == 3.0


def test_einzelner_track_liefert_keine_abstaende(make_bbox):
    bbox = make_bbox([(t, 100.0, 100.0, 1) for t in (0, 16667)])
    m = overlap_metrics(bbox, box_w=OV.box_w)
    assert np.isinf(m["min_dist"]) and m["frac11"] == 0.0


def test_klassifikation():
    ergebnisse = {
        1: {"min_dist": 100.0, "frac11": 0.0, "mean11": 0.0},   # sauber
        2: {"min_dist": 3.0, "frac11": 1.0, "mean11": 0.9},     # Overlap
        3: {"min_dist": 2.0, "frac11": 1.0, "mean11": 2.1},     # Kreuzung
    }
    cats = classify(ergebnisse, OV)
    assert cats["MIN_OVERLAP"] == [1]
    assert cats["MAX_OVERLAP"] == [2, 3]
    assert cats["CROSSING"] == [3]
