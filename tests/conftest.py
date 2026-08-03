#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gemeinsame Testbausteine.

BBOX_DTYPE und make_bbox standen wortgleich in test_labels.py und
test_overlap.py. Zwei Wahrheiten ueber dasselbe Dateiformat -- genau die
Doppelung, gegen die im Produktivcode ueberall argumentiert wird.
"""
import numpy as np
import pytest

BBOX_DTYPE = np.dtype([("t", "<i8"), ("x", "<f4"), ("y", "<f4"),
                       ("w", "<f4"), ("h", "<f4"), ("class_id", "<u4"),
                       ("track_id", "<u4"), ("class_confidence", "<f4")])


@pytest.fixture
def make_bbox():
    """Baut ein bbox-Array aus (t, x_ecke, y_ecke, track_id)-Tupeln."""
    def bauen(entries):
        b = np.zeros(len(entries), dtype=BBOX_DTYPE)
        for i, (t, x, y, tid) in enumerate(entries):
            b[i] = (t, x, y, 11.0, 11.0, 0, tid, 1.0)
        return b
    return bauen