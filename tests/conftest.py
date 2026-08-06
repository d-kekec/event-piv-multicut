#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gemeinsame Testbausteine.

BBOX_DTYPE und make_bbox standen wortgleich in test_labels.py und
test_overlap.py. Zwei Wahrheiten ueber dasselbe Dateiformat -- genau die
Doppelung, gegen die im Produktivcode ueberall argumentiert wird.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from mini_dataset import BBOX_DTYPE, baue_mini_datensatz


@pytest.fixture
def make_bbox():
    """Baut ein bbox-Array aus (t, x_ecke, y_ecke, track_id)-Tupeln."""
    def bauen(entries):
        b = np.zeros(len(entries), dtype=BBOX_DTYPE)
        for i, (t, x, y, tid) in enumerate(entries):
            b[i] = (t, x, y, 11.0, 11.0, 0, tid, 1.0)
        return b
    return bauen

@pytest.fixture
def mini_datensatz(tmp_path, monkeypatch):
    """Vollstaendiger Mini-Datensatz auf tmp_path, Wurzeln umgebogen.

    DATA_ROOT und OUT_ROOT sind Modul-Globals, die config.py beim IMPORT
    aufloest -- umbiegen geht deshalb nur ueber monkeypatch.setattr, nicht
    ueber setenv (test_config.py zeigt es fuer OUT_ROOT).

    config wird INNERHALB der Fixture importiert: ein Import auf Modulebene
    machte conftest.py von gesetzten Umgebungsvariablen abhaengig, und dann
    liesse sich kein einziger Test mehr sammeln, ohne dass sie stehen.

    Rueckgabe: Namensraum mit data, out, kreuzung, sauber.
    """
    from eventpiv import config
    data, out = tmp_path / "data", tmp_path / "out"
    runs = baue_mini_datensatz(data, "train")
    monkeypatch.setattr(config, "DATA_ROOT", data)
    monkeypatch.setattr(config, "OUT_ROOT", out)
    return SimpleNamespace(data=data, out=out, **runs)