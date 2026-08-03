#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data.py -- Rohdaten laden (Events, Bounding Boxes).

Duenne Schicht ueber PSEELoader/NumPy. Einzige Aufgabe: aus (split, run_nr)
Arrays machen. Enthaelt KEINE Fachlogik -- die liegt in labels.py, graph.py etc.
"""
from __future__ import annotations

import numpy as np

from eventpiv import config
from eventpiv.psee_io.psee_loader import PSEELoader


def load_bbox(split: str, run_nr: int) -> np.ndarray:
    """Bounding-Box-Array eines Runs (strukturiert: t,x,y,w,h,class_id,track_id,...).

    (x,y) ist die ECKE oben links, w=h=11 (Konstante des Datensatzes).
    """
    _, bbox_path = config.run_files(split, run_nr)
    return np.load(bbox_path)


def load_events(split: str, run_nr: int, t0=None, dt=None) -> np.ndarray:
    """Events eines Runs als strukturiertes Array mit Feldern t,x,y,p.

    t0 : Startzeit in µs (None = ab Anfang)
    dt : Fensterlaenge in µs (None = bis Ende des Runs)

    Ganzer Run = alles im RAM. Fuer Experimente t0/dt setzen.
    """
    dat_path, _ = config.run_files(split, run_nr)
    loader = PSEELoader(str(dat_path))
    if t0 is not None:
        loader.seek_time(t0)
    if dt is None:
        # event_count() ist der DATEIGESAMTWERT, nicht der Rest ab t0.
        # load_n_events klammert intern auf das tatsaechlich Verbleibende --
        # das Ergebnis stimmt, aber der Puffer wird auf Dateigroesse
        # angelegt. Fuer "ab t0 bis Ende" mit grossem t0 daher teuer.
        return loader.load_n_events(loader.event_count())
    return loader.load_delta_t(dt)
