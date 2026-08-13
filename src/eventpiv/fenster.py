#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
fenster.py -- ein Zeitfenster laden und die Ground Truth darauf ausrichten.

  Fenster       Events und, falls vorhanden, die dazu AUSGERICHTETE GT
  lade_fenster  aus (Auswahl, Ausschnitt) ein Fenster bauen

Vom Rechnen getrennt, weil das hier I/O ueber NFS ist: solve_window und
solve_run bekommen ein fertiges Fenster und sind dadurch frei von Pfaden,
also auf von Hand gebauten Arrays pruefbar -- ohne Datensatz.

Die Ausrichtung ist die fehleranfaelligste Stelle der Stufe. Reihenfolge
deshalb fest: erst auf dem UNGEFILTERTEN Strom ausrichten, dann dieselbe
Polaritaetsmaske auf beide Seiten.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pathlib import Path

from eventpiv.evaluate import align_gt_window
from eventpiv.kern.data import load_events
from eventpiv.kern.graph import polarity_mask


@dataclass
class Fenster:
    """Ein geladenes Zeitfenster: Events und, falls vorhanden, die dazu
    AUSGERICHTETE Ground Truth.

    Warum ein eigener Typ und kein dict: die Ausrichtung der GT auf den
    Eventstrom ist die fehleranfaelligste Stelle dieser Stufe --
    align_gt_window existiert nur deswegen, und ein gefilterter Strom
    laeuft dort sofort auseinander. Ein Typ macht "ausgerichtet oder gar
    nicht" zu einer Eigenschaft des Objekts statt zu einer Zusage im Kopf.

    track_id und status sind None, wenn es keine Label-Datei gibt -- der
    Normalfall auf echten Messdaten. Sie sind NIE einzeln None.

    t, x, y in float64, weil der KDTree das erwartet. p bleibt in seinem
    Speicherformat: es geht in keine Rechnung ein.
    """
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    track_id: np.ndarray | None = None
    status: np.ndarray | None = None

    @property
    def hat_gt(self) -> bool:
        return self.track_id is not None

    def __len__(self) -> int:
        return len(self.t)


def lade_fenster(sel, win, *, polarity, gt_path=None) -> Fenster:
    """Laedt ein Zeitfenster eines Runs und richtet die GT darauf aus.

    Vom Loesen getrennt: Laden ist I/O ueber NFS, Loesen ist Rechnen.
    solve_window ist dadurch frei von Pfaden und auf vorgerechneten Arrays
    pruefbar.

    Die vollen GT-Arrays (rund 190 MB je Run) leben nur innerhalb des
    with-Blocks; was das Fenster behaelt, ist fenstergross.

    polarity ist Pflicht und ohne Vorgabewert: ein vergessener Filter waere
    im Einsatz eine andere Knotenmenge als im Training, und das faellt an
    keiner Kennzahl auf.

    gt_path = None oder eine fehlende Datei heisst: keine GT. Kein Fehler.
    """
    ev = load_events(sel.split, sel.run_nr, t0=win.t0, dt=win.dt)
    t = ev["t"].astype(np.float64)
    x = ev["x"].astype(np.float64)
    y = ev["y"].astype(np.float64)

    # Erst auf dem UNGEFILTERTEN Strom ausrichten, dann dieselbe Maske auf
    # beide Seiten -- umgekehrt bekaeme align_gt_window eine Teilfolge.
    tid = sta = None
    if gt_path is not None and Path(gt_path).exists():
        with np.load(gt_path) as g:
            sl = align_gt_window(g["t"], t)
            tid = g["track_id"][sl]
            sta = g["status"][sl]

    m = polarity_mask(ev["p"], polarity)
    return Fenster(t=t[m], x=x[m], y=y[m], p=ev["p"][m],
                   track_id=None if tid is None else tid[m],
                   status=None if sta is None else sta[m])
