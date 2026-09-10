#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trajektorien.py -- BAUSTEIN [7]: aus der Partition die BAHNEN.

  lade_on_strom  (t, x, y) des ON-Stroms eines Runs, ohne Ground Truth
  gruppiere      Events nach Trajektorie, CSR-artig
  bahn           (t, x, y) EINER Trajektorie

Was [6c] als Event -> Trajektorie ablegt, wird hier zur Eingabe des Fits:
je Trajektorie drei gleich lange Arrays in aufsteigender Zeit.

DER VERTRAG zu [6c]: trajektorie_id hat EINEN Eintrag je ON-Event des
Runs, in der Reihenfolge des polaritaetsgefilterten Stroms. gruppiere
verlangt die Stromlaenge und bricht bei Abweichung ab -- ein Versatz um
einen Eintrag ordnete jedes Event der falschen Bahn zu und faellt an
keiner Zahl auf.

OHNE GROUND TRUTH: gelesen wird die .dat, nicht die Label-Datei. Echte
Messdaten haben keine Boxen, und der Fit braucht sie nicht.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eventpiv.kern.data import load_events
from eventpiv.kern.graph import polarity_mask


@dataclass
class Trajektorien:
    """Die Trajektorien eines Runs, plus der Weg zu ihren Events.

    cluster_id   (m,) Trajektorien-ID je Zeile, aufsteigend
    reihenfolge  Indizes in den ON-Strom, nach Trajektorie gruppiert
    grenzen      m+1 Offsets in reihenfolge

    Dieselbe Ueberlegung wie bei Knoten: die drei gehoeren zusammen und
    sind einzeln wertlos. Keine Tabelle wie dort -- es gibt genau eine
    Spalte, und ein dict verspraeche mehr.

    cluster_id ist die Bruecke zurueck zur Partition. Nach kompakt() sind
    die IDs lueckenlos, dann ist sie die Identitaet. Aber nur dann, und
    das ist keine Zusicherung des Solvers.
    """
    cluster_id: np.ndarray
    reihenfolge: np.ndarray
    grenzen: np.ndarray

    def __len__(self) -> int:
        return len(self.grenzen) - 1


def _leer() -> Trajektorien:
    return Trajektorien(np.empty(0, np.int64), np.empty(0, np.int64),
                        np.zeros(1, np.int64))


def lade_on_strom(split: str, run_nr: int, polarity: str = "on"):
    """(t, x, y) des Runs als float64, nach Polaritaet gefiltert.

    float64, weil der Fit damit rechnet: die Zeitstempel gehen bis 8e6 µs
    und liegen damit unter der float32-Grenze von 2^24, aber der Abstand
    ist kleiner als eine Groessenordnung. Preis: rund 90 MB je Array bei
    11 Mio. ON-Events.
    """
    ev = load_events(split, run_nr)
    m = polarity_mask(ev["p"], polarity)
    return (ev["t"][m].astype(np.float64),
            ev["x"][m].astype(np.float64),
            ev["y"][m].astype(np.float64))


def gruppiere(trajektorie_id, n_strom: int) -> Trajektorien:
    """Events nach Trajektorie gruppieren. Reine Funktion.

    trajektorie_id  ein Eintrag je ON-Event, -1 fuer nicht zugeordnete
    n_strom         Laenge des Eventstroms, gegen den geprueft wird

    Events mit -1 fallen heraus: sie gehoeren zu keiner Bahn.

    kind="stable" haelt die ZEITORDNUNG innerhalb einer Trajektorie: der
    ON-Strom ist zeitlich sortiert, und der Kalman-Filter bildet sein dt
    aus aufeinanderfolgenden Messungen. Ohne die Stabilitaet muesste der
    Aufrufer je Bahn nachsortieren.
    """
    tid = np.asarray(trajektorie_id)
    if len(tid) != int(n_strom):
        raise ValueError(
            f"VERTRAG VERLETZT: {int(n_strom):,} Events im Strom, "
            f"{len(tid):,} Trajektorien-IDs. Andere Polaritaet, anderer "
            "Run oder ein anderes Artefakt.")
    gilt = tid >= 0
    if not gilt.any():
        return _leer()

    idx_gilt = np.flatnonzero(gilt)
    werte, grp = np.unique(tid[gilt], return_inverse=True)
    m = len(werte)
    ordn = np.argsort(grp, kind="stable")
    return Trajektorien(
        werte.astype(np.int64),
        idx_gilt[ordn].astype(np.int64),
        np.concatenate([[0], np.cumsum(np.bincount(grp, minlength=m))]
                       ).astype(np.int64))


def bahn(tr: Trajektorien, i: int, t, x, y):
    """(t, x, y) der i-ten Trajektorie, zeitlich aufsteigend.

    Sicht auf den Strom, keine Kopie der Rohdaten -- die Indizierung legt
    drei kleine Arrays an, nicht drei grosse.
    """
    idx = tr.reihenfolge[tr.grenzen[i]:tr.grenzen[i + 1]]
    return np.asarray(t)[idx], np.asarray(x)[idx], np.asarray(y)[idx]