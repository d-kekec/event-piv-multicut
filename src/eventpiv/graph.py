#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graph.py -- die (alpha, r_max)-Nachbarschaft. EINE Definition fuer alle Nutzer.

Zentraler Zweck der Zusammenlegung: build_pair_dataset (Training) und
solve_with_nn (Einsatz) MUESSEN dieselbe Kantenmenge erzeugen -- sonst sieht
das Netz im Einsatz eine andere Nachbarschaft als im Training. Bisher war das
eine Konvention (der Orchestrator reichte alpha/r_max an beide durch); jetzt
ist es strukturell erzwungen, weil beide diese Funktion aufrufen.

GEOMETRIE (wichtig, weicht von BA_NN_Kantenkosten.md §2 ab):
Der KDTree arbeitet auf (x, y, t*s) mit s = r_max/alpha und sucht mit Radius
r_max. Die tatsaechliche Bedingung ist damit

    dx^2 + dy^2 + s^2*dt^2 <= r_max^2          (KUGEL)

und nicht die dokumentierte Quader-Bedingung |dt| <= alpha UND dr <= r_max.
Die Kugel IMPLIZIERT beide Quaderbedingungen (aus der Summe folgt jeder
Summand einzeln), ist also strenger: Volumenverhaeltnis Kugel:Zylinder = 2:3.
Ein Paar mit dt = alpha und dr = 3px erfuellt den Quader, liegt aber ausserhalb
der Kugel.

Konsequenz: die frueher im Code stehende Nachpruefung auf die Quaderbedingung
konnte NIE etwas herausfiltern. Sie ist hier als verify=... erhalten, aber
per Default aus -- das Ergebnis ist bitgleich, nur ohne den toten Aufwand.
Stand 22.07.2026: Verhalten unveraendert uebernommen, Bewertung der
Kugel-vs-Quader-Frage offen.

KONTRAKT der Rueckgabe (auf den sich alle Aufrufer verlassen):
  - i, j sind Index-Arrays gleicher Laenge
  - t[i] <= t[j] fuer jedes Paar  (i = FRUEHERES Event)
    -> dt = t[j]-t[i] >= 0, wie im Training vorausgesetzt
  - keine Selbstkanten (i != j)
  - jede ungerichtete Kante genau einmal
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

K_MAX_DEFAULT = 24          # k-NN-Kappung, s. BA_NN_Kantenkosten.md §2.2


def time_scale(alpha: float, r_max: float) -> float:
    """Skalierung der Zeitachse, damit alpha in der Metrik r_max entspricht."""
    return r_max / alpha


def neighborhood_pairs(t, x, y, alpha, r_max, *,
                       use_knn: bool = False,
                       k_max: int = K_MAX_DEFAULT,
                       dedup: bool = False,
                       verify: bool = False):
    """Alle Paare in der (alpha, r_max)-Nachbarschaft. Reine Funktion.

    t, x, y : 1D-Arrays gleicher Laenge (float64 empfohlen -- KDTree)
    alpha   : µs, zeitliche Reichweite
    r_max   : px, raeumliche Reichweite
    use_knn : Kappung auf die k_max naechsten Nachbarn (MODELLIERUNGS-
              ENTSCHEIDUNG, nicht nur Optimierung -- BA_NN §2.2)
    k_max   : nur bei use_knn
    dedup   : doppelte ungerichtete Kanten entfernen. Bei use_knn NOETIG
              (die k-NN-Abfrage liefert i->j und j->i getrennt), bei
              query_pairs ein No-op (liefert bereits i<j eindeutig).
              ACHTUNG: sortiert die Kantenreihenfolge um.
    verify  : zusaetzliche Pruefung der Quaderbedingung. Kann per Konstruktion
              nichts herausfiltern (s. Modul-Docstring) -- nur als
              Selbstkontrolle in Tests gedacht.

    Rueckgabe: (i, j) int64-Arrays, i = frueheres Event.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(t)

    empty = (np.empty(0, np.int64), np.empty(0, np.int64))
    if n < 2:
        return empty

    s = time_scale(alpha, r_max)
    pts = np.column_stack([x, y, t * s])
    tree = cKDTree(pts)

    if use_knn:
        # k_max+1, weil der naechste Nachbar der Punkt selbst ist (Spalte 0).
        # Fehlende Nachbarn markiert cKDTree mit Index == n -> herausfiltern.
        _, idx = tree.query(pts, k=k_max + 1,
                            distance_upper_bound=r_max, workers=-1)
        i = np.repeat(np.arange(n), k_max)
        j = idx[:, 1:].ravel()
        ok = j < n
        i, j = i[ok], j[ok]
    else:
        pairs = tree.query_pairs(r=r_max, output_type="ndarray")
        if len(pairs) == 0:
            return empty
        i, j = pairs[:, 0], pairs[:, 1]

    if verify:
        ok = (np.abs(t[i] - t[j]) <= alpha) & \
             ((x[i] - x[j])**2 + (y[i] - y[j])**2 <= r_max**2)
        i, j = i[ok], j[ok]

    if len(i) == 0:
        return empty

    # --- Kontrakt herstellen: i = frueheres Event ---
    swap = t[i] > t[j]
    i, j = np.where(swap, j, i), np.where(swap, i, j)

    if dedup:
        lo, hi = np.minimum(i, j), np.maximum(i, j)
        key = lo.astype(np.int64) * n + hi
        _, uniq = np.unique(key, return_index=True)
        i, j = i[uniq], j[uniq]

    return i.astype(np.int64), j.astype(np.int64)


def edge_deltas(t, x, y, i, j):
    """Rohe Relativgroessen je Kante: (dt, dx, dy) mit dt >= 0.

    Einzige Stelle, an der aus Indexpaaren Features werden -- Training und
    Einsatz benutzen dieselbe. Die konkreten Feature-Modi (dr / absxy / raw6)
    bauen darauf auf (model.py bzw. train.py).
    """
    return (t[j] - t[i]), (x[j] - x[i]), (y[j] - y[i])
