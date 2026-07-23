#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graph.py -- die (alpha, r_max)-Nachbarschaft. EINE Definition fuer alle Nutzer.

Zentraler Zweck der Zusammenlegung: build_pair_dataset (Training) und
solve_with_nn (Einsatz) MUESSEN dieselbe Kantenmenge erzeugen -- sonst sieht
das Netz im Einsatz eine andere Nachbarschaft als im Training. Bisher war das
eine Konvention (der Orchestrator reichte alpha/r_max an beide durch); jetzt
ist es strukturell erzwungen, weil beide diese Funktion aufrufen.

GEOMETRIE (Befund 22.07.2026, BA_NN_Kantenkosten.md §2):
Der KDTree arbeitet auf (x, y, t*s) mit s = r_max/alpha. Mit Suchradius
r_max liefert er von sich aus ein ELLIPSOID, nicht den in der Doku
beschriebenen Zylinder -- und das Ellipsoid ist strenger (es impliziert
beide Zylinderbedingungen). Alle Ergebnisse bis 23.07.2026 entstanden damit.
Ueber den Parameter `geom` sind jetzt drei Formen waehlbar, s. dort.

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

NEIGHBORHOOD_MODES = ("ellipsoid", "cylinder", "hyperboloid")
V_MAX_DEFAULT = 1.02e-4     # px/µs (= 0.102 px/ms, p99.9 gepoolt)


def time_scale(alpha: float, r_max: float) -> float:
    """Skalierung der Zeitachse, damit alpha in der Metrik r_max entspricht."""
    return r_max / alpha

def neighborhood_pairs(t, x, y, alpha, r_max, *,
                       geom: str = "ellipsoid",
                       v_max: float = V_MAX_DEFAULT,
                       use_knn: bool = False,
                       k_max: int = K_MAX_DEFAULT,
                       dedup: bool = False):
    """Alle Paare in der (alpha, r_max)-Nachbarschaft. Reine Funktion.

    t, x, y : 1D-Arrays gleicher Laenge (float64 empfohlen -- KDTree)
    alpha   : µs, zeitliche Reichweite
    r_max   : px, raeumliche Reichweite
    use_knn : Kappung auf die k_max naechsten Nachbarn (MODELLIERUNGS-
              ENTSCHEIDUNG, nicht nur Optimierung -- BA_NN §2.2)
    k_max   : nur bei use_knn
              Bei geom != "ellipsoid" wird mit dem groesseren Kugelradius
              gesucht und danach gefiltert -- es koennen daher weniger als
              k_max Nachbarn uebrig bleiben.
    dedup   : doppelte ungerichtete Kanten entfernen. Bei use_knn NOETIG
              (die k-NN-Abfrage liefert i->j und j->i getrennt), bei
              query_pairs ein No-op (liefert bereits i<j eindeutig).
              ACHTUNG: sortiert die Kantenreihenfolge um.
    geom: welche Raum-Zeit-Form die Nachbarschaft hat.
      "ellipsoid"   (Δx²+Δy²)/r_max² + Δt²/α² ≤ 1
                    Was der KDTree-Trick von sich aus liefert. Der erlaubte
                    Radius SCHRUMPFT mit wachsendem Δt und ist bei Δt = α
                    null -- physikalisch die falsche Richtung.
      "cylinder"    |Δt| ≤ α  UND  Δr ≤ r_max
                    Die in BA_NN_Kantenkosten.md §2 dokumentierte Form.
      "hyperboloid" |Δt| ≤ α  UND  Δr ≤ √(r_max² + (v_max·Δt)²)
                    Folgt der Herleitung §2.1: die Taille ist die
                    Wolkenausdehnung, die Aufweitung die Partikeldrift.
                    Geht asymptotisch in den Kegel ueber, ohne dessen Knick.

    Es gilt  ellipsoid ⊆ cylinder ⊆ hyperboloid  (in tests/test_graph.py
    geprueft). Der KDTree kann nur Kugeln -- fuer cylinder/hyperboloid wird
    daher mit einer UMSCHLIESSENDEN Kugel gesucht und exakt nachgefiltert.
    Preis: ~1.9x mehr Kandidaten beim Zylinder. Bei sehr grossen Fenstern
    ist das der RAM-Peak, nicht die Endkantenzahl.

    Rueckgabe: (i, j) int64-Arrays, i = frueheres Event.
    """
    if geom not in NEIGHBORHOOD_MODES:
        raise ValueError(f"geom muss in {NEIGHBORHOOD_MODES} liegen")

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

    # Radius der Kugel, die den jeweiligen Koerper umschliesst.
    # Im skalierten Raum entspricht |Δt| ≤ α gerade |s·Δt| ≤ r_max.
    if geom == "ellipsoid":
        q = r_max
    elif geom == "cylinder":
        q = np.sqrt(2.0) * r_max
    else:
        q = np.hypot(np.hypot(r_max, v_max * alpha), r_max)

    if use_knn:
        _, idx = tree.query(pts, k=k_max + 1,
                            distance_upper_bound=q, workers=-1)
        i = np.repeat(np.arange(n), k_max)
        j = idx[:, 1:].ravel()
        ok = j < n
        i, j = i[ok], j[ok]
    else:
        pairs = tree.query_pairs(r=q, output_type="ndarray")
        if len(pairs) == 0:
            return empty
        i, j = pairs[:, 0], pairs[:, 1]

    if geom != "ellipsoid":
        dt = np.abs(t[i] - t[j])
        dr2 = (x[i] - x[j])**2 + (y[i] - y[j])**2
        grenze = (r_max**2 if geom == "cylinder"
                  else r_max**2 + (v_max * dt)**2)
        ok = (dt <= alpha) & (dr2 <= grenze)
        i, j = i[ok], j[ok]

    if len(i) == 0:
        return empty

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
