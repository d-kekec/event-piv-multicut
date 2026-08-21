#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graph.py -- die (alpha, r_max)-Nachbarschaft. EINE Definition fuer alle Nutzer.

Zentraler Zweck: Paarbau [2] und Solve [4] MUESSEN dieselbe Kantenmenge
erzeugen -- sonst sieht das Netz im Einsatz eine andere Nachbarschaft als
im Training. Strukturell erzwungen dadurch, dass beide diese Funktion
rufen.

GEOMETRIE: Der KDTree arbeitet auf (x, y, t*s) mit s = r_max/alpha und
liefert mit Suchradius r_max ein ELLIPSOID. Ueber `geom` sind drei Formen
waehlbar.

KONTRAKT der Rueckgabe (auf den sich alle Aufrufer verlassen):
  - i, j sind Index-Arrays gleicher Laenge
  - t[i] <= t[j] fuer jedes Paar  (i = FRUEHERES Event)
    -> dt = t[j]-t[i] >= 0, wie im Training vorausgesetzt
  - keine Selbstkanten (i != j)
  - jede ungerichtete Kante genau einmal
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from eventpiv.kern.runconfig import (NEIGHBORHOOD_GEOMS, POLARITIES, Neighborhood,
                                Solver)

# EINE Quelle fuer diese Zahlen: runconfig. Die Namen bleiben hier, weil
# Tests und Standardargumente sie benutzen -- der Wert kommt von dort.
_NB_DEFAULT   = Neighborhood()
V_MAX_DEFAULT = _NB_DEFAULT.v_max     # px/µs, p99.9 der Partikelgeschwindigkeit
K_MAX_DEFAULT = Solver().k_max        # k-NN-Kappung


def time_scale(alpha: float, r_max: float) -> float:
    """Skalierung der Zeitachse, damit alpha in der Metrik r_max entspricht."""
    return r_max / alpha


def polarity_mask(p, polarity: str) -> np.ndarray:
    """Welche Events als KNOTEN zugelassen sind. Boolesche Maske.

    Liegt aus demselben Grund hier wie neighborhood_pairs: Stufe [2] und
    Stufe [4] muessen dieselbe Knotenmenge bilden, und eine Abweichung
    faellt an keiner Kennzahl auf.

    "on"   nur positive Events. Die Kanten zwischen ON- und OFF-Sichel sind
           die schwaechste Bindung innerhalb eines Partikels und reissen
           zuerst; sie fallen hier ganz weg.
    "both" alle Events.

    Gefiltert wird der GRAPH, nicht der Datensatz -- Stufe 2 braucht beide
    Zentroiden fuer Position und Richtung.

    ANWENDUNG: erst auf dem ungefilterten Strom ausrichten
    (evaluate.align_gt_window), dann dieselbe Maske auf beide Seiten. Ein
    gefilterter Strom ist eine Teilfolge, und die Ausrichtung laeuft sofort
    auseinander.
    """
    p = np.asarray(p)
    if polarity == "both":
        return np.ones(len(p), dtype=bool)
    if polarity == "on":
        return p == 1
    # Doppelt mit Neighborhood.__post_init__ -- absichtlich, wie bei geom:
    # die Funktion ist rein und auch ohne Konfiguration aufrufbar.
    raise ValueError(f"polarity muss in {POLARITIES} liegen")


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
    use_knn : Kappung auf die k_max naechsten Nachbarn. MODELLIERUNGS-
              ENTSCHEIDUNG, nicht nur Optimierung: an einer Kreuzung sind
              die raeumlich naechsten Nachbarn gerade die Events des ANDEREN
              Partikels -- die trennenden Kanten fielen weg.
    k_max   : nur bei use_knn. Bei geom != "ellipsoid" wird mit dem
              groesseren Kugelradius gesucht und danach gefiltert, es
              koennen also weniger als k_max Nachbarn uebrig bleiben.
    dedup   : doppelte ungerichtete Kanten entfernen. Bei use_knn NOETIG
              (die Abfrage liefert i->j und j->i getrennt), bei query_pairs
              ein No-op. ACHTUNG: sortiert die Kantenreihenfolge um.
    geom: die Raum-Zeit-Form der Nachbarschaft.
      "ellipsoid"   (Δx²+Δy²)/r_max² + Δt²/α² ≤ 1
                    Was der KDTree-Trick von sich aus liefert. Der erlaubte
                    Radius SCHRUMPFT mit wachsendem Δt und ist bei Δt = α
                    null -- physikalisch die falsche Richtung.
      "cylinder"    |Δt| ≤ α  UND  Δr ≤ r_max          (Arbeitspunkt)
      "hyperboloid" |Δt| ≤ α  UND  Δr ≤ √(r_max² + (v_max·Δt)²)
                    Die Taille ist die Wolkenausdehnung, die Aufweitung die
                    Partikeldrift. Bei kleinem alpha praktisch der Zylinder.

    Es gilt ellipsoid ⊆ cylinder ⊆ hyperboloid. Der KDTree kann nur Kugeln;
    fuer die beiden weiteren Formen wird mit einer UMSCHLIESSENDEN Kugel
    gesucht und exakt nachgefiltert. Eine zu grosse Kugel ist nur langsam,
    eine zu KLEINE verliert still Kanten -- deshalb der Brute-Force-
    Vergleich in tests/test_graph.py.

    Rueckgabe: (i, j) int64-Arrays, i = frueheres Event.
    """
    if geom not in NEIGHBORHOOD_GEOMS:
        # Doppelt mit Neighborhood.__post_init__ -- absichtlich: die
        # Funktion ist rein und auch ohne Konfiguration aufrufbar.
        raise ValueError(f"geom muss in {NEIGHBORHOOD_GEOMS} liegen")

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
        # j != i zusaetzlich: bei zwei Events auf exakt derselben (x, y, t)
        # kann Spalte 0 der andere Punkt sein.
        ok = (j < n) & (j != i)
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

    # NICHT auf zwei Zeilen aufteilen: die rechte Seite einer Tupelzuweisung
    # wird vollstaendig ausgewertet, bevor zugewiesen wird. Getrennt saehe
    # das zweite np.where das bereits vertauschte i, und die Haelfte aller dt
    # kehrte still das Vorzeichen um.
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

    Einzige Stelle, an der aus Indexpaaren Rohgroessen werden. Die
    Feature-Modi bauen darauf auf (model.build_features).
    """
    return (t[j] - t[i]), (x[j] - x[i]), (y[j] - y[i])


def jaccard_paare(i, j, n, chunk: int = 50_000):
    """Anteil gemeinsamer Nachbarn je Kante: |N(i) ∩ N(j)| / |N(i) ∪ N(j)|.

    Liegt aus demselben Grund hier wie neighborhood_pairs: [2] und [4]
    muessen dieselbe Rechnung benutzen. Die Nachbarschaft IST die
    Kantenmenge, kein zweiter KDTree.

    i, j : Indexpaare aus neighborhood_pairs, VOR jedem Filter -- solve
        kennt keinen Status und sieht alle Events als Knoten.
    n    : Knotenzahl des Fensters.

    Symmetrisch unter i<->j und skalenfrei: der Nenner kuerzt die lokale
    Dichte weg, an der raw6 gescheitert ist.

    Blockweise ueber die KANTEN, nicht ueber die Nachbarn: A[i_c] traegt
    die vollstaendigen Zeilen, das Ergebnis ist exakt. chunk steuert nur
    den Zwischenspeicher.

    Rueckgabe: float32-Array der Laenge len(i), Werte in [0, 1].
    """
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    if len(i) == 0:
        return np.empty(0, np.float32)

    # Beide Richtungen: A[k] soll die volle Nachbarschaft von k tragen,
    # neighborhood_pairs liefert jede Kante aber nur einmal.
    A = csr_matrix((np.ones(2 * len(i), np.int8),
                    (np.concatenate([i, j]), np.concatenate([j, i]))),
                   shape=(n, n))
    # getnnz statt sum(axis=1): zaehlt Eintraege, nicht Werte -- gegen
    # doppelte Kanten unempfindlich.
    grad = A.getnnz(axis=1).astype(np.float64)

    c = np.empty(len(i), np.float64)
    for s in range(0, len(i), chunk):
        sl = slice(s, min(s + chunk, len(i)))
        c[sl] = np.asarray(A[i[sl]].multiply(A[j[sl]]).sum(axis=1)).ravel()

    vereinigung = grad[i] + grad[j] - c
    return (c / np.maximum(vereinigung, 1.0)).astype(np.float32)
