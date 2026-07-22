#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Invariantentests fuer graph.neighborhood_pairs.

Kernstueck ist der Brute-Force-Vergleich: eine offensichtlich korrekte
O(n^2)-Referenz gegen die KDTree-Implementierung. Das prueft die Kernlogik
absolut (nicht relativ zu einer Referenzdatei) und bleibt gueltig, auch wenn
sich Parameter oder Implementierung aendern.
"""
import numpy as np
import pytest

from eventpiv.graph import edge_deltas, neighborhood_pairs


def brute_force(t, x, y, alpha, r_max):
    """O(n^2)-Referenz. Bildet die KUGEL-Bedingung ab, die der Code
    tatsaechlich verwendet (s. graph.py-Docstring), NICHT die Quader-
    Bedingung aus BA_NN_Kantenkosten.md §2."""
    s = r_max / alpha
    out = []
    for a in range(len(t)):
        for b in range(a + 1, len(t)):
            d2 = ((x[a]-x[b])**2 + (y[a]-y[b])**2 + (s*(t[a]-t[b]))**2)
            if d2 <= r_max**2:
                out.append((a, b) if t[a] <= t[b] else (b, a))
    return sorted(out)


def als_menge(i, j):
    return sorted(zip(i.tolist(), j.tolist()))

def ungerichtet(paare):
    return {(min(a, b), max(a, b)) for a, b in paare}


@pytest.fixture
def wolke():
    """40 Events, zufaellig, aber deterministisch. Skalen wie im Datensatz:
    Positionen um 100 px, Zeiten ueber 20 ms."""
    rng = np.random.default_rng(42)
    t = np.sort(rng.uniform(0, 20_000, 40))
    x = rng.uniform(90, 115, 40)
    y = rng.uniform(90, 115, 40)
    return t, x, y


def test_gegen_brute_force(wolke):
    """Der wichtigste Test: KDTree-Ergebnis == offensichtlich korrekte Referenz."""
    t, x, y = wolke
    i, j = neighborhood_pairs(t, x, y, alpha=10_000, r_max=7.0)
    assert als_menge(i, j) == brute_force(t, x, y, 10_000, 7.0)


def test_gegen_brute_force_mehrere_parameter(wolke):
    t, x, y = wolke
    for alpha, r_max in [(3_000, 5.0), (5_000, 6.0), (10_000, 7.06)]:
        i, j = neighborhood_pairs(t, x, y, alpha=alpha, r_max=r_max)
        assert als_menge(i, j) == brute_force(t, x, y, alpha, r_max), \
            f"Abweichung bei alpha={alpha}, r_max={r_max}"


def test_kontrakt_i_ist_frueheres_event(wolke):
    """dt >= 0 -- darauf verlaesst sich das gesamte Feature-Design."""
    t, x, y = wolke
    i, j = neighborhood_pairs(t, x, y, alpha=10_000, r_max=7.0)
    assert (t[i] <= t[j]).all()
    dt, _, _ = edge_deltas(t, x, y, i, j)
    assert (dt >= 0).all()


def test_keine_selbstkanten_keine_duplikate(wolke):
    t, x, y = wolke
    i, j = neighborhood_pairs(t, x, y, alpha=10_000, r_max=7.0)
    assert (i != j).all()
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    assert len(np.unique(lo * len(t) + hi)) == len(i)


def test_verify_filtert_nichts(wolke):
    """Belegt die Aussage aus dem graph.py-Docstring: die Quader-Nachpruefung
    ist per Konstruktion ein No-op, weil die Kugel sie impliziert."""
    t, x, y = wolke
    ohne = neighborhood_pairs(t, x, y, 10_000, 7.0, verify=False)
    mit  = neighborhood_pairs(t, x, y, 10_000, 7.0, verify=True)
    np.testing.assert_array_equal(ohne[0], mit[0])
    np.testing.assert_array_equal(ohne[1], mit[1])


def test_dedup_aendert_die_kantenmenge_nicht(wolke):
    """Ohne k-NN liefert query_pairs bereits eindeutige Kanten -> dedup
    veraendert nur die Reihenfolge, nicht die Menge."""
    t, x, y = wolke
    ohne = neighborhood_pairs(t, x, y, 10_000, 7.0, dedup=False)
    mit  = neighborhood_pairs(t, x, y, 10_000, 7.0, dedup=True)
    assert als_menge(*ohne) == als_menge(*mit)


def test_knn_behaelt_die_naechsten_nachbarn(wolke):
    """k-NN darf nur WEGLASSEN, und zwar nur die ENTFERNTEN Nachbarn.

    Kein Test auf den Knotengrad: der k-NN-Graph ist asymmetrisch. Jeder
    Knoten fragt k_max Nachbarn ab, kann aber von beliebig vielen anderen
    ausgewaehlt werden -- zentral gelegene Events werden zu Hubs. Der Grad
    ist nach oben nur durch n-1 begrenzt.
    """
    t, x, y = wolke
    alpha, r_max, k = 10_000, 7.0, 3
    s = r_max / alpha

    voll = set(als_menge(*neighborhood_pairs(t, x, y, alpha, r_max)))
    knn  = set(als_menge(*neighborhood_pairs(t, x, y, alpha, r_max,
                                             use_knn=True, k_max=k, dedup=True)))

    assert knn <= voll                      # erfindet keine Kanten
    assert len(knn) <= len(t) * k           # Obergrenze der Kantenzahl

    def d(a, b):
        return np.hypot(np.hypot(x[a]-x[b], y[a]-y[b]), s*(t[a]-t[b]))

    for v in range(len(t)):
        nachbarn = [b if a == v else a for a, b in voll if v in (a, b)]
        nachbarn.sort(key=lambda u: d(v, u))
        for u in nachbarn[:k]:              # die k naechsten MUESSEN bleiben
            assert tuple(sorted((v, u), key=lambda z: (t[z], z))) in knn \
                   or (min(v, u), max(v, u)) in ungerichtet(knn)


def test_monotonie_in_r_max(wolke):
    """Groesseres r_max => Obermenge (bei mitskaliertem alpha bleibt s gleich)."""
    t, x, y = wolke
    klein = set(als_menge(*neighborhood_pairs(t, x, y, 5_000, 5.0)))
    gross = set(als_menge(*neighborhood_pairs(t, x, y, 10_000, 10.0)))
    assert klein <= gross


def test_isolierte_punkte_erzeugen_keine_kanten():
    """Drei Events, weit auseinander -> gar keine Nachbarschaft."""
    t = np.array([0.0, 5_000.0, 10_000.0])
    x = np.array([10.0, 500.0, 1000.0])
    y = np.array([10.0, 500.0, 1000.0])
    i, j = neighborhood_pairs(t, x, y, alpha=10_000, r_max=7.0)
    assert len(i) == 0 and len(j) == 0


def test_leere_und_einzelne_eingabe():
    leer = np.empty(0)
    i, j = neighborhood_pairs(leer, leer, leer, 10_000, 7.0)
    assert len(i) == 0 and len(j) == 0
    ein = np.array([1.0])
    i, j = neighborhood_pairs(ein, ein, ein, 10_000, 7.0)
    assert len(i) == 0 and len(j) == 0


def test_determinismus(wolke):
    t, x, y = wolke
    a = neighborhood_pairs(t, x, y, 10_000, 7.0)
    b = neighborhood_pairs(t, x, y, 10_000, 7.0)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
