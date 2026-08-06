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

from eventpiv.graph import (edge_deltas, neighborhood_pairs, polarity_mask,
                            NEIGHBORHOOD_GEOMS, V_MAX_DEFAULT)


def brute_force_geom(t, x, y, alpha, r_max, geom, v_max=V_MAX_DEFAULT):
    """O(n^2)-Referenz fuer alle drei Formen -- unabhaengig vom KDTree."""
    s = r_max / alpha
    out = []
    for a in range(len(t)):
        for b in range(a + 1, len(t)):
            dt = abs(t[a] - t[b])
            dr2 = (x[a] - x[b])**2 + (y[a] - y[b])**2
            if geom == "ellipsoid":
                drin = dr2 + (s * dt)**2 <= r_max**2
            elif geom == "cylinder":
                drin = (dt <= alpha) and (dr2 <= r_max**2)
            else:
                drin = (dt <= alpha) and (dr2 <= r_max**2 + (v_max * dt)**2)
            if drin:
                out.append((a, b) if t[a] <= t[b] else (b, a))
    return sorted(out)


@pytest.mark.parametrize("geom", NEIGHBORHOOD_GEOMS)
@pytest.mark.parametrize("alpha,r_max", [(3_000, 5.0), (10_000, 7.06),
                                         (20_000, 3.0)])
def test_alle_geometrien_gegen_brute_force(wolke, geom, alpha, r_max):
    """Die umschliessende Kugel plus Nachfilter muss exakt den Koerper
    treffen -- zu grosse Kugel waere nur langsam, eine zu KLEINE wuerde
    still Kanten verlieren."""
    t, x, y = wolke
    i, j = neighborhood_pairs(t, x, y, alpha, r_max, geom=geom)
    assert als_menge(i, j) == brute_force_geom(t, x, y, alpha, r_max, geom)

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


def test_geometrien_sind_ineinander_enthalten(wolke):
    """ellipsoid ⊆ cylinder ⊆ hyperboloid -- die Formen sind geschachtelt.

    Ersetzt den frueheren verify-Test: dass die Quaderbedingung das
    Ellipsoid nicht filtert, ist der Spezialfall ellipsoid ⊆ cylinder.
    """
    t, x, y = wolke
    mengen = {m: set(als_menge(*neighborhood_pairs(t, x, y, 10_000, 7.0,
                                                   geom=m)))
              for m in ("ellipsoid", "cylinder", "hyperboloid")}
    assert mengen["ellipsoid"] <= mengen["cylinder"] <= mengen["hyperboloid"]
    assert len(mengen["cylinder"]) > len(mengen["ellipsoid"])


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
            assert (min(v, u), max(v, u)) in ungerichtet(knn)


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

# --- polarity_mask: die Knotenmenge ------------------------------------

def test_polarity_both_laesst_alles_durch():
    p = np.array([1, 0, 1, 1, 0], np.uint8)
    assert polarity_mask(p, "both").all()


def test_polarity_on_nimmt_genau_die_positiven():
    p = np.array([1, 0, 1, 1, 0], np.uint8)
    np.testing.assert_array_equal(polarity_mask(p, "on"),
                                  [True, False, True, True, False])


def test_maske_hat_immer_eventlaenge_und_ist_boolesch():
    """Sie indiziert vier Arrays gleichzeitig (t, x, y, GT) -- eine
    Index-Maske statt einer booleschen wuerde dort still etwas anderes tun."""
    for pol in ("on", "both"):
        m = polarity_mask(np.array([1, 0, 1], np.uint8), pol)
        assert m.dtype == bool and len(m) == 3


def test_leere_eingabe():
    for pol in ("on", "both"):
        assert len(polarity_mask(np.empty(0, np.uint8), pol)) == 0


def test_unbekannte_polaritaet_faellt_auf():
    """Doppelt mit Neighborhood.__post_init__ -- absichtlich, wie bei geom:
    die Funktion ist rein und auch ohne Konfiguration aufrufbar."""
    with pytest.raises(ValueError, match="polarity"):
        polarity_mask(np.array([1, 0], np.uint8), "off")


def test_on_only_entfernt_die_gemischtpolaren_kanten(wolke):
    """Der Sinn des Filters, an der Kantenmenge gemessen.

    Die ON- und die OFF-Sichel eines Partikels liegen rund 6 px
    auseinander; die Kanten ZWISCHEN ihnen sind die schwaechste Bindung
    innerhalb eines Partikels und reissen zuerst. Unter "on" bleiben nur
    Kanten uebrig, deren beide Endpunkte positiv sind -- und die Menge ist
    genau die Teilmenge der vollen, nicht eine andere.
    """
    t, x, y = wolke
    rng = np.random.default_rng(7)
    p = rng.integers(0, 2, len(t)).astype(np.uint8)

    voll = set(als_menge(*neighborhood_pairs(t, x, y, 10_000, 7.0)))

    m = polarity_mask(p, "on")
    idx = np.flatnonzero(m)                 # lokal -> global
    i, j = neighborhood_pairs(t[m], x[m], y[m], 10_000, 7.0)
    gefiltert = ungerichtet(zip(idx[i].tolist(), idx[j].tolist()))

    assert gefiltert <= ungerichtet(voll)
    assert gefiltert == {(a, b) for a, b in ungerichtet(voll)
                         if p[a] == 1 and p[b] == 1}
    assert len(gefiltert) < len(voll)