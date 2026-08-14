#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests fuer runconfig: kanonische Form, Identitaet, Auswahlwerte.

Die kanonische Form ist der Konsistenz-VERTRAG (check_consistency vergleicht
sie) UND die Grundlage jedes Artefaktpfads -- zwei Dinge, die dieselbe
Rechnung benutzen. Genau deshalb muss sie geprueft sein.
"""
import pytest

from eventpiv.kern.runconfig import (Neighborhood, canonical, same_neighborhood,
                                slug)


def test_kanonische_form_ignoriert_int_float():
    """3 und 3.0 duerfen NICHT auseinanderlaufen -- sonst erzeugte derselbe
    Arbeitspunkt zwei Verzeichnisse, und check_consistency schluege an."""
    assert canonical(Neighborhood(r_max=3)) == canonical(Neighborhood(r_max=3.0))


def test_v_max_gehoert_zur_identitaet():
    """Auch wenn v_max nur bei geom=hyperboloid wirkt: ueberkonservativ ist
    zulaessig, eine bedingte Identitaetsregel waere eine spaetere Falle."""
    a, b = Neighborhood(), Neighborhood(v_max=2e-4)
    assert not same_neighborhood(a, b)
    assert slug(a) != slug(b)


def test_unbekannte_auswahlwerte_fallen_zur_konfigurationszeit_auf():
    """Frueher fiel das erst tief im Lauf auf -- da stand der Slug schon und
    ein Verzeichnis hiess danach."""
    from eventpiv.kern.runconfig import Training

    with pytest.raises(ValueError, match="geom"):
        Neighborhood(geom="wuerfel")
    with pytest.raises(ValueError, match="feature_mode"):
        Training(feature_mode="quatsch")
    with pytest.raises(ValueError, match="imbalance"):
        Training(imbalance="egal")


def test_stufe2_typcast_wie_bei_neighborhood():
    """r_max=15 und r_max=15.0 duerfen nicht zwei Verzeichnisse ergeben --
    dieselbe Luecke, die _Typed bei Neighborhood schliesst. Ueber Hydra
    faellt sie nie auf, bei direkter Konstruktion in Tests und
    Auswertungsskripten schon."""
    from eventpiv.kern.runconfig import Kandidaten

    assert canonical(Kandidaten(r_max=15)) == canonical(Kandidaten(r_max=15.0))


def test_stufe2_gruppen_pruefen_ihre_auswahlwerte():
    """Wie bei Neighborhood.geom: geprueft wird zur KONFIGURATIONSZEIT.
    Faellt es erst im Lauf auf, steht der Slug schon und ein Verzeichnis
    heisst danach."""
    from eventpiv.kern.runconfig import Kandidaten, Transformer

    with pytest.raises(ValueError, match="geom"):
        Kandidaten(geom="wuerfel")
    with pytest.raises(ValueError, match="min_events"):
        Kandidaten(min_events=1)
    with pytest.raises(ValueError, match="n_events"):
        Kandidaten(n_events=0)
    with pytest.raises(ValueError, match="pooling"):
        Transformer(pooling="max")
    with pytest.raises(ValueError, match="imbalance"):
        Transformer(imbalance="egal")


def test_d_model_teilbar_durch_n_heads():
    """Die Kopfbreite ist d_model/n_heads. Geht das nicht auf, bricht erst
    torch beim Bau des Netzes ab -- also nach dem Laden der Daten und lange
    nachdem der Slug feststand."""
    from eventpiv.kern.runconfig import Transformer

    with pytest.raises(ValueError, match="n_heads"):
        Transformer(d_model=64, n_heads=5)
    Transformer(d_model=64, n_heads=4)       # muss durchlaufen


def test_stufe2_identitaet_haengt_an_jedem_feld():
    """Der lesbare Kern nennt fuenf bzw. vier Felder, den Rest traegt der
    Hash. Zwei Kandidatensaetze, die sich nur in min_events unterscheiden,
    duerfen nicht denselben Pfad bekommen -- sie enthalten verschiedene
    Knoten."""
    from eventpiv.kern.runconfig import Kandidaten, Transformer

    k = Kandidaten()
    for abweichend in (Kandidaten(min_events=20), Kandidaten(seed=1),
                       Kandidaten(v_max=2e-4), Kandidaten(n_events=64)):
        assert slug(abweichend) != slug(k)

    t = Transformer()
    for abweichend in (Transformer(lr=1e-4), Transformer(dropout=0.1),
                       Transformer(d_ff=512)):
        assert slug(abweichend) != slug(t)