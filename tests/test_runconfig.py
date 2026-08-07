#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests fuer runconfig: kanonische Form, Identitaet, Auswahlwerte.

Die kanonische Form ist der Konsistenz-VERTRAG (check_consistency vergleicht
sie) UND die Grundlage jedes Artefaktpfads -- zwei Dinge, die dieselbe
Rechnung benutzen. Genau deshalb muss sie geprueft sein.
"""
import pytest

from eventpiv.runconfig import (Neighborhood, canonical, same_neighborhood,
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
    from eventpiv.runconfig import Training

    with pytest.raises(ValueError, match="geom"):
        Neighborhood(geom="wuerfel")
    with pytest.raises(ValueError, match="feature_mode"):
        Training(feature_mode="quatsch")
    with pytest.raises(ValueError, match="imbalance"):
        Training(imbalance="egal")