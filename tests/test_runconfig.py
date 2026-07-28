#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests fuer runconfig: abgeleitete Defaults, kanonische Form, Identitaet.

Die kanonische Form ist seit dem 27.07.2026 der Konsistenz-VERTRAG
(check_consistency vergleicht sie) UND die Grundlage jedes Artefaktpfads.
Sie war bis hierher ungetestet -- eine Luecke genau an der Stelle, an der
zwei Dinge zusammenfallen, die frueher getrennt waren.
"""
import pytest

from eventpiv.runconfig import (Neighborhood, PairSampling, canonical,
                                resolve, same_neighborhood, slug, win_dt_for)


def test_win_dt_am_referenzpunkt():
    """Bei den Referenzparametern muss genau das Referenzfenster herauskommen."""
    assert win_dt_for(10_000, 7.06) == 40_000


def test_win_dt_waechst_bei_kleinerer_nachbarschaft():
    """Kleinere Nachbarschaft -> weniger NEIN je Zeit -> breiteres Fenster."""
    assert win_dt_for(3_000, 5.0) > win_dt_for(10_000, 7.06)


def test_win_dt_deckel():
    assert win_dt_for(100, 1.0, cap=300_000) == 300_000


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


def test_resolve_ueberschreibt_null_nicht():
    """`is None` statt `or`: win_dt=0 ist gueltig (wenn auch unsinnig) und
    darf nicht still durch den Default ersetzt werden (TODO C2)."""
    ps = resolve(Neighborhood(), PairSampling(win_dt=0, win_dt_yes=0))
    assert ps.win_dt == 0 and ps.win_dt_yes == 0


def test_resolve_leitet_ab():
    nb = Neighborhood(alpha=10_000, r_max=7.06)
    ps = resolve(nb, PairSampling())
    assert ps.win_dt == 40_000 and ps.win_dt_yes == 40_000