#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests fuer runconfig: abgeleitete Defaults, kanonische Form, Identitaet.

Die kanonische Form ist der Konsistenz-VERTRAG (check_consistency vergleicht
sie) UND die Grundlage jedes Artefaktpfads -- zwei Dinge, die dieselbe
Rechnung benutzen. Genau deshalb muss sie geprueft sein.
"""
import pytest

from eventpiv.runconfig import (Neighborhood, PairSampling, Tracklets,
                                canonical, resolve, resolve_tracklets,
                                same_neighborhood, slug, win_dt_for)


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
    darf nicht still durch den Default ersetzt werden."""
    ps = resolve(Neighborhood(), PairSampling(win_dt=0, win_dt_yes=0))
    assert ps.win_dt == 0 and ps.win_dt_yes == 0


def test_resolve_leitet_ab():
    nb = Neighborhood(alpha=10_000, r_max=7.06)
    ps = resolve(nb, PairSampling())
    assert ps.win_dt == 40_000 and ps.win_dt_yes == 40_000


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

# --- Tracklets: das Fenster von Stufe 1 --------------------------------

def test_tracklets_unaufgeloest_faellt_beim_slug_auf():
    """slug() ruft short() IMMER auf. Ein unaufgeloestes dt ergaebe still
    einen anderen Hash fuer denselben Arbeitspunkt -- dieselbe Engstelle
    wie bei PairSampling."""
    with pytest.raises(RuntimeError, match="unaufgeloest"):
        slug(Tracklets())


def test_resolve_tracklets_nimmt_alpha():
    """Die Konvention alpha = tracklet_dt ist ein Vorgabewert, keine Regel
    im Kopf -- die Zahl existiert an genau einer Stelle."""
    tk = resolve_tracklets(Neighborhood(alpha=5_000), Tracklets())
    assert tk.dt == 5_000


def test_resolve_tracklets_ueberschreibt_null_nicht():
    """`is None` statt `or`: dt=0 ist ungueltig, aber ANGEGEBEN. Es soll
    dort auffallen, wo es wirkt, statt hier verdeckt zu werden."""
    tk = resolve_tracklets(Neighborhood(alpha=5_000), Tracklets(dt=0))
    assert tk.dt == 0


def test_abweichendes_verhaeltnis_bleibt_moeglich():
    tk = resolve_tracklets(Neighborhood(alpha=10_000), Tracklets(dt=3_000))
    assert tk.dt == 3_000


def test_slug_unterscheidet_tracklet_dt():
    assert slug(Tracklets(dt=5_000)) != slug(Tracklets(dt=10_000))


def test_tracklet_dt_wird_auf_int_gecastet():
    """Wie r_max=3 gegen 3.0: sonst bekaeme derselbe Arbeitspunkt zwei
    Verzeichnisse."""
    assert canonical(Tracklets(dt=5_000)) == canonical(Tracklets(dt=5_000.0))


# --- _Derivable: die Aufloesung folgt den vorhandenen Gruppen ----------

def test_derivable_loest_pairs_weiterhin_auf():
    from eventpiv.runconfig import PairCfg

    cfg = PairCfg()
    cfg.resolve_derived()
    assert cfg.pairs.win_dt is not None and cfg.pairs.win_dt_yes is not None


def test_derivable_loest_beide_gruppen_auf():
    """Eine Stufe mit pairs UND tracklets -- die gibt es noch nicht, der
    Zweig aber schon. Ohne diesen Test faellt eine vergessene Aufloesung
    erst auf, wenn der Treiber existiert."""
    from dataclasses import dataclass, field

    from eventpiv.runconfig import _Derivable

    @dataclass
    class Beides(_Derivable):
        neighborhood: Neighborhood = field(default_factory=Neighborhood)
        pairs: PairSampling = field(default_factory=PairSampling)
        tracklets: Tracklets = field(default_factory=Tracklets)

    cfg = Beides()
    cfg.resolve_derived()
    assert cfg.pairs.win_dt is not None
    assert cfg.tracklets.dt == cfg.neighborhood.alpha


def test_derivable_ohne_abgeleitete_gruppe_bricht_ab():
    """Faengt eine umbenannte oder vergessene Gruppe. Ohne den Abbruch
    liefe resolve_derived still ins Leere, und short() braeche erst Stufen
    spaeter mit einer Meldung ueber ein unaufgeloestes Feld."""
    from dataclasses import dataclass, field

    from eventpiv.runconfig import _Derivable

    @dataclass
    class Leer(_Derivable):
        neighborhood: Neighborhood = field(default_factory=Neighborhood)

    with pytest.raises(RuntimeError, match="abgeleiteten"):
        Leer().resolve_derived()