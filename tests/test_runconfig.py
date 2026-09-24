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

    from eventpiv.kern.runconfig import PairSampling

    with pytest.raises(ValueError, match="ja_vorrat"):
        PairSampling(ja_vorrat=0.0)
    with pytest.raises(ValueError, match="ja_frac"):
        PairSampling(ja_frac=1.5)


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


def test_kandidaten_v_max_muss_zur_nachbarschaft_passen():
    """kandidaten.v_max und neighborhood.v_max sind dieselbe Groesse in zwei
    Gruppen. Wird eine Gruppe vergessen, baute [6a] still einen
    Kandidatensatz mit dem Wert eines anderen Datensatzes. KandidatenCfg
    bricht deshalb zur Konfigurationszeit ab -- ausser bei io.strict=false."""
    from eventpiv.kern.runconfig import IO, Kandidaten, KandidatenCfg

    nb = Neighborhood(v_max=1.79e-3)
    KandidatenCfg()                                      # Vorgaben passen
    with pytest.raises(ValueError, match="v_max"):
        KandidatenCfg(neighborhood=nb)                   # Gruppe vergessen
    KandidatenCfg(neighborhood=nb, kandidaten=Kandidaten(v_max=0.00179))
    KandidatenCfg(neighborhood=nb, io=IO(strict=False))  # bewusster Akt


def test_weight_ist_kein_gueltiger_imbalance_modus():
    """Die Gewichtung war seit Wochen unbenutzt und tat in beiden Stufen
    Verschiedenes -- Stufe 1 gewichtete nur, Stufe 2 zog zusaetzlich. Ein
    alter Aufruf muss zur Konfigurationszeit abbrechen, nicht still ein
    anderes Verhalten liefern."""
    from eventpiv.kern.runconfig import Training, Transformer

    with pytest.raises(ValueError, match="imbalance"):
        Training(imbalance="weight")
    with pytest.raises(ValueError, match="imbalance"):
        Transformer(imbalance="weight")


def test_fit_typcast_und_auswahlwerte():
    """r_mess=8 und 8.0 duerfen nicht auseinanderlaufen, und ein Filter
    mit q_c = 0 teilt durch null."""
    from eventpiv.kern.runconfig import Fit

    assert canonical(Fit(r_mess=8)) == canonical(Fit(r_mess=8.0))
    for feld in ("r_mess", "q_c", "dt_aus", "init_us", "min_dauer_us"):
        with pytest.raises(ValueError, match=feld):
            Fit(**{feld: 0.0})


def test_fit_default_ist_der_gemessene_arbeitspunkt():
    """r_mess und q_c sind gemessen, nicht gewaehlt (Journal 12.09.).
    Faellt der Test, wurde der Arbeitspunkt ohne Messung verschoben."""
    from eventpiv.kern.runconfig import Fit

    f = Fit()
    assert f.r_mess == 8.0 and f.q_c == 20.0
    assert (f.c0, f.c1) == (0.95, 29.1)