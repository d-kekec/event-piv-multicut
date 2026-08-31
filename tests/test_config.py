#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pfadableitung: rechnet, legt nichts an."""
import pytest

from eventpiv.kern import config
from eventpiv.kern.runconfig import (Kandidaten, Neighborhood, PairSampling,
                                Selection, Solver, Solver2, Training,
                                Transformer, Window)


@pytest.fixture
def gruppen():
    return Neighborhood(), PairSampling(), Training()


def test_pfadfunktionen_legen_nichts_an(tmp_path, monkeypatch, gruppen):
    """Ein Pfad zu FRAGEN darf nichts erzeugen.

    Sonst findet ein Lauf mit falschen Overrides ein leeres, selbst
    angelegtes Verzeichnis vor und meldet "0/9 loesbar" -- ein
    Konfigurationsfehler, der wie eine Messung aussieht.
    """
    nb, ps, tr = gruppen
    monkeypatch.setattr(config, "OUT_ROOT", tmp_path)
    sel, win, sv = Selection(split="train", run_nr=3), Window(), Solver()

    pfade = [
        config.wp_dir(nb),
        config.pairs_file(nb, ps),
        config.model_file(nb, ps, tr),
        config.solve_file(nb, ps, tr, sel, win, sv),
        config.tracklet_file(nb, ps, tr, sel, sv),
        config.kandidaten_file(nb, ps, tr, sel, sv, Kandidaten()),
        config.transformer_file(nb, ps, tr, sv, Kandidaten(), Transformer()),
        config.solve2_file(nb, ps, tr, sv, Kandidaten(), Transformer(), sel,
                           Solver2()),
        config.label_dir("train"),
        config.label_file("train", 3),
        config.results_dir(),
    ]
    for p in pfade:
        assert str(p).startswith(str(tmp_path)), f"{p} liegt ausserhalb OUT_ROOT"

    assert list(tmp_path.iterdir()) == [], \
        "Pfadableitung hat Verzeichnisse angelegt"


def test_create_true_legt_nur_das_verzeichnis_an(tmp_path, monkeypatch, gruppen):
    """create=True legt das VERZEICHNIS an, nie die Datei."""
    nb, ps = gruppen[0], gruppen[1]
    monkeypatch.setattr(config, "OUT_ROOT", tmp_path)

    p = config.pairs_file(nb, ps, create=True)
    assert p.parent.is_dir()
    assert not p.exists()


def test_project_root_traegt_die_marke():
    """Nicht die Ebenenzahl pruefen, sondern die Aussage: PROJECT_ROOT ist
    das Verzeichnis mit der pyproject.toml. Ein Umzug von config.py darf
    daran nichts aendern -- mit parents[N] taete er es still."""
    assert (config.PROJECT_ROOT / "pyproject.toml").is_file()
    assert (config.PROJECT_ROOT / "src" / "eventpiv").is_dir()


def test_stufe2_haengt_unter_tracklets(tmp_path, monkeypatch, gruppen):
    """Stufe 2 liest das Tracklet-Artefakt (A) -- der Baum IST die
    Abhaengigkeitskette, also muss ihr Artefakt darunter haengen.
    """
    nb, ps, tr = gruppen
    monkeypatch.setattr(config, "OUT_ROOT", tmp_path)
    sel, sv = Selection(split="train", run_nr=3), Solver()
    kd, tf = Kandidaten(), Transformer()

    kp = config.kandidaten_file(nb, ps, tr, sel, sv, kd)
    mp = config.transformer_file(nb, ps, tr, sv, kd, tf)
    tracklets = config.tracklet_file(nb, ps, tr, sel, sv).parent

    for p in (kp, mp):
        assert tracklets in p.parents, f"{p} haengt nicht unter tracklets/"

    # Der Kandidatensatz ist die EINGABE des Netzes, und der Baum IST die
    # Abhaengigkeitskette -- also muss das Netz darunter liegen.
    assert kp.parent in mp.parents

    # Zwei Kandidatensaetze mit verschiedener Stichprobengroesse duerfen
    # sich nicht ueberschreiben.
    assert config.kandidaten_file(nb, ps, tr, sel, sv,
                                  Kandidaten(n_events=64)) != kp


def test_stufe2_pfade_legen_nur_verzeichnisse_an(tmp_path, monkeypatch, gruppen):
    """create=True legt das Verzeichnis an, nie die Datei.

    Die Signaturen weichen von den uebrigen ab: transformer_file nimmt
    kein Selection (ein Netz gilt fuer alle Runs), solve2_file nimmt es an
    siebter statt vierter Stelle. Ein Fehler darin faellt sonst erst beim
    Schreiben auf.
    """
    nb, ps, tr = gruppen
    monkeypatch.setattr(config, "OUT_ROOT", tmp_path)
    sel, sv = Selection(split="train", run_nr=3), Solver()
    kd, tf, sv2 = Kandidaten(), Transformer(), Solver2()

    for p in (config.kandidaten_file(nb, ps, tr, sel, sv, kd, create=True),
              config.transformer_file(nb, ps, tr, sv, kd, tf, create=True),
              config.solve2_file(nb, ps, tr, sv, kd, tf, sel, sv2,
                                 create=True)):
        assert p.parent.is_dir()
        assert not p.exists()