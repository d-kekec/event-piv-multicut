#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pfadableitung: rechnet, legt nichts an."""
import pytest

from eventpiv import config
from eventpiv.runconfig import (Neighborhood, PairSampling, Selection, Solver,
                                Training, Window)


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