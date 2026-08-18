#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ziele, Masken und der Holdout-Split.

Die Schleife selbst wird hier nicht geprueft -- sie braucht Artefakte.
Geprueft wird, was still falsch sein kann: welche Paare ein Ziel
bekommen, welchen Wert, und dass der Holdout nach RUNS trennt.
"""
import numpy as np
import pytest

from eventpiv.kern.runconfig import Stufe2Cfg
from eventpiv.kette.train2 import paar_ziele, split_nach_runs, _untersampeln


def _knoten():
    """Vier Knoten: rein 7, rein 7, rein 9, verschmolzen 7 (Anteil 0,6)."""
    gt = np.array([7, 7, 9, 7], np.int64)
    n_gt = np.array([1, 1, 1, 2], np.int32)
    anteil = np.array([1.0, 1.0, 1.0, 0.6])
    return gt, n_gt, anteil


I = np.array([0, 0, 0, 1, 2])
J = np.array([1, 2, 3, 3, 3])
#             7-7  7-9  7-M  7-M  9-M


def test_streng_schliesst_verschmolzene_aus():
    gt, n_gt, a = _knoten()
    z, g = paar_ziele(gt, n_gt, a, I, J, modus="streng", min_anteil=0.9)
    np.testing.assert_array_equal(g, [True, True, False, False, False])
    np.testing.assert_allclose(z, [1.0, 0.0, 0.0, 0.0, 0.0])


def test_mehrheit_torbedingung_greift():
    """gt_anteil = 0,6 liegt unter der Schwelle: der verschmolzene Knoten
    bleibt draussen. Bei 0,5 als Schwelle kaeme er herein."""
    gt, n_gt, a = _knoten()
    _, g9 = paar_ziele(gt, n_gt, a, I, J, modus="mehrheit", min_anteil=0.9)
    z5, g5 = paar_ziele(gt, n_gt, a, I, J, modus="mehrheit", min_anteil=0.5)
    np.testing.assert_array_equal(g9, [True, True, False, False, False])
    np.testing.assert_array_equal(g5, [True] * 5)
    # Hart nach Mehrheit: der verschmolzene zaehlt als Partikel 7.
    np.testing.assert_allclose(z5, [1.0, 0.0, 1.0, 1.0, 0.0])


def test_weich_gibt_zwischenwerte():
    """Der Zielwert ist die Wahrscheinlichkeit, dass zwei zufaellig
    gezogene Events zum selben Partikel gehoeren. Reine Paare bekommen
    dabei wieder exakt 1."""
    gt, n_gt, a = _knoten()
    z, g = paar_ziele(gt, n_gt, a, I, J, modus="weich", min_anteil=0.9)
    assert g.all()
    np.testing.assert_allclose(z, [1.0, 0.0, 0.6, 0.6, 0.0])


def test_weich_bleibt_bei_verschiedener_mehrheit_hart():
    """Bekannte Unvollstaendigkeit: der Anteil des MINDERHEITSpartikels
    steht nicht im Artefakt, also bleibt (9, verschmolzen 7+9) auf 0,
    obwohl 0,4 richtig waere. Hier festgehalten, damit es nicht
    versehentlich als vollstaendig gilt."""
    gt, n_gt, a = _knoten()
    z, _ = paar_ziele(gt, n_gt, a, np.array([2]), np.array([3]),
                      modus="weich", min_anteil=0.9)
    assert z[0] == 0.0


def test_rauschknoten_fallen_in_allen_modi_heraus():
    gt = np.array([7, -1], np.int64)
    n_gt = np.array([1, 0], np.int32)
    a = np.array([1.0, 0.0])
    for modus in ("streng", "mehrheit", "weich"):
        _, g = paar_ziele(gt, n_gt, a, np.array([0]), np.array([1]),
                          modus=modus, min_anteil=0.9)
        assert not g[0], modus


def test_holdout_trennt_nach_runs():
    """Kein Paar eines Validierungsruns darf im Training landen."""
    run = np.repeat([1, 2, 3, 4, 5], 20).astype(np.int16)
    ziel = np.tile([1.0] * 15 + [0.0] * 5, 5).astype(np.float32)
    g = np.ones(len(run), bool)
    it, iv, vr = split_nach_runs(run, ziel, g, val_frac=0.4, seed=0)
    assert len(vr) == 2
    assert not np.isin(run[it], vr).any()
    assert np.isin(run[iv], vr).all()
    assert len(it) + len(iv) == len(run)

def test_fest_gesetzte_runs_bleiben_im_training():
    """Die GROESSE des Holdouts folgt allen Runs, nur die Auswahl nicht."""
    run = np.repeat([1, 2, 3, 4, 5], 20).astype(np.int16)
    ziel = np.tile([1.0] * 15 + [0.0] * 5, 5).astype(np.float32)
    g = np.ones(len(run), bool)
    it, _, vr = split_nach_runs(run, ziel, g, val_frac=0.4, seed=0,
                                fest=[2, 3])
    assert len(vr) == 2
    assert not np.isin([2, 3], vr).any()
    assert np.isin(run[it], [2, 3]).all(axis=None) is not None
    assert set(np.unique(run[it])) >= {2, 3}


def test_unbekannter_fester_run_bricht_ab():
    run = np.repeat([1, 2, 3, 4, 5], 20).astype(np.int16)
    ziel = np.tile([1.0] * 15 + [0.0] * 5, 5).astype(np.float32)
    with pytest.raises(SystemExit, match="nicht im Datensatz"):
        split_nach_runs(run, ziel, np.ones(100, bool), val_frac=0.4, seed=0,
                        fest=[7])


def test_zu_wenige_waehlbare_runs_brechen_ab():
    run = np.repeat([1, 2, 3, 4, 5], 20).astype(np.int16)
    ziel = np.tile([1.0] * 15 + [0.0] * 5, 5).astype(np.float32)
    with pytest.raises(SystemExit, match="waehlbar"):
        split_nach_runs(run, ziel, np.ones(100, bool), val_frac=0.4, seed=0,
                        fest=[1, 2, 3, 4])


def test_holdout_bricht_bei_einer_klasse_ab():
    """Eine Seite ohne NEIN macht TPR, TNR und AUC ueber den GANZEN Lauf
    wertlos -- das ist kein Schoenheitsfehler und darf nicht als nan
    durchlaufen."""
    run = np.repeat([1, 2], 10).astype(np.int16)
    ziel = np.concatenate([np.ones(10), np.zeros(10)]).astype(np.float32)
    with pytest.raises(SystemExit, match="nur eine Klasse"):
        split_nach_runs(run, ziel, np.ones(20, bool), val_frac=0.5, seed=0)


def test_untersampeln_trifft_das_verhaeltnis():
    ziel = np.concatenate([np.ones(1000), np.zeros(100)]).astype(np.float32)
    idx = np.arange(len(ziel))
    aus = _untersampeln(idx, ziel, 3.0, np.random.default_rng(0))
    assert int((ziel[aus] >= 0.5).sum()) == 300
    assert int((ziel[aus] < 0.5).sum()) == 100
    # NEIN bleibt vollstaendig -- es ist die knappe Klasse.
    assert set(np.flatnonzero(ziel < 0.5)) <= set(aus)

    
def test_vorhandener_checkpoint_bricht_ab(tmp_path, monkeypatch):
    """Der Schutz muss VOR lade_kandidaten greifen. Danach waere er
    wertlos: die Stunde Rechenzeit ist dann schon ausgegeben."""
    from eventpiv.kern import config
    from eventpiv.kette import train2

    ziel = tmp_path / "model_P.pt"
    ziel.write_bytes(b"")
    monkeypatch.setattr(config, "transformer_file", lambda *a, **k: ziel)
    monkeypatch.setattr(train2, "lade_kandidaten", lambda *a, **k:
                        pytest.fail("lade_kandidaten darf nicht laufen"))
    with pytest.raises(SystemExit, match="existiert bereits"):
        train2.train_stufe2(Stufe2Cfg())