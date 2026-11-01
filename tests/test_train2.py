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
from eventpiv.kette.train2 import (_epochengroesse, _holdout_index, _toepfe,
                                   _ziehe_epoche, paar_ziele, split_nach_runs)


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


def test_rauschknoten_fallen_in_allen_modi_heraus():
    gt = np.array([7, -1], np.int64)
    n_gt = np.array([1, 0], np.int32)
    a = np.array([1.0, 0.0])
    for modus in ("streng", "mehrheit"):
        _, g = paar_ziele(gt, n_gt, a, np.array([0]), np.array([1]),
                          modus=modus, min_anteil=0.9)
        assert not g[0], modus


RUNS = [1, 2, 3, 4, 5]
JA, NEIN = [15] * 5, [5] * 5


def test_holdout_waehlt_ganze_runs():
    vr = split_nach_runs(RUNS, JA, NEIN, val_frac=0.4, seed=0)
    assert len(vr) == 2
    assert set(vr.tolist()) <= set(RUNS)
    assert (np.diff(vr) > 0).all()


def test_fest_gesetzte_runs_bleiben_im_training():
    """Die GROESSE des Holdouts folgt allen Runs, nur die Auswahl nicht."""
    vr = split_nach_runs(RUNS, JA, NEIN, val_frac=0.4, seed=0, fest=[2, 3])
    assert len(vr) == 2
    assert not np.isin([2, 3], vr).any()


def test_unbekannter_fester_run_bricht_ab():
    with pytest.raises(SystemExit, match="nicht im Datensatz"):
        split_nach_runs(RUNS, JA, NEIN, val_frac=0.4, seed=0, fest=[7])


def test_zu_wenige_waehlbare_runs_brechen_ab():
    with pytest.raises(SystemExit, match="waehlbar"):
        split_nach_runs(RUNS, JA, NEIN, val_frac=0.4, seed=0,
                        fest=[1, 2, 3, 4])


def test_holdout_bricht_bei_einer_klasse_ab():
    """Eine Seite ohne NEIN macht TPR, TNR und AUC ueber den GANZEN Lauf
    wertlos -- das ist kein Schoenheitsfehler und darf nicht als nan
    durchlaufen."""
    with pytest.raises(SystemExit, match="nur eine Klasse"):
        split_nach_runs([1, 2], [10, 0], [0, 10], val_frac=0.5, seed=0)


def test_toepfe_und_holdout_wie_die_rechnung_auf_allen_paaren():
    """Dieselben Paare in derselben Reihenfolge wie die direkte Rechnung
    ueber Arrays je Paar. Die Reihenfolge zaehlt: gezogen wird nach
    Position."""
    rng = np.random.default_rng(5)
    runs = np.array([7, 3, 9, 4])
    n_roh = np.array([40, 0, 25, 60])
    run = np.repeat(runs, n_roh)
    gueltig = rng.random(len(run)) < 0.8
    ziel_roh = rng.random(len(run)) < 0.7
    ist_val_paar = np.isin(run, [9])
    idx_t = np.flatnonzero(gueltig & ~ist_val_paar)
    soll_ja = idx_t[ziel_roh[idx_t]]
    soll_nein = idx_t[~ziel_roh[idx_t]]
    soll_v = np.flatnonzero(gueltig & ist_val_paar)

    # Nur die gueltigen, je Run zusammen -- wie lade_kandidaten sie ablegt.
    grenzen = np.concatenate([[0], np.cumsum(
        [int(gueltig[run == r].sum()) for r in runs])])
    ja, nein = _toepfe(ziel_roh[gueltig], grenzen, np.isin(runs, [9]))
    v = _holdout_index(grenzen, np.isin(runs, [9]))
    global_ = np.flatnonzero(gueltig)       # kompakt -> alle Paare
    np.testing.assert_array_equal(global_[ja], soll_ja)
    np.testing.assert_array_equal(global_[nein], soll_nein)
    np.testing.assert_array_equal(global_[v], soll_v)


def _artefakt(pfad, rng, m, p, n_ev=4):
    """Ein kleines [6a]-Artefakt mit den Feldern, die lade_kandidaten liest."""
    np.savez_compressed(
        pfad, t_min=rng.uniform(0, 1e6, m), x_c=rng.uniform(0, 1280, m),
        y_c=rng.uniform(0, 720, m),
        ev=rng.normal(size=(m, n_ev, 3)).astype(np.float32),
        maske=rng.random((m, n_ev)) < 0.8,
        paar_i=rng.integers(0, m, p).astype(np.int32),
        paar_j=rng.integers(0, m, p).astype(np.int32),
        gt_track_id=rng.integers(0, 4, m).astype(np.int64),
        n_gt=rng.choice([0, 1, 2], m, p=[0.1, 0.8, 0.1]).astype(np.int32),
        gt_anteil=np.where(rng.random(m) < 0.8, 1.0, 0.7))


def test_lade_kandidaten_wie_die_rechnung_auf_allen_paaren(tmp_path,
                                                         monkeypatch):
    """Knoten unveraendert und in Ladereihenfolge; Paare sind die gueltigen
    der direkten Rechnung, mit Knotenversatz und in Artefaktreihenfolge.
    Ein Run ohne Paare bleibt ein leerer Block."""
    from eventpiv.kern import config
    from eventpiv.kern.runconfig import Kandidaten
    from eventpiv.kette.train2 import lade_kandidaten

    rng = np.random.default_rng(11)
    # Run: (Knoten, Paare). Run 1 hat mehr Knoten, als es Paare gibt --
    # so kann ein Paarversatz nie zufaellig dem Knotenversatz gleichen.
    groessen = {3: (5, 30), 1: (40, 0), 2: (6, 45)}
    for r, (m, p) in groessen.items():
        _artefakt(tmp_path / f"run{r}.npz", rng, m, p)
    monkeypatch.setattr(config, "kandidaten_file",
                        lambda *a, **k: tmp_path / f"run{a[3].run_nr}.npz")
    cfg = Stufe2Cfg(kandidaten=Kandidaten(runs=[3, 1, 2]))
    d = lade_kandidaten(cfg)

    knoten = {k: [] for k in ("ev", "maske", "t_min", "x_c", "y_c")}
    i, j, z, g = [], [], [], []
    versatz = 0
    for r in (3, 1, 2):
        with np.load(tmp_path / f"run{r}.npz") as f:
            zr, gr = paar_ziele(f["gt_track_id"], f["n_gt"], f["gt_anteil"],
                                f["paar_i"], f["paar_j"],
                                modus=cfg.transformer.label_modus,
                                min_anteil=cfg.transformer.gt_min_anteil)
            for k in knoten:
                knoten[k].append(f[k])
            i.append(f["paar_i"] + versatz)
            j.append(f["paar_j"] + versatz)
            z.append(zr)
            g.append(gr)
            versatz += len(f["t_min"])
    g = np.concatenate(g)
    for k in ("ev", "maske"):
        np.testing.assert_array_equal(getattr(d, k).numpy(),
                                      np.concatenate(knoten[k]))
    for k in ("t_min", "x_c", "y_c"):
        np.testing.assert_array_equal(
            getattr(d, k).numpy(),
            np.concatenate(knoten[k]).astype(np.float32))
    np.testing.assert_array_equal(d.paar_i, np.concatenate(i)[g])
    np.testing.assert_array_equal(d.paar_j, np.concatenate(j)[g])
    np.testing.assert_array_equal(d.ziel, np.concatenate(z)[g] >= 0.5)
    np.testing.assert_array_equal(d.n_roh, [30, 0, 45])
    np.testing.assert_array_equal(np.diff(d.grenzen) > 0,
                                  [True, False, True])  # Run 1 ist leer
    # Die Daten tragen beide Klassen, und der Filter hat etwas entfernt.
    assert 0 < d.ziel.sum() < len(d.ziel) < len(g)

    
def test_vorhandener_checkpoint_bricht_ab(tmp_path, monkeypatch):
    """Der Schutz muss VOR lade_kandidaten greifen. Danach waere er
    wertlos: die Rechenzeit fuer das Laden ist dann schon ausgegeben."""
    from eventpiv.kern import config
    from eventpiv.kette import train2

    ziel = tmp_path / "model_P.pt"
    ziel.write_bytes(b"")
    monkeypatch.setattr(config, "transformer_file", lambda *a, **k: ziel)
    monkeypatch.setattr(train2, "lade_kandidaten", lambda *a, **k:
                        pytest.fail("lade_kandidaten darf nicht laufen"))
    with pytest.raises(SystemExit, match="existiert bereits"):
        train2.train_stufe2(Stufe2Cfg())


def test_paar_ziele_unabhaengig_vom_index_dtype():
    """paar_i/paar_j liegen als int32 vor, um Speicher zu sparen. Ein
    stiller dtype-Wechsel darf die Ziele nicht beruehren."""
    gt, n_gt, a = _knoten()
    z64, g64 = paar_ziele(gt, n_gt, a, I.astype(np.int64),
                          J.astype(np.int64), modus="mehrheit", min_anteil=0.5)
    z32, g32 = paar_ziele(gt, n_gt, a, I.astype(np.int32),
                          J.astype(np.int32), modus="mehrheit", min_anteil=0.5)
    np.testing.assert_array_equal(g64, g32)
    np.testing.assert_allclose(z64, z32)


def test_epochengroesse_haelt_das_verhaeltnis_unter_dem_deckel():
    """Der Deckel verkleinert beide Klassen im selben Verhaeltnis -- sonst
    waere er ein Eingriff in den Betriebspunkt statt eine Stichprobe."""
    ohne = _epochengroesse(98_000_000, 3_850_000, 3.0, None)
    mit = _epochengroesse(763_000_000, 191_000_000, 3.0, 10_000_000)
    assert ohne == (11_550_000, 3_850_000, True)
    assert mit[0] + mit[1] == 10_000_000
    assert mit[0] / mit[1] == pytest.approx(3.0)
    assert mit[2]


def test_epochengroesse_meldet_unerreichbare_ratio():
    """Reicht der JA-Topf nicht, folgt der Betriebspunkt den Daten. Das
    muss als Feld sichtbar sein, nicht nur als Druckzeile."""
    n_ja, n_nein, erreicht = _epochengroesse(1_000, 5_000, 3.0, None)
    assert (n_ja, n_nein) == (1_000, 5_000)
    assert not erreicht


def test_ziehe_epoche_trifft_die_klassen_und_ist_reproduzierbar():
    ja, nein = np.arange(0, 8_000), np.arange(8_000, 10_000)
    a = _ziehe_epoche(ja, nein, 300, 100, np.random.default_rng(7))
    b = _ziehe_epoche(ja, nein, 300, 100, np.random.default_rng(7))
    c = _ziehe_epoche(ja, nein, 300, 100, np.random.default_rng(8))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c), "Seed ohne Wirkung"
    assert int((a < 8_000).sum()) == 300 and int((a >= 8_000).sum()) == 100
    assert (np.diff(a) > 0).all(), "sortiert und ohne Doppelte"


def test_ziehe_epoche_wechselt_ueber_epochen():
    """Der Sinn der Neuziehung: nicht dieselben Paare in jeder Epoche."""
    ja, nein = np.arange(0, 8_000), np.arange(8_000, 10_000)
    rng = np.random.default_rng(0)
    mengen = [set(_ziehe_epoche(ja, nein, 300, 100, rng).tolist())
              for _ in range(5)]
    assert len(set.union(*mengen)) > 2 * len(mengen[0])


def test_weich_ist_kein_gueltiger_modus():
    """label_modus="weich" gibt es nicht: der Anteil des
    Minderheitspartikels steht nicht im Artefakt. Ein Aufruf damit muss
    zur Konfigurationszeit abbrechen."""
    from eventpiv.kern.runconfig import Transformer
    with pytest.raises(ValueError, match="label_modus"):
        Transformer(label_modus="weich")