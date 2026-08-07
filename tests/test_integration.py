#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Die ganze Kette [1] -> [2] -> [3] -> [4] ueber einen echten Schreibpfad.

WOZU. Unit-Tests koennen eine Fehlerklasse strukturell nicht fangen: die,
bei der eine Konfiguration ein PLAUSIBLES Ergebnis erzeugt. Beide
dokumentierten Vorfaelle gehoeren dazu -- animate_clusters war seit dem
29.07. kaputt, und ein Doku-Commit loeschte `return loader.load_delta_t(dt)`
in data.py. pytest blieb beide Male gruen, gefunden hat sie der Anker.

Dieser Test ersetzt den Anker NICHT. Er laeuft auf synthetischen Daten und
prueft die VERDRAHTUNG -- Pfade, Formate, Aufrufreihenfolge, Vertraege. Er
kann nicht wissen, welche Zahl richtig ist; das kann nur der Anker. Die
beiden beantworten verschiedene Fragen und ersetzen einander nicht.

KEINE HYDRA. Die Konfigurationen werden direkt gebaut, wie in allen
anderen Tests -- das ist die Probe darauf, dass das Framework in scripts/
geblieben ist (Code_Karte §3).

SOLVER: die CC-Baseline und device="cpu". Damit braucht der Test weder
RAMA noch eine freie GPU. Was RAMA auf einer bekannten Instanz liefert,
steht in RAMA_Anbindung §5.1 und wird hier nicht nachgeprueft.

Laufzeit rund zehn Sekunden -- der langsamste Test der Suite und der
einzige, der Dateien schreibt.
"""
import json

import numpy as np
import pytest

from eventpiv import config
from eventpiv.labels import LABEL_FORMAT
from eventpiv.pairs import build_pair_dataset
from eventpiv.runconfig import (IO, Compute, LabelCfg, Labeling, Neighborhood,
                                PairCfg, PairSampling, Selection, SolveCfg,
                                Solver, TrainCfg, Training, Window, slug)
from eventpiv.solve import solve_and_evaluate
from eventpiv.train import train_edge_nn

# --- Der Arbeitspunkt des Tests ---------------------------------------
# Nicht die Vorgabewerte: die sind auf 8-s-Runs geeicht, die Fixture ist
# 1 s lang. r_max = 5 und alpha = 10 000 liegen zugleich im Zielregime von
# Stufe 1.
NB = dict(alpha=10_000, r_max=5.0, geom="cylinder")

# Nur die Kreuzungs-Runs: saubere liefern physikalisch null NEIN, und die
# Kreuzungs-Runs tragen selbst reichlich JA. Zwei Runs sind das Minimum
# fuer einen run-getrennten Holdout.
# Schwellen grob, nicht scharf: sie fangen "gar nichts gesammelt", nicht
# eine Abweichung um zehn Prozent.
MIND_JA, MIND_NEIN = 1_000, 200


def _gruppen(mini_datensatz):
    nb = Neighborhood(**NB)
    ps = PairSampling(split="train", runs=list(mini_datensatz.kreuzung),
                      ja_frac=0.02, max_yes=200_000, max_no=200_000, seed=0)
    tr = Training(feature_mode="dr", imbalance="sample", sample_ratio=3.0,
                  hidden=4, epochs=3, batch=4096, val_frac=0.2, seed=0)
    return nb, ps, tr


def test_kette_labels_paare_training_solve(mini_datensatz):
    nb, ps, tr = _gruppen(mini_datensatz)
    cp = Compute(device="cpu", cost_batch=1_000_000)

    # ---------------------------------------------------------------
    # [1] Events -> track_id + status
    # Leere Run-Liste heisst ALLE Runs des Splits -- das prueft zugleich
    # config.available_runs gegen echte Dateien.
    # ---------------------------------------------------------------
    from eventpiv.labels import build_labels
    failed = build_labels(LabelCfg(io=IO(), labeling=Labeling(),
                                   selection=Selection(split="train")),
                          verbose=False)
    assert failed == []

    for run_nr in (1, 2, 3, 4):
        lf = config.label_file("train", run_nr)
        assert lf.exists(), f"{lf} fehlt"
        with np.load(lf) as z:
            meta = json.loads(str(z["meta"]))
            assert meta["label_format"] == LABEL_FORMAT
            assert meta["n"] == len(z["t"])
            # p wird mitgespeichert -- ON-only filtert spaeter DARAUS,
            # nicht aus dem Eventstrom (Code_Guide §2.4).
            assert set(np.unique(z["p"])) <= {0, 1}

    # ---------------------------------------------------------------
    # [2] Paar-Datensatz
    # ---------------------------------------------------------------
    pcfg = PairCfg(io=IO(), neighborhood=nb, pairs=ps, compute=cp)

    gesehen = []
    ppath, daten = build_pair_dataset(pcfg, verbose=False,
                                      on_summary=gesehen.append)

    assert ppath == config.pairs_file(nb, ps)
    assert ppath.exists()
    # Die Senke ist ein Vertrag, kein Beiwerk: scripts/ haengt daran W&B.
    assert len(gesehen) == 1
    k = gesehen[0]
    assert k["n_ja"] > MIND_JA and k["n_nein"] > MIND_NEIN
    # Labels entstehen fuer alle vier Runs, Paare nur aus den beiden
    # Kreuzungs-Runs: saubere Runs liefern physikalisch null NEIN.
    assert {z["run"] for z in k["per_run"]} == set(mini_datensatz.kreuzung)

    with np.load(ppath) as d:
        assert int((d["label"] == 1).sum()) == k["n_ja"]
        assert int((d["label"] == 0).sum()) == k["n_nein"]
        # Der Vertrag aus graph.py, eine Ebene weiter: dt >= 0 ueberall.
        assert (d["dt"] >= 0).all()
        pmeta = json.loads(str(d["meta"]))
        assert pmeta["neighborhood"]["alpha"] == nb.alpha

    # ---------------------------------------------------------------
    # [3] Training
    # ---------------------------------------------------------------
    tcfg = TrainCfg(io=IO(), neighborhood=nb, pairs=ps, training=tr,
                    compute=cp)

    start, epochen = [], []
    mpath, kennzahlen = train_edge_nn(tcfg, verbose=False,
                                      on_start=start.append,
                                      on_epoch=epochen.append)

    assert mpath == config.model_file(nb, ps, tr)
    assert mpath.exists()
    assert len(start) == 1 and len(epochen) == tr.epochs
    # Der Holdout ist run-getrennt und stratifiziert -- beide Klassen
    # muessen drin sein, sonst waeren AUC und TPR NaN.
    assert not np.isnan(kennzahlen["holdout"]["auc"])
    # Zwei Runs, val_frac=0.2 -> genau einer in den Holdout, einer bleibt
    # im Training. split_runs prueft, dass beide Seiten beide Klassen haben.
    assert len(start[0]["holdout_runs"]) == 1

    metrics = mpath.with_suffix(".metrics.json")
    assert metrics.exists()
    m = json.loads(metrics.read_text(encoding="utf-8"))
    assert set(m) >= {"identity", "data", "history", "final"}
    assert len(m["history"]) == tr.epochs

    # ---------------------------------------------------------------
    # [4] Solve mit der CC-Baseline
    # ---------------------------------------------------------------
    sel = Selection(split="train", run_nr=1)
    win = Window(t0=450_000, dt=20_000)     # mitten in der Kreuzung
    sv = Solver(mode="P", backend="cc")
    scfg = SolveCfg(io=IO(), neighborhood=nb, pairs=ps, training=tr,
                    solver=sv, selection=sel, window=win, compute=cp)

    info = solve_and_evaluate(scfg, verbose=False)

    assert info["backend"] == "cc"
    assert info["n_events"] > 0 and info["n_edges"] > 0
    # on_edges hat gefeuert: die GT-Senke reicht diese Felder herein.
    assert "edge_ja_frac" in info and "edge_bewertbar_frac" in info
    # cluster_metrics hat gerechnet.
    assert 0.0 <= info["purity_cluster_mean"] <= 1.0
    assert info["frag_mean"] >= 1.0
    assert info["n_gt"] >= 2

    # ---------------------------------------------------------------
    # Der Artefaktbaum IST die Abhaengigkeitskette (Code_Karte §4).
    # ---------------------------------------------------------------
    wp = mini_datensatz.out / "wp" / slug(nb)
    assert ppath == wp / "pairs" / slug(ps) / "pairs.npz"
    assert mpath == wp / "pairs" / slug(ps) / slug(tr) / "model.pt"
    # Stufe [4] schreibt hier NICHTS: solve_and_evaluate gibt info zurueck,
    # die JSON legt scripts/solve_with_nn.py an. Die Kette schreibt also bis
    # einschliesslich [3] echte Dateien, und die letzte Ausgabe bleibt
    # jenseits der Framework-Grenze. Geprueft wird deshalb der ABGELEITETE
    # Pfad -- samt _cc-Suffix, das den Anker-Dateinamen unberuehrt laesst.
    sjson = config.solve_file(nb, ps, tr, sel, win, sv)
    assert sjson == (mpath.parent / "solves"
                     / "train_run1_t450000_dt20000_P_cc.json")
    assert not sjson.parent.exists(), \
        "Pfadableitung hat ein Verzeichnis angelegt -- create=False gilt nicht"

    # Labels liegen DANEBEN, nicht unter wp/ -- sie haengen von keinem
    # Nachbarschaftsparameter ab.
    assert config.label_file("train", 1).parent.parent.parent == \
        mini_datensatz.out


def test_falsche_nachbarschaft_faellt_beim_solve_auf(mini_datensatz):
    """Netz und Einsatz-Nachbarschaft passen nicht -- genau die Pruefung,
    die es geben soll.

    Erreichbar nur ueber io.model_path: bei abgeleitetem Pfad gaebe es das
    Netz unter der falschen Nachbarschaft gar nicht, und der Lauf brechae
    schon mit FileNotFoundError ab. Der Notausgang auf dem LESEpfad ist
    hier also nicht Bequemlichkeit, sondern die einzige Art, die Pruefung
    ueberhaupt auszuloesen.
    """
    nb, ps, tr = _gruppen(mini_datensatz)
    cp = Compute(device="cpu", cost_batch=1_000_000)

    from eventpiv.labels import build_labels
    build_labels(LabelCfg(io=IO(), labeling=Labeling(),
                          selection=Selection(split="train")), verbose=False)
    pcfg = PairCfg(io=IO(), neighborhood=nb, pairs=ps, compute=cp)
    build_pair_dataset(pcfg, verbose=False)
    tcfg = TrainCfg(io=IO(), neighborhood=nb, pairs=ps, training=tr,
                    compute=cp)
    mpath, _ = train_edge_nn(tcfg, verbose=False)

    anders = Neighborhood(**{**NB, "alpha": NB["alpha"] * 2})
    scfg = SolveCfg(io=IO(model_path=str(mpath)), neighborhood=anders,
                    pairs=ps, training=tr, solver=Solver(backend="cc"),
                    selection=Selection(split="train", run_nr=1),
                    window=Window(t0=450_000, dt=20_000), compute=cp)

    with pytest.raises(ValueError, match="inkonsistent"):
        solve_and_evaluate(scfg, verbose=False)