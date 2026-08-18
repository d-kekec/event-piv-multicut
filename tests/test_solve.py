#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Der CC-Vergleichssolver. Fuer solve.py gab es bisher keine Testdatei.

Geprueft wird die Baseline, nicht RAMA: RAMA braucht eine GPU, und was es
auf der bekannten Instanz liefert, steht in RAMA_Anbindung.md §5.1.
"""
import numpy as np
import pytest
import torch

from eventpiv.kern import config
from eventpiv.kern.runconfig import (IO, Compute, Neighborhood, PairSampling,
                                Selection, SolveCfg, Solver, Training, Window)
from eventpiv.kette.fenster import Fenster, lade_fenster
from eventpiv.kette.loeser import SOLVER_FUNKTIONEN, solve_cc
from eventpiv.kette.solve import solve_window


def test_formatzusicherungen():
    lab = solve_cc(np.array([0, 1]), np.array([1, 2]),
                   np.array([1.0, 1.0]), num_nodes=3)
    assert lab.dtype == np.int32 and len(lab) == 3


def test_negative_kanten_trennen():
    """Zwei Dreiergruppen, verbunden durch eine einzige negative Kante."""
    i = np.array([0, 1, 2, 3, 4, 3])
    j = np.array([1, 2, 3, 4, 5, 5])
    c = np.array([5.0, 5.0, -8.0, 4.0, 3.0, 2.0])
    lab = solve_cc(i, j, c, num_nodes=6)
    assert lab[0] == lab[1] == lab[2]
    assert lab[3] == lab[4] == lab[5]
    assert lab[0] != lab[3]


def test_stimmt_mit_rama_auf_der_bekannten_instanz_ueberein():
    """Die 6-Knoten-Instanz aus RAMA_Anbindung.md §5.1.

    RAMA liefert dort [0,0,0,1,1,1], beweisbar optimal. CC liefert dieselbe
    Partition -- die negative Kante im frustrierten Dreieck faellt weg, das
    Dreieck bleibt trotzdem zusammenhaengend. Der Test haelt fest, dass die
    Baseline nicht generell schlechter ist; sie ist es an einer bestimmten
    Stelle, und die zeigt der naechste Test.
    """
    i = np.array([0, 1, 0, 2, 3, 4, 3])
    j = np.array([1, 2, 2, 3, 4, 5, 5])
    c = np.array([5.0, 5.0, -1.0, -8.0, 4.0, 3.0, 2.0])
    lab = solve_cc(i, j, c, num_nodes=6)
    assert len(set(lab.tolist())) == 2
    assert lab[0] == lab[1] == lab[2] and lab[3] == lab[4] == lab[5]


def test_eine_positive_kante_verschmilzt_zwei_wolken():
    """Die Kernaussage der Baseline, und der Grund fuer ihre Existenz.

    Zwei Gruppen, zwischen denen zehn Kanten liegen: neun stark negativ,
    eine schwach positiv. Die Bilanz spricht klar fuer den Schnitt, und
    Multicut duerfte die positive Kante mitschneiden. CC kann das nicht --
    es kennt nur die Kantenpraeferenz, nie die Bilanz.
    """
    i = np.array([0, 1] + [0] * 10)
    j = np.array([1, 2] + [3] * 10)          # alles gegen Knoten 3
    c = np.array([5.0, 5.0] + [-9.0] * 9 + [0.5])
    lab = solve_cc(i, j, c, num_nodes=4)
    assert lab[0] == lab[3], "CC muesste hier verschmelzen"


def test_kantenlose_knoten_bekommen_minus_eins():
    """Gleiche Buchhaltung wie RAMA (dort ueber sanitize_graph), sonst sind
    n_clusters_rama und n_nodes_unassigned nicht vergleichbar."""
    lab = solve_cc(np.array([0]), np.array([1]), np.array([1.0]), num_nodes=4)
    assert lab[0] == lab[1] and lab[0] >= 0
    assert lab[2] == -1 and lab[3] == -1


def test_knoten_mit_nur_negativen_kanten_bleibt_zugeordnet():
    """Er ist allein, aber nicht unzugeordnet -- der Unterschied zaehlt."""
    lab = solve_cc(np.array([0]), np.array([1]), np.array([-1.0]), num_nodes=2)
    assert lab[0] >= 0 and lab[1] >= 0 and lab[0] != lab[1]


def test_kosten_null_bleiben_erhalten():
    """Dieselbe Lesart wie solve/frac_neg: 0 zaehlt als nicht-negativ."""
    lab = solve_cc(np.array([0]), np.array([1]), np.array([0.0]), num_nodes=2)
    assert lab[0] == lab[1]


def test_leere_instanz_faellt_auf():
    with pytest.raises(ValueError):
        solve_cc(np.empty(0, np.int64), np.empty(0, np.int64),
                 np.empty(0, np.float32), num_nodes=5)


def test_backend_waehlt_die_funktion():
    assert set(SOLVER_FUNKTIONEN) == {"rama", "cc"}
    assert SOLVER_FUNKTIONEN["cc"] is solve_cc


def test_unbekanntes_backend_faellt_zur_konfigurationszeit_auf():
    with pytest.raises(ValueError):
        Solver(backend="unionfind")


def test_vorgabe_steht_im_dateinamen():
    """Seit die Vorgabe -0,5 ist, traegt sie ein Suffix -- und muss es
    tragen: sonst laegen Laeufe bei 0,0 und bei -0,5 auf demselben Pfad.

    Der Anker bleibt unter "P" erreichbar, weil er cost_offset selbst
    ausschreibt.
    """
    assert Solver().short() == "P_o-0.5"
    assert Solver(cost_offset=0.0).short() == "P"
    assert Solver(backend="cc").short() == "P_cc_o-0.5"
    assert Solver(backend="cc", cost_offset=0.0).short() == "P_cc"

# --- Fenster und lade_fenster -----------------------------------------

def test_fenster_ohne_gt(mini_datensatz):
    f = lade_fenster(Selection(split="train", run_nr=1),
                     Window(t0=450_000, dt=20_000),
                     polarity="both", gt_path=None)
    assert len(f) > 0 and not f.hat_gt
    assert f.t.dtype == np.float64 and f.x.dtype == np.float64
    assert len(f.p) == len(f)


def test_fehlende_gt_datei_ist_kein_fehler(mini_datensatz):
    """Der Normalfall auf echten Messdaten -- dort gibt es keine Labels."""
    f = lade_fenster(Selection(split="train", run_nr=1),
                     Window(t0=450_000, dt=20_000), polarity="both",
                     gt_path=mini_datensatz.out / "gibtsnicht.npz")
    assert not f.hat_gt and f.track_id is None and f.status is None


def test_fenster_mit_gt_ist_ausgerichtet(mini_datensatz):
    """track_id und status haben Fensterlaenge, nicht Runlaenge -- die
    Ausrichtung ist die fehleranfaelligste Stelle der Stufe."""
    from eventpiv.kern.runconfig import LabelCfg, Labeling
    from eventpiv.kette.labels import build_labels
    build_labels(LabelCfg(io=IO(), labeling=Labeling(),
                          selection=Selection(split="train")), verbose=False)

    sel, win = Selection(split="train", run_nr=1), Window(t0=450_000,
                                                          dt=20_000)
    f = lade_fenster(sel, win, polarity="both",
                     gt_path=config.label_file("train", 1))
    assert f.hat_gt
    assert len(f.track_id) == len(f) and len(f.status) == len(f)
    # Mitten in der Kreuzung -- alle drei Statusklassen muessen vorkommen.
    assert set(np.unique(f.status)) == {0, 1, 2}


def test_solve_window_braucht_keinen_datensatz():
    """Der eigentliche Gewinn der Aufteilung: die Funktion laedt nichts
    mehr und laesst sich auf von Hand gebauten Arrays pruefen -- ohne
    Datensatz, ohne Label-Datei, ohne Pfade."""
    from eventpiv.kette.model import EdgeMLP, n_features

    rng = np.random.default_rng(0)
    # Zwei dichte Wolken, weit auseinander.
    t = np.repeat(np.arange(40, dtype=np.float64) * 100.0, 2)
    x = np.tile([100.0, 400.0], 40) + rng.normal(0, 1.0, 80)
    y = np.tile([100.0, 400.0], 40) + rng.normal(0, 1.0, 80)
    o = np.argsort(t, kind="stable")
    f = Fenster(t=t[o], x=x[o], y=y[o], p=np.ones(80, np.uint8))

    cfg = SolveCfg(io=IO(), neighborhood=Neighborhood(alpha=10_000, r_max=5.0,
                                                      geom="cylinder"),
                   pairs=PairSampling(), training=Training(),
                   solver=Solver(backend="cc"),
                   selection=Selection(split="train", run_nr=1),
                   window=Window(t0=0, dt=10_000),
                   compute=Compute(device="cpu", cost_batch=1_000_000))

    modell = EdgeMLP(n_features("dr"), 4)
    labels, info = solve_window(cfg, f, modell, torch.zeros(2),
                                torch.ones(2), "dr", verbose=False)
    assert info["n_edges"] > 0
    assert labels is not None and len(labels) == len(f)


def test_leeres_fenster_bricht_nicht_ab():
    from eventpiv.kette.model import EdgeMLP, n_features

    leer = np.empty(0, np.float64)
    f = Fenster(t=leer, x=leer, y=leer, p=np.empty(0, np.uint8))
    cfg = SolveCfg(io=IO(), neighborhood=Neighborhood(),
                   pairs=PairSampling(), training=Training(),
                   solver=Solver(backend="cc"),
                   selection=Selection(split="train", run_nr=1),
                   window=Window(),
                   compute=Compute(device="cpu"))
    labels, info = solve_window(cfg, f, EdgeMLP(n_features("dr"), 4),
                                torch.zeros(2), torch.ones(2), "dr",
                                verbose=False)
    assert labels is None and info["n_events"] == 0


def test_kantenstatistik_kommt_aus_dem_fenster():
    """Ohne Senke: traegt das Fenster eine GT, misst solve_window die
    Kanten dagegen -- und laesst es sonst."""
    from eventpiv.kette.model import EdgeMLP, n_features

    rng = np.random.default_rng(1)
    t = np.repeat(np.arange(40, dtype=np.float64) * 100.0, 2)
    x = np.tile([100.0, 400.0], 40) + rng.normal(0, 1.0, 80)
    y = np.tile([100.0, 400.0], 40) + rng.normal(0, 1.0, 80)
    o = np.argsort(t, kind="stable")
    tid = np.tile([7, 9], 40).astype(np.int16)[o]

    gem = dict(t=t[o], x=x[o], y=y[o], p=np.ones(80, np.uint8))
    cfg = SolveCfg(io=IO(), neighborhood=Neighborhood(alpha=10_000, r_max=5.0,
                                                      geom="cylinder"),
                   pairs=PairSampling(), training=Training(),
                   solver=Solver(backend="cc"),
                   selection=Selection(split="train", run_nr=1),
                   window=Window(t0=0, dt=10_000),
                   compute=Compute(device="cpu", cost_batch=1_000_000))
    modell, mu, sig = EdgeMLP(n_features("dr"), 4), torch.zeros(2), torch.ones(2)

    mit = Fenster(**gem, track_id=tid, status=np.zeros(80, np.uint8))
    _, info = solve_window(cfg, mit, modell, mu, sig, "dr", verbose=False)
    assert "edge_ja_frac" in info and "t_edges" in info

    ohne = Fenster(**gem)
    _, info2 = solve_window(cfg, ohne, modell, mu, sig, "dr", verbose=False)
    assert "edge_ja_frac" not in info2


# --- ON-only ----------------------------------------------------------

def test_polarity_on_laesst_nur_positive_events_durch(mini_datensatz):
    sel, win = Selection(split="train", run_nr=1), Window(t0=450_000,
                                                          dt=20_000)
    beide = lade_fenster(sel, win, polarity="both")
    nur_on = lade_fenster(sel, win, polarity="on")
    assert set(np.unique(nur_on.p)) == {1}
    assert len(nur_on) == int((beide.p == 1).sum())
    assert 0 < len(nur_on) < len(beide)


def test_gt_bleibt_unter_der_maske_ausgerichtet(mini_datensatz):
    """Erst ausrichten, dann filtern. Andersherum bekaeme align_gt_window
    eine Teilfolge und liefe sofort auseinander."""
    from eventpiv.kern.runconfig import LabelCfg, Labeling
    from eventpiv.kette.labels import build_labels
    build_labels(LabelCfg(io=IO(), labeling=Labeling(),
                          selection=Selection(split="train")), verbose=False)

    sel, win = Selection(split="train", run_nr=1), Window(t0=450_000,
                                                          dt=20_000)
    gt = config.label_file("train", 1)
    beide = lade_fenster(sel, win, polarity="both", gt_path=gt)
    nur_on = lade_fenster(sel, win, polarity="on", gt_path=gt)

    assert len(nur_on.track_id) == len(nur_on)
    np.testing.assert_array_equal(nur_on.track_id,
                                  beide.track_id[beide.p == 1])
    np.testing.assert_array_equal(nur_on.status, beide.status[beide.p == 1])


def test_polarity_ist_pflicht():
    """Ohne Vorgabewert: ein vergessener Filter faellt an keiner Kennzahl
    auf, also muss er beim Aufruf auffallen."""
    with pytest.raises(TypeError):
        lade_fenster(Selection(split="train", run_nr=1), Window())


# -------------------------------------------------------------------------------


def test_cost_offset_steht_im_dateinamen():
    """Ohne Suffix ueberschriebe ein Sweep seine eigenen Ergebnisse."""
    assert Solver(cost_offset=0.0).short() == "P"
    assert Solver(cost_offset=-0.5).short() == "P_o-0.5"
    assert Solver(cost_offset=1.0).short() == "P_o1"


def test_negativer_versatz_schneidet_mehr():
    """Der Versatz verschiebt die Schwelle, ohne die Rangfolge zu aendern --
    er kann also nur MEHR Kanten negativ machen, nie andere."""
    from eventpiv.kette.model import EdgeMLP, n_features

    rng = np.random.default_rng(3)
    t = np.repeat(np.arange(40, dtype=np.float64) * 100.0, 2)
    x = np.tile([100.0, 104.0], 40) + rng.normal(0, 1.0, 80)
    y = np.tile([100.0, 104.0], 40) + rng.normal(0, 1.0, 80)
    o = np.argsort(t, kind="stable")
    f = Fenster(t=t[o], x=x[o], y=y[o], p=np.ones(80, np.uint8))

    # EINMAL gebaut, nicht je Aufruf: EdgeMLP wird zufaellig initialisiert,
    # und zwei Netze liefern verschiedene Kosten. Der Test vergliche sonst
    # Netze statt Versaetze.
    torch.manual_seed(0)
    modell = EdgeMLP(n_features("dr"), 4)

    def loese(offset):
        cfg = SolveCfg(io=IO(),
                       neighborhood=Neighborhood(alpha=10_000, r_max=5.0,
                                                 geom="cylinder"),
                       pairs=PairSampling(), training=Training(),
                       solver=Solver(backend="cc", cost_offset=offset),
                       selection=Selection(split="train", run_nr=1),
                       window=Window(t0=0, dt=10_000),
                       compute=Compute(device="cpu", cost_batch=1_000_000))
        return solve_window(cfg, f, modell, torch.zeros(2), torch.ones(2),
                            "dr", verbose=False)[1]

    # Grenzwerte statt Zwischenwerte: welche Kosten ein UNTRAINIERTES Netz
    # liefert, ist Zufall der Initialisierung -- ein Test darauf pruefte
    # nicht den Versatz. Diese beiden Aussagen gelten fuer JEDE
    # Kostenverteilung, und genau das ist die Mechanik des Versatzes.
    ohne = loese(0.0)
    assert loese(-1e6)["frac_neg"] == 1.0        # alles negativ
    assert loese(+1e6)["frac_neg"] == 0.0        # nichts negativ
    for offset in (-1e6, +1e6):
        m = loese(offset)
        assert m["frac_neg_roh"] == ohne["frac_neg_roh"]
        assert m["cost_offset"] == offset


def test_solve_run_schreitet_ab(mini_datensatz):
    """Der Treiber auf dem Mini-Datensatz: 1 s bei alpha = 10 ms sind
    100 Fenster. Geprueft werden die Zusicherungen, nicht die Qualitaet."""
    from eventpiv.kern.runconfig import LabelCfg, Labeling
    from eventpiv.kette.labels import build_labels
    from eventpiv.kette.model import EdgeMLP, n_features
    from eventpiv.kette.tracklets import solve_run

    build_labels(LabelCfg(io=IO(), labeling=Labeling(),
                          selection=Selection(split="train")), verbose=False)

    nb = Neighborhood(alpha=10_000, r_max=5.0, geom="cylinder", polarity="on")
    cfg = SolveCfg(io=IO(), neighborhood=nb, pairs=PairSampling(),
                   training=Training(), solver=Solver(backend="cc"),
                   selection=Selection(split="train", run_nr=1),
                   window=Window(t0=None, dt=None),
                   compute=Compute(device="cpu", cost_batch=1_000_000))

    torch.manual_seed(0)
    gesehen = []
    tid, zeilen, lauf = solve_run(cfg, EdgeMLP(n_features("dr"), 4),
                                  torch.zeros(2), torch.ones(2), "dr",
                                  verbose=False, on_fenster=gesehen.append)

    assert len(gesehen) == len(zeilen) == lauf["n_fenster"] > 50
    assert tid.dtype == np.int32 and len(tid) == lauf["n_events"]
    # Globale IDs: jede vergebene ID gehoert zu genau EINEM Fenster.
    for z in zeilen:
        assert z["fenster"] >= 0
    zugeordnet = tid[tid >= 0]
    assert lauf["n_tracklets"] == len(np.unique(zugeordnet))
    assert lauf["n_events_unassigned"] == int((tid < 0).sum())
    # Die Volkszaehlung ist da, weil der Run eine GT hat.
    assert all("n_nein" in z for z in zeilen)
    assert sum(z["n_nein"] for z in zeilen) > 0, "Kreuzungs-Run ohne NEIN"


def test_tracklet_ids_sind_global(mini_datensatz):
    """Ein Tracklet darf nie ueber eine Fenstergrenze reichen, und zwei
    Fenster duerfen sich keine ID teilen."""
    from eventpiv.kern.runconfig import LabelCfg, Labeling
    from eventpiv.kette.labels import build_labels
    from eventpiv.kette.model import EdgeMLP, n_features
    from eventpiv.kette.tracklets import solve_run

    build_labels(LabelCfg(io=IO(), labeling=Labeling(),
                          selection=Selection(split="train")), verbose=False)
    nb = Neighborhood(alpha=10_000, r_max=5.0, geom="cylinder", polarity="on")
    cfg = SolveCfg(io=IO(), neighborhood=nb, pairs=PairSampling(),
                   training=Training(), solver=Solver(backend="cc"),
                   selection=Selection(split="train", run_nr=1),
                   window=Window(t0=None, dt=None),
                   compute=Compute(device="cpu", cost_batch=1_000_000))
    torch.manual_seed(0)
    tid, _, _ = solve_run(cfg, EdgeMLP(n_features("dr"), 4), torch.zeros(2),
                          torch.ones(2), "dr", verbose=False)

    f = lade_fenster(cfg.selection, Window(t0=None, dt=None), polarity="on",
                     gt_path=config.label_file("train", 1))
    grenze = np.floor((f.t - f.t[0]) / nb.alpha).astype(np.int64)
    m = tid >= 0
    # Je Tracklet-ID darf es genau EINEN Fensterindex geben.
    paare = set(zip(tid[m].tolist(), grenze[m].tolist()))
    assert len(paare) == len(set(t for t, _ in paare))


def test_tracklet_artefakt_rundreise(mini_datensatz):
    """Die Indizierungszusicherung: ein Eintrag je ON-Event, in der
    Reihenfolge des gefilterten Label-Stroms. Haelt sie nicht, ist die
    Datei stumm unbrauchbar -- nichts wuerde abstuerzen."""
    import json

    from eventpiv.kern.graph import polarity_mask
    from eventpiv.kern.runconfig import LabelCfg, Labeling
    from eventpiv.kette.labels import build_labels
    from eventpiv.kette.model import EdgeMLP, n_features
    from eventpiv.kette.tracklets import schreibe_tracklets, solve_run

    build_labels(LabelCfg(io=IO(), labeling=Labeling(),
                          selection=Selection(split="train")), verbose=False)

    nb = Neighborhood(alpha=10_000, r_max=5.0, geom="cylinder", polarity="on")
    sel, sv = Selection(split="train", run_nr=1), Solver(backend="cc")
    cfg = SolveCfg(io=IO(), neighborhood=nb, pairs=PairSampling(),
                   training=Training(), solver=sv, selection=sel,
                   window=Window(t0=None, dt=None),
                   compute=Compute(device="cpu", cost_batch=1_000_000))

    torch.manual_seed(0)
    tid, zeilen, lauf = solve_run(cfg, EdgeMLP(n_features("dr"), 4),
                                  torch.zeros(2), torch.ones(2), "dr",
                                  verbose=False)
    pfad = schreibe_tracklets(cfg, tid, zeilen, lauf)

    assert pfad == config.tracklet_file(nb, cfg.pairs, cfg.training, sel, sv)
    assert pfad.exists()

    with np.load(pfad) as d:
        np.testing.assert_array_equal(d["tracklet_id"], tid)
        for k in ("fenster_t0", "fenster_n_events", "fenster_n_edges",
                  "fenster_n_nein", "fenster_n_clusters"):
            assert len(d[k]) == len(zeilen)
        meta = json.loads(str(d["meta"]))
        assert meta["stage"] == "tracklets"
        assert meta["neighborhood"]["polarity"] == "on"
        assert meta["n_tracklets"] == lauf["n_tracklets"]

    # Die Zusicherung selbst: Laenge = Zahl der ON-Events der Label-Datei.
    with np.load(config.label_file("train", 1)) as g:
        m = polarity_mask(g["p"], nb.polarity)
    assert len(tid) == int(m.sum())


def test_stumm_gibt_den_deskriptor_zurueck(capfd):
    """Nach dem Block muss stdout wieder funktionieren -- ein verlorener
    Deskriptor machte den Rest des Prozesses stumm, ohne abzustuerzen."""
    from eventpiv.kette.tracklets import _stumm

    with _stumm():
        print("das darf niemand sehen")
    print("das schon")
    assert "das schon" in capfd.readouterr().out