#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Der Regressionsanker: zwei Fenster auf echten Daten, feste Zahlen.

WOZU, und warum das kein gewoehnlicher Test ist: die Suite laeuft auf einem
synthetischen Mini-Datensatz und prueft die VERDRAHTUNG. Welche Zahl
richtig ist, kann sie nicht wissen. Der Anker kann es -- er haelt fest, was
eine bekannte Konfiguration auf bekannten Daten liefert.

Er braucht deshalb den echten Datensatz, gebaute Label-Dateien und ein
trainiertes Netz und ist mit `anker` markiert, also standardmaessig
abgewaehlt:

    pytest -m anker

DIE ZAHLEN SIND KEINE ZIELVORGABE. Sie sind der Stand, gegen den eine
Aenderung gemessen wird. Aendert sich eine absichtlich, wird sie hier
angepasst -- im selben Commit wie die Aenderung, mit der Begruendung in der
Commit-Nachricht.

Zwei Fenster, weil eines nicht reicht: run61/3820 ms traegt 3,35 Mio.
Kanten auf 49 Partikeln, run32/7070 ms ein Drittel davon. Beide
verschmelzen um ein halbes Prozent der Events.

STUFE 2 hat einen EIGENEN Anker unter eigenem Marker, auf einem GANZEN
Run statt auf Fenstern -- sie loest den Run am Stueck. Er kostet rund
drei Minuten gegen zehn Sekunden fuer Stufe 1, deshalb die Trennung:
`pytest -m anker` bleibt der Griff fuer jeden Commit.

Er haengt am trainierten Netz und ist bei jedem Neutraining anzupassen;
solange der Arbeitspunkt steht, ist das kein Nachteil, sondern der Zweck.

Die Zahlen gelten fuer DIESE ZWEI FENSTER und sind keine Aussage ueber
einen Run -- ein Netz kann hier schlechter und ueber 800 Fenster besser
sein. Der Anker faengt Regressionen, er bewertet nicht.

purity_cluster_mean steht NICHT mehr in den Zahlen: der Anteil sauberer
Cluster geht mit wachsender Fragmentierung zwangslaeufig gegen 1 und ist
am heutigen Arbeitspunkt (frag_mean rund 6) nicht mehr aussagekraeftig.
An seiner Stelle stehen frac_neg -- die Groesse, an der ein verlorenes
mu/sig oder ein falscher feature_mode sofort auffaellt -- und
merge_event_frac, die Zielgroesse von Stufe 1.
"""
import pytest

from eventpiv.kern import config
from eventpiv.kern.runconfig import (Neighborhood, PairSampling, Selection,
                                SolveCfg, Solver, Training, Window)

# KEIN modulweiter Marker: Stufe 1 und Stufe 2 sind getrennt waehlbar.
#
#   pytest -m anker              Stufe 1, rund 10 s
#   pytest -m anker2             Stufe 2, rund 3 min (ein ganzer Run)
#   pytest -m "anker or anker2"  beides

# (run, t0, n_edges, frac_neg, purity_event, frag_mean, merge_event_frac)
ANKER = [
    (61, 3_820_000, 3_346_692, 0.4118, 0.9942, 5.469, 0.006787),
    (32, 7_070_000, 1_182_616, 0.3966, 0.9989, 6.459, 0.005050),
]


@pytest.mark.anker
@pytest.mark.parametrize("run_nr,t0,n_edges,frac_neg,p_event,frag,merge",
                         ANKER)
def test_anker(run_nr, t0, n_edges, frac_neg, p_event, frag, merge):
    from eventpiv.kette.solve import solve_and_evaluate

    cfg = SolveCfg(selection=Selection(split="train", run_nr=run_nr),
                   window=Window(t0=t0, dt=10_000))
    mpath = config.model_file(cfg.neighborhood, cfg.pairs, cfg.training)
    if not mpath.exists():
        pytest.skip(f"{mpath} fehlt -- erst [2] und [3] bauen.")

    info = solve_and_evaluate(cfg, verbose=False)

    # Die Kantenzahl EXAKT: sie folgt allein aus der Geometrie und dem
    # Polaritaetsfilter. Weicht sie ab, hat sich die Nachbarschaft
    # geaendert, und alles Weitere ist ohnehin nicht mehr vergleichbar.
    assert info["n_edges"] == n_edges
    # frac_neg ZUERST: sie haengt allein am Netz und am Versatz, nicht am
    # Solver. Weicht sie ab, ist die Ursache dort und nicht in der
    # Partition -- und alles Weitere waere Folgefehler.
    assert info["frac_neg"] == pytest.approx(frac_neg, abs=5e-4)
    assert info["purity_event_mean"] == pytest.approx(p_event, abs=5e-4)
    assert info["frag_mean"] == pytest.approx(frag, abs=5e-3)
    # Enger als die uebrigen: auf run32 sind es 8e-5, eine Toleranz von
    # 5e-4 liesse dort jeden Wert durch.
    assert info["merge_event_frac"] == pytest.approx(merge, abs=5e-5)


@pytest.mark.anker
def test_arbeitspunkt_ist_der_erwartete():
    """Faengt den Fall, dass eine Vorgabe wandert und der Anker deshalb
    etwas anderes misst, als sein Name behauptet. Der Anker faehrt seit
    dem 23.08. reine Vorgaben -- keine ausgeschriebene Gruppe mehr.
    """
    nb = Neighborhood()
    assert (nb.alpha, nb.r_max, nb.geom, nb.polarity) == \
        (10_000, 5.0, "cylinder", "on")
    assert Solver().mode == "P" and Solver().cost_offset == 0.0
    assert PairSampling().split == "train" and len(PairSampling().runs) == 30
    assert Training().feature_mode == "drjac" and Training().sample_ratio == 1.0
    assert (Training().hidden, Training().lr) == (32, 3e-2)


# (run, n_knoten, n_kanten, frac_neg, n_cluster, merge_event_frac, hota)
#
# EIN Run genuegt: er traegt 15,4 Mio. Kanten auf 90 Partikeln, und ein
# zweiter kostete drei Minuten fuer dieselbe Aussage. run61 hat lange
# Ueberlappungen, IDSW bleibt dort > 0 -- gerade deshalb reagiert er auf
# jede Verschiebung des Betriebspunkts. Und er steht in kandidaten.runs,
# seine Kandidaten fallen also bei jedem Arbeitspunkt ohnehin an.
ANKER2 = [
    (61, 152082, 15404004, 0.053322, 103, 0.0079262, 0.97203),
]


@pytest.mark.anker2
@pytest.mark.parametrize("run_nr,n_knoten,n_kanten,frac_neg,n_cluster,"
                         "merge,hota", ANKER2)
def test_anker_stufe2(run_nr, n_knoten, n_kanten, frac_neg, n_cluster,
                      merge, hota):
    """Stufe 2 auf einem ganzen Run, gegen feste Zahlen.

    Anders als bei Stufe 1 wird kein Fenster geschnitten -- Stufe 2 loest
    den Run am Stueck, und ein Ausschnitt waere ein anderes Problem.
    Laufzeit rund zwei Minuten je Run.
    """
    from eventpiv.kern.runconfig import Solve2Cfg
    from eventpiv.kette.solve2 import solve2_run
    from eventpiv.kette.transformer import load_checkpoint

    cfg = Solve2Cfg(selection=Selection(split="train", run_nr=run_nr))
    mpath = config.transformer_file(cfg.neighborhood, cfg.pairs,
                                    cfg.training, cfg.solver,
                                    cfg.kandidaten, cfg.transformer)
    if not mpath.exists():
        pytest.skip(f"{mpath} fehlt -- erst [6a] und [6b] bauen.")
    kpfad = config.kandidaten_file(cfg.neighborhood, cfg.pairs, cfg.training,
                                   cfg.selection, cfg.solver, cfg.kandidaten)
    if not kpfad.exists():
        pytest.skip(f"{kpfad} fehlt -- erst build_kandidaten.py fuer "
                    f"run{run_nr}.")

    model, norm, _ = load_checkpoint(mpath, device=cfg.compute.device)
    _, lauf = solve2_run(cfg, model, norm, run_nr, verbose=False)

    # Knoten und Kanten EXAKT: beide folgen allein aus min_events und der
    # Geometrie. Weichen sie ab, ist der Kandidatensatz ein anderer, und
    # alles Weitere waere nicht mehr vergleichbar.
    assert lauf["n_knoten"] == n_knoten
    assert lauf["n_kanten"] == n_kanten
    # frac_neg ZUERST unter den weichen Zahlen: sie haengt am Netz und am
    # cost_offset, nicht am Solver. Weicht sie ab, ist die Ursache dort.
    assert lauf["frac_neg"] == pytest.approx(frac_neg, abs=5e-4)
    assert lauf["n_cluster"] == n_cluster
    assert lauf["stufe2"]["merge_event_frac"] == pytest.approx(merge, abs=5e-5)
    assert lauf["stufe2"]["hota"] == pytest.approx(hota, abs=5e-4)


@pytest.mark.anker2
def test_stufe2_arbeitspunkt_ist_der_erwartete():
    """Gegenstueck fuer Stufe 2. Braucht keinen Datensatz -- geprueft wird
    die Vorgabe, nicht ihr Ergebnis."""
    from eventpiv.kern.runconfig import Kandidaten, Solver2, Transformer

    kd = Kandidaten()
    assert (kd.alpha, kd.r_max, kd.geom) == (250_000, 15.0, "hyperboloid")
    assert (kd.min_events, kd.n_events) == (15, 32)
    assert kd.split == "train" and len(kd.runs) == 30
    tf = Transformer()
    assert tf.label_modus == "mehrheit" and tf.gt_min_anteil == 0.9
    assert (tf.epochs, tf.train_max) == (50, 4_000_000)
    assert tf.imbalance == "sample" and tf.sample_ratio == 3.0
    assert (tf.d_model, tf.n_layers, tf.pooling) == (64, 2, "cls")
    assert Solver2().mode == "P" and Solver2().cost_offset == 1.0