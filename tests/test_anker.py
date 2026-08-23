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
Kanten auf 49 Partikeln und verschmilzt messbar (0,45 % der Events);
run32/7070 ms traegt ein Drittel davon und verschmilzt praktisch nicht.
Das eine ist die Messung, das andere die Kontrolle -- steigt dort etwas
ueber null, ist etwas kaputt.

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

pytestmark = pytest.mark.anker

# (run, t0, n_edges, frac_neg, purity_event, frag_mean, merge_event_frac)
ANKER = [
    (61, 3_820_000, 3_346_692, 0.3952, 0.9955, 5.571, 0.004494),
    (32, 7_070_000, 1_182_616, 0.3857, 1.0000, 6.270, 0.000081),
]


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