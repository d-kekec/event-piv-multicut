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

Zwei Fenster, weil eines nicht reicht: run61/3820 ms verschmilzt GROB
(ein Cluster mit rund dreieinhalb Partikeln), run32/7070 ms verschmilzt
FEIN und oft. Ein Anker auf nur einem von beiden sieht die andere
Fehlerart nicht.
"""
import pytest

from eventpiv.kern import config
from eventpiv.kern.runconfig import (Neighborhood, PairSampling, Selection,
                                SolveCfg, Solver, Training, Window)

pytestmark = pytest.mark.anker

# (run, t0, n_edges, purity_cluster, purity_event, frag_mean)
ANKER = [
    (61, 3_820_000, 3_346_692, 0.995, 0.979, 2.84),
    (32, 7_070_000, 1_182_616, 0.993, 0.995, 3.51),
]


@pytest.mark.parametrize("run_nr,t0,n_edges,p_cluster,p_event,frag", ANKER)
def test_anker(run_nr, t0, n_edges, p_cluster, p_event, frag):
    from eventpiv.solve import solve_and_evaluate

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
    # Die Kennzahlen auf drei Nachkommastellen -- so werden sie gedruckt,
    # und feiner ist die Aussage nicht.
    assert info["purity_cluster_mean"] == pytest.approx(p_cluster, abs=5e-4)
    assert info["purity_event_mean"] == pytest.approx(p_event, abs=5e-4)
    assert info["frag_mean"] == pytest.approx(frag, abs=5e-3)


def test_arbeitspunkt_ist_der_erwartete():
    """Faengt den Fall, dass eine Vorgabe wandert und der Anker deshalb
    etwas anderes misst, als sein Name behauptet."""
    nb = Neighborhood()
    assert (nb.alpha, nb.r_max, nb.geom, nb.polarity) == \
        (10_000, 5.0, "cylinder", "on")
    assert Solver().mode == "P" and Solver().cost_offset == 0.0
    assert PairSampling().split == "train" and Training().feature_mode == "dr"