#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Die Regressionsanker: bekannter Arbeitspunkt, echte Daten, feste Zahlen.

WOZU, und warum das kein gewoehnlicher Test ist: die Suite laeuft auf einem
synthetischen Mini-Datensatz und prueft die VERDRAHTUNG. Welche Zahl
richtig ist, kann sie nicht wissen. Der Anker kann es -- er haelt fest, was
ein Arbeitspunkt auf bekannten Daten liefert.

JE DATENSATZ: es laeuft der Anker des Datensatzes, den `eventpiv_use`
gewaehlt hat (EVENTPIV_NAME); ohne Eintrag wird uebersprungen. Die
Konfiguration entsteht aus den Gruppendateien unter configs/, gelesen wie
Hydra sie setzt -- die Werte der Datei ueber den Vorgaben der Dataclass.
Hydra selbst wird nicht gebraucht.

Die Anker brauchen den Datensatz, gebaute Label-Dateien und die
trainierten Netze und sind deshalb markiert und standardmaessig
abgewaehlt. Kein modulweiter Marker: die Stufen sind getrennt waehlbar,
und der Vorgabentest laeuft in jeder Suite.

    pytest -m anker      Stufe 1, je Eintrag ein Fenster, dazu die Slugs
    pytest -m anker2     Stufe 2, ein ganzer Run

Traegt rema einen Datensatz nicht, laufen die Anker als Job:
`sbatch slurm/anker.sbatch`, MARKER waehlt die Stufe.

DIE ZAHLEN SIND KEINE ZIELVORGABE. Sie sind der Stand, gegen den eine
Aenderung gemessen wird. Aendert sich eine absichtlich, wird sie hier
angepasst -- im selben Commit wie die Aenderung, mit der Begruendung in der
Commit-Nachricht. Fehlen sie, misst der Anker und nennt die Werte im
Skip-Grund (`pytest -rs`).

Ein Fenster ist genau ein Stufe-1-Fenster lang (neighborhood.alpha).
Stufe 2 loest einen ganzen Run am Stueck; ein Ausschnitt waere ein anderes
Problem. Beide haengen an den trainierten Netzen und sind bei jedem
Neutraining anzupassen. Die Zahlen gelten fuer diese Fenster und Runs,
nicht fuer einen Datensatz -- der Anker faengt Regressionen, er bewertet
nicht.

purity_cluster_mean steht nicht darin: der Anteil sauberer Cluster geht
mit wachsender Fragmentierung zwangslaeufig gegen 1. An seiner Stelle
stehen frac_neg -- an ihr faellt ein verlorenes mu/sig oder ein falscher
feature_mode sofort auf -- und merge_event_frac, die Zielgroesse von
Stufe 1.
"""
import os
from pathlib import Path

import pytest
import yaml

from eventpiv.kern import config
from eventpiv.kern.runconfig import (Fit, Kandidaten, Labeling, Neighborhood,
                                PairSampling, Selection, SolveCfg, Solver,
                                Solver2, Training, Transformer, Window,
                                digest, slug)

# --- Arbeitspunkte ---------------------------------------------------

SATZ = os.environ.get("EVENTPIV_NAME")
CONFIGS = Path(__file__).resolve().parents[1] / "configs"

# Welcher Arbeitspunkt in den Vorgaben von runconfig.py steht.
VORGABE = "batch1"

# Feld der Stufen-Konfiguration -> (Gruppenverzeichnis, Dataclass)
GRUPPEN = {
    "neighborhood": ("neighborhood", Neighborhood),
    "pairs": ("pairsampling", PairSampling),
    "training": ("training", Training),
    "labeling": ("labeling", Labeling),
    "kandidaten": ("kandidatensatz", Kandidaten),
    "transformer": ("transformersatz", Transformer),
    "fit": ("fitsatz", Fit),
}

# Feste Kennungen je Arbeitspunkt: die Slugs, unter denen seine Artefakte
# liegen, fuer labeling der Hash. Aendert sich eine, zeigt der Arbeitspunkt
# auf andere Artefakte.
SLUGS = {
    "batch1": {
        "neighborhood": "a10000_rmax5_cylinder_on-d0a970",
        "pairs": "split_train_rn30-82455f",
        "training": "drjac_sample-a2591a",
        "labeling": "8579fe",
        "kandidaten": "split_train_rn30_a250000_rmax15_n32-a9e3a6",
        "transformer": "d64_l2_cls_sample_mehrheit-b6d239",
        "fit": "q20_r8_rts-ab4b11",
    },
    "stella": {
        "neighborhood": "a500_rmax5.1_hyperboloid_on-cf244e",
        "pairs": "split_train_rn30-89b883",
        "training": "dr_sample-7e042d",
        "labeling": "2845f5",
        "kandidaten": "split_train_rn30_a7500_rmax8_n32-010e37",
        "transformer": "d64_l2_cls_sample_mehrheit-e71ec8",
        "fit": "q1e+07_r3_rts-e3eb78",
    },
}
# Zweiter Arbeitspunkt auf STELLA: nur Kandidaten und Netz weichen ab.
SLUGS_STELLA_A2_15MS = {
    "kandidaten": "split_train_rn30_a15000_rmax8_n32-ecee29",
    "transformer": "d64_l2_cls_sample_mehrheit-b6d239",
}
# Solver beider Stufen, auf allen Arbeitspunkten gleich.
SOLVER = ("P", "ramaP_o1")


def arbeitspunkt(name, **andere):
    """Die Gruppen eines Arbeitspunkts, wie Hydra sie setzt: die Werte der
    Gruppendatei ueber den Vorgaben der Dataclass. `andere` waehlt fuer
    einzelne Felder eine andere Datei, etwa kandidaten="stella_a2_15ms".
    Ein unbekanntes Feld in einer Datei bricht im Konstruktor ab."""
    out = {}
    for feld, (ordner, klasse) in GRUPPEN.items():
        datei = CONFIGS / ordner / f"{andere.get(feld, name)}.yaml"
        werte = yaml.safe_load(datei.read_text(encoding="utf-8")) or {}
        out[feld] = klasse(**werte)
    return out


def _kennung(feld, gruppe):
    # Labels tragen keinen Slug, ihre Identitaet ist der Hash allein.
    return digest(gruppe) if feld == "labeling" else slug(gruppe)


def _faelle(tabelle, n):
    """Die Eintraege des gewaehlten Datensatzes. Ohne Eintrag einer, der
    mit Grund uebersprungen wird."""
    if SATZ in tabelle:
        return tabelle[SATZ]
    return [pytest.param(*[None] * n, marks=pytest.mark.skip(
        reason=f"kein Anker fuer EVENTPIV_NAME={SATZ!r}"))]


# --- Stufe 1 ---------------------------------------------------------

# (run, t0, soll), soll = (n_edges, frac_neg, purity_event, frag_mean,
# merge_event_frac). None: noch nicht gemessen.
ANKER = {
    "batch1": [
        (61, 3_820_000, (3_346_692, 0.4118, 0.9942, 5.469, 0.006787)),
        (32, 7_070_000, (1_182_616, 0.3966, 0.9989, 6.459, 0.005050)),
    ],
    "stella": [
        (13, 1_000_000, (2_514_947, 0.3117, 0.9995, 1.882, 0.003321)),
        (66, 1_000_000, (492_018, 0.0968, 1.0000, 1.038, 0.000000)),
    ],
}


@pytest.mark.anker
@pytest.mark.parametrize("run_nr,t0,soll", _faelle(ANKER, 3))
def test_anker(run_nr, t0, soll):
    from eventpiv.kette.solve import solve_and_evaluate

    ap = arbeitspunkt(SATZ)
    cfg = SolveCfg(neighborhood=ap["neighborhood"], pairs=ap["pairs"],
                   training=ap["training"],
                   selection=Selection(split="train", run_nr=run_nr),
                   window=Window(t0=t0, dt=ap["neighborhood"].alpha))
    mpath = config.model_file(cfg.neighborhood, cfg.pairs, cfg.training)
    if not mpath.exists():
        pytest.skip(f"{mpath} fehlt -- erst [2] und [3] bauen.")

    info = solve_and_evaluate(cfg, verbose=False)
    if soll is None:
        pytest.skip("noch keine Sollwerte, gemessen: "
                    f"{info['n_edges']}, {info['frac_neg']:.4f}, "
                    f"{info['purity_event_mean']:.4f}, "
                    f"{info['frag_mean']:.3f}, "
                    f"{info['merge_event_frac']:.6f}")
    n_edges, frac_neg, p_event, frag, merge = soll

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
    # Enger als die uebrigen: die Verschmelzung liegt bei Bruchteilen
    # eines Prozents, 5e-4 liesse fast jede Aenderung durch.
    assert info["merge_event_frac"] == pytest.approx(merge, abs=5e-5)


def test_vorgaben_sind_ein_arbeitspunkt():
    """Die Vorgaben in runconfig.py sind genau EIN Arbeitspunkt. Faengt
    eine Vorgabe, die allein wandert -- der Slug zeigte dann auf andere
    Artefakte. Ohne Datensatz und ohne Marker, laeuft in jeder Suite."""
    ist = {feld: _kennung(feld, klasse())
           for feld, (_, klasse) in GRUPPEN.items()}
    assert ist == SLUGS[VORGABE]
    assert (Solver().short(), Solver2().short()) == SOLVER


@pytest.mark.anker
def test_gruppen_ergeben_den_arbeitspunkt():
    """Die Gruppendateien des gewaehlten Datensatzes ergeben dessen
    Arbeitspunkt: dieselben Slugs, unter denen seine Artefakte liegen."""
    if SATZ not in SLUGS:
        pytest.skip(f"kein Arbeitspunkt fuer EVENTPIV_NAME={SATZ!r}")
    ist = {feld: _kennung(feld, g)
           for feld, g in arbeitspunkt(SATZ).items()}
    assert ist == SLUGS[SATZ]
    if SATZ == "stella":
        b = arbeitspunkt("stella", kandidaten="stella_a2_15ms",
                         transformer="stella_a2_15ms")
        assert {f: slug(b[f]) for f in SLUGS_STELLA_A2_15MS} == \
            SLUGS_STELLA_A2_15MS


# --- Stufe 2 ---------------------------------------------------------

# (run, soll), soll = (n_knoten, n_kanten, frac_neg, n_cluster,
# merge_event_frac, hota). None: noch nicht gemessen.
#
# EIN Run je Datensatz genuegt: er loest am Stueck und reagiert auf jede
# Verschiebung des Betriebspunkts. Er steht in kandidaten.runs, seine
# Kandidaten fallen also bei jedem Arbeitspunkt ohnehin an.
ANKER2 = {
    "batch1": [
        (61, (152082, 15404004, 0.053322, 103, 0.0079262, 0.97203)),
    ],
    "stella": [
        (66, (455588, 7113668, 0.047661, 1820, 0.0047655, 0.99102)),
    ],
}


@pytest.mark.anker2
@pytest.mark.parametrize("run_nr,soll", _faelle(ANKER2, 2))
def test_anker_stufe2(run_nr, soll):
    """Stufe 2 auf einem ganzen Run, gegen feste Zahlen."""
    from eventpiv.kern.runconfig import Solve2Cfg
    from eventpiv.kette.solve2 import solve2_run
    from eventpiv.kette.transformer import load_checkpoint

    ap = arbeitspunkt(SATZ)
    cfg = Solve2Cfg(neighborhood=ap["neighborhood"], pairs=ap["pairs"],
                    training=ap["training"], kandidaten=ap["kandidaten"],
                    transformer=ap["transformer"],
                    selection=Selection(split="train", run_nr=run_nr))
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
    s2 = lauf["stufe2"]
    if soll is None:
        pytest.skip("noch keine Sollwerte, gemessen: "
                    f"{lauf['n_knoten']}, {lauf['n_kanten']}, "
                    f"{lauf['frac_neg']:.6f}, {lauf['n_cluster']}, "
                    f"{s2['merge_event_frac']:.7f}, {s2['hota']:.5f}")
    n_knoten, n_kanten, frac_neg, n_cluster, merge, hota = soll

    # Knoten und Kanten EXAKT: beide folgen allein aus min_events und der
    # Geometrie. Weichen sie ab, ist der Kandidatensatz ein anderer, und
    # alles Weitere waere nicht mehr vergleichbar.
    assert lauf["n_knoten"] == n_knoten
    assert lauf["n_kanten"] == n_kanten
    # frac_neg ZUERST unter den weichen Zahlen: sie haengt am Netz und am
    # cost_offset, nicht am Solver. Weicht sie ab, ist die Ursache dort.
    assert lauf["frac_neg"] == pytest.approx(frac_neg, abs=5e-4)
    assert lauf["n_cluster"] == n_cluster
    assert s2["merge_event_frac"] == pytest.approx(merge, abs=5e-5)
    assert s2["hota"] == pytest.approx(hota, abs=5e-4)