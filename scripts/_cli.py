#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bruecke Hydra -> Dataclass. EINZIGE Stelle in scripts/, die mit
OmegaConf rechnet; die Skripte nennen DictConfig nur als Typ.

Warum nicht einfach OmegaConf.to_object: Hydra erlaubt mit '+' das Anhaengen
beliebiger Schluessel, und to_object wirft sie danach still weg.
'+alpha=20000' -- flach statt gruppiert -- wuerde damit weder auffallen noch
wirken.

BEDIENUNG, fuer alle Skripte unter scripts/. Jedes Skript nennt in seinem
Abschnitt "Aufruf" nur, was ihm eigen ist.

  Uebersicht Stufe, Kurzname, scripts/<skript>.py, slurm/<datei>.sbatch;
             was die Stufen tun, steht im Glossar in src/eventpiv/__init__.py.
             [1]  Labels         build_event_labels    labels
             [2]  Paare          build_pair_dataset    pairs
             [3]  Training       train_edge_nn         train
             [4]  Fenster-Solve  solve_with_nn         kein
             [5a] Tracklets      build_tracklets       tracklets
             [6a] Kandidaten     build_kandidaten      kandidaten
             [6b] Transformer    train_stufe2          stufe2
             [6c] Stufe-2-Solve  solve_stufe2          solve2
             [7]  Fit            fit_trajektorien      fit
             [A]  Run            animate_run           kein
             [B]  Cluster        animate_clusters      kein
             [C]  Tracklets      animate_tracklets     kein
             [D]  Trajektorien   animate_trajektorien  animate_trajektorien
             Anker               tests/test_anker.py   anker
             -    Paarbewertung  evaluate_pairs        kein
             -    Ueberlappung   identify_overlap_runs kein
  Splits     [1], [4], [5a], [6a], [6c], [7], [A] bis [D] laufen je Split; fuer
             einen anderen nur selection.split setzen. Woraus [3], [6b] lernen,
             legen pairs.split und kandidaten.split fest; sie bleiben train.
  batch1     Auf batch1 braucht jede Stufe ihre Gruppen; welche, nennt der
             Docstring des Skripts.
  Umgebung   EVENTPIV_DATA (Verzeichnis mit train/, val/, test/) und
             EVENTPIV_OUT (Ausgabewurzel) muessen gesetzt sein, sonst
             bricht kern/config.py ab. Je Datensatz ein eigenes Paar:
             Label-Dateien tragen keinen Slug.
  Overrides  gruppe.feld=wert, etwa selection.split=val. Listen ohne
             Leerzeichen und in Anfuehrungszeichen, sonst greift die Shell
             die eckigen Klammern: 'selection.runs=[8,13]'. Wahrheitswerte
             true/false, ein leeres Feld null. Ein Text mit Komma braucht
             innere Anfuehrungszeichen, sonst liest Hydra das Komma als
             Sweep und bricht mit "Ambiguous value" ab:
             "anim.panels='tracklets,gt'".
  Gruppen    gruppe=name schaltet configs/<gruppe>/<name>.yaml zu, etwa
             datensatz=batch1. Ohne Gruppe gelten die Vorgaben aus
             kern/runconfig.py. Welche Gruppen ein Skript kennt, steht in
             seiner Treiberdatei configs/<stufe>.yaml.
  +schluessel=wert  weist to_cfg ab.
  --cfg job --resolve  druckt die zusammengesetzte Konfiguration und
             rechnet nichts; main() laeuft dabei nicht, to_cfg prueft also
             auch nichts.
  Pfade      io.out_path und io.json_out werden nicht um ~ erweitert.
  W&B        WANDB_MODE=disabled vor dem Aufruf schaltet W&B ganz ab,
             WANDB_MODE=offline schreibt nur lokal; fuer Laeufe ohne
             W&B-Konto eines von beiden setzen. Ohne W&B laufen ohnehin
             build_event_labels, identify_overlap_runs und animate_*.
  Protokoll  Jede Treiberdatei setzt hydra.run.dir: Hydra legt sein
             Protokoll mit der zusammengesetzten Konfiguration unter
             $EVENTPIV_OUT/hydra/<name>/<Zeitstempel>/ ab, mit --multirun
             unter multirun_<Zeitstempel>/<Jobnummer>/. <name> ist der
             Name der Treiberdatei, bei den Animationen verkuerzt
             (animrun, animclusters auch fuer animate_tracklets,
             animtraj). job.chdir ist aus: relative Pfade gelten ab dem
             Aufrufverzeichnis.
  slurm/     Allgemeines zu den sbatch-Skripten. Variablen direkt vor
             sbatch setzen, nicht exportieren: ein exportiertes EXTRA
             griffe in jedem spaeteren Job derselben Shell. Nicht ueber
             --export: Slurm trennt dort an Kommas, und 'runs=[2,6]' kaeme
             zerteilt an. sbatch reicht die Umgebung der absendenden Shell
             weiter, also auch EVENTPIV_DATA und EVENTPIV_OUT.
             EXTRA geht ungequotet ans Skript und zerfaellt in einzelne
             Overrides. EXTRA_SETS hat Vorrang vor EXTRA; je Satz startet
             ein eigener Prozess, und ein fehlgeschlagener Satz bricht den
             Job nicht ab.
             Optionen an sbatch ueberschreiben die #SBATCH-Zeilen, etwa
             --time oder --gres. Partition und Zeiten in den #SBATCH-Zeilen
             sowie die Pfade zu venv und Repo sind an den eigenen Cluster
             anzupassen. logs/ muss im Absendeverzeichnis existieren: Slurm
             legt es nicht an, und ohne das Verzeichnis scheitert der Job
             ohne Log. Ohne GPU-Bedarf kein --gres: ein Job ohne
             GPU-Anforderung wartet nicht auf eine freie Karte.
             Mehrere Laeufe ueber die sbatch-Skripte, nicht ueber Hydras
             --multirun: das faehrt alle im selben Prozess.
             Steht ein Job auf einem Knoten mit vollem Swap bei 100 % CPU
             ohne Fortschritt, NUMPY_MADVISE_HUGEPAGE=0 vor sbatch setzen.
"""
from dataclasses import asdict
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf


def _flat(d: dict, prefix: str = "") -> set[str]:
    out: set[str] = set()
    for k, v in d.items():
        key = f"{prefix}{k}"
        out |= _flat(v, key + ".") if isinstance(v, dict) else {key}
    return out


def to_cfg(cfg: DictConfig) -> Any:
    """DictConfig -> Dataclass, mit hartem Abbruch bei unbekannten Schluesseln.

    Rueckgabetyp Any, keine Nachlaessigkeit: welche Dataclass herauskommt,
    entscheidet das im ConfigStore registrierte Schema, also der Aufrufer --
    statisch nicht bestimmbar. Die Zusicherung steht deshalb an der
    Aufrufstelle ("cfg: TrainCfg = to_cfg(...)").
    """
    obj: Any = OmegaConf.to_object(cfg)
    # cast, weil to_container statisch eine Union liefert (ein Config
    # koennte auch Liste oder Skalar sein). Dass hier immer ein DictConfig
    # ankommt, sichert das Framework zu, nicht das Typsystem.
    roh = cast(dict, OmegaConf.to_container(cfg, resolve=True))
    fremd = _flat(roh) - _flat(asdict(obj))
    if fremd:
        raise SystemExit(
            f"Unbekannte Konfigurationsschluessel: {sorted(fremd)}\n"
            "Mit '+' angehaengt? Parameter gehoeren in ihre Gruppe "
            "(neighborhood.alpha=..., nicht alpha=...).")
    return obj


def konfusionstabelle(cm: dict):
    """Die 2x2-Matrix als W&B-Tabelle.

    Aus den VIER ZAHLEN gebaut, nicht ueber wandb.plot.confusion_matrix --
    das verlangt die Vorhersage je Datenpunkt und waere bei Millionen Paaren
    absurd.

    Geloggt wird sie ueber run.log, nicht ueber run.summary: nur der
    Log-Eintrag traegt den Deskriptor (_type, ncols, nrows, artifact_path),
    aus dem die Oberflaeche ein Panel baut. Gilt fuer wandb 0.28.1.
    """
    import wandb
    t = wandb.Table(columns=["", "Predicted JA", "Predicted NEIN"])
    t.add_data("GT JA", cm["tp"], cm["fn"])
    t.add_data("GT NEIN", cm["fp"], cm["tn"])
    return t