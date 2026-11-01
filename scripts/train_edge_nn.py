#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer eventpiv.kette.train -- Hydra, W&B, Aufruf. Stufe [3].

Die Framework-Grenze: hier sind Hydra UND W&B bekannt, in src/eventpiv/
keines von beiden. train.py liefert verschachtelte Kennzahlen an zwei
Senken; was daraus wird, entscheidet diese Datei.

Aufruf (allgemeine Bedienung: scripts/_cli.py). Die Gruppen muessen die
des Paarbaus [2] sein: sie leiten den Eingabepfad ab.
  python scripts/train_edge_nn.py
  # batch1; datensatz=batch1 setzt neighborhood, pairsampling und training
  python scripts/train_edge_nn.py datensatz=batch1
  # ueber slurm/train.sbatch (Bedarf in dessen Kopf): Overrides in EXTRA;
  # EXTRA_SETS fuehrt mehrere Saetze nacheinander aus, durch ';' getrennt
  sbatch slurm/train.sbatch
  EXTRA="datensatz=batch1" sbatch slurm/train.sbatch
"""
from dataclasses import asdict

import hydra
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from _cli import to_cfg, konfusionstabelle
from eventpiv.bewerten.metrics import flatten
from eventpiv.kern.runconfig import TrainCfg, slug
from eventpiv.kette.train import train_edge_nn

ConfigStore.instance().store(name="train_schema", node=TrainCfg)

PROJECT = "event-piv-multicut"


def senken(run):
    """(on_start, on_epoch) fuer train_edge_nn.

    FEHLERPOLITIK, bewusst unsymmetrisch: wandb.init DARF abbrechen -- dort
    ist noch nichts gerechnet, und ein stiller Lauf ohne Log faellt sonst
    erst hinterher auf. wandb.log darf NICHT abbrechen: dort steckt schon
    Rechenzeit im Lauf, und metrics.json entsteht ohnehin unabhaengig von
    W&B.
    """
    def on_start(daten):
        # Messungen, keine Konfiguration -> summary. config bleibt dadurch
        # exakt asdict(cfg) und zeichengleich mit der Override-Syntax.
        try:
            run.summary.update(flatten({"data": daten}))
        except Exception as e:
            print(f"  WARN: W&B summary fehlgeschlagen: {e}")

    def on_epoch(zeile):
        try:
            run.log(flatten(zeile))
        except Exception as e:
            print(f"  WARN: W&B log fehlgeschlagen: {e}")

    return on_start, on_epoch


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(dcfg: DictConfig) -> None:
    cfg: TrainCfg = to_cfg(dcfg)

    # group = Nachbarschaft (wp/), name = Paar-Datensatz + Netz. group
    # allein reicht nicht: darunter liegen mehrere Paar-Datensaetze.
    # slug() braucht die Dataclass, nicht den DictConfig -- to_cfg liefert sie.
    run = wandb.init(
        project=PROJECT,
        group=slug(cfg.neighborhood),
        job_type="train",
        name=f"{slug(cfg.pairs)}+{slug(cfg.training)}",
        config=asdict(cfg),
    )
    run.define_metric("*", step_metric="epoch")   # x-Achse ist die Epoche
    
    # Explizit nach stdout: wandb meldet sich ueber stderr, und
    # slurm/train.sbatch trennt beide Stroeme. Ohne diese Zeile stuenden
    # Ergebnisse und Lauf-URL in verschiedenen Dateien.
    print(f"W&B: {run.url}", flush=True)

    on_start, on_epoch = senken(run)
    try:
        path, kennzahlen = train_edge_nn(cfg, on_start=on_start,
                                         on_epoch=on_epoch)
        # unter final/, damit die Schlusswerte in der Runtabelle nicht mit
        # den gleichnamigen Verlaufswerten aus holdout/ verwechselt werden.
        run.summary.update(flatten({"final": kennzahlen}))
        run.summary["model_path"] = str(path)
        # Tabellen ueber log, nicht ueber summary: nur der Log-Eintrag traegt
        # den Deskriptor, aus dem die Oberflaeche ein Panel baut. In summary
        # wird die Tabelle zwar geschrieben, aber nie angezeigt.
        # EINMAL am Ende mit der letzten Epoche als Schritt -- eine
        # Momentaufnahme je Lauf, kein Verlauf.
        run.log({"epoch": cfg.training.epochs,
                 "train/confusion":
                     konfusionstabelle(kennzahlen["train"]["cm"]),
                 "holdout/confusion":
                     konfusionstabelle(kennzahlen["holdout"]["cm"])})
    finally:
        run.finish()


if __name__ == "__main__":
    main()