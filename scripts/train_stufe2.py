#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer eventpiv.kette.train2 -- Hydra, W&B, Aufruf. Stufe [6b].

Die Framework-Grenze wie bei train_edge_nn.py: hier sind Hydra und W&B
bekannt, in src/eventpiv/ keines von beiden.

Aufruf (allgemeine Bedienung: scripts/_cli.py). Kein selection: gelesen
werden die Artefakte [6a] aller Runs aus kandidaten.runs im Split
kandidaten.split, gebaut mit denselben Gruppen. Ein vorhandenes Netz
bricht ab, io.overwrite=true trainiert neu. io.train_fest nennt Runs, die
nicht in den Holdout duerfen. Auf batch1 ohne transformersatz=batch1
trainiert es still ein Netz mit den Vorgaben.
  python scripts/train_stufe2.py
  # Probelauf mit wenigen Epochen, eigener Slug
  python scripts/train_stufe2.py transformer.epochs=5
  # batch1
  E="datensatz=batch1 kandidatensatz=batch1 transformersatz=batch1"
  python scripts/train_stufe2.py $E
  # ueber slurm/stufe2.sbatch (Bedarf in dessen Kopf): nichts Pflicht;
  # OFFSET setzt solver.cost_offset, also die Quelle aus Stufe 1, KD_RUNS
  # kandidaten.runs, NOT_HOLDOUT io.train_fest; Overrides in EXTRA,
  # EXTRA_SETS fuehrt mehrere Saetze nacheinander aus, durch ';' getrennt
  sbatch slurm/stufe2.sbatch
  EXTRA="$E" sbatch slurm/stufe2.sbatch
"""
from dataclasses import asdict

import hydra
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from _cli import to_cfg, konfusionstabelle
from eventpiv.bewerten.metrics import flatten
from eventpiv.kern.runconfig import Stufe2Cfg, slug
from eventpiv.kette.train2 import train_stufe2

ConfigStore.instance().store(name="stufe2_schema", node=Stufe2Cfg)

PROJECT = "event-piv-multicut"


def senken(run):
    """(on_start, on_epoch). Fehlerpolitik wie in train_edge_nn.py:
    wandb.init darf abbrechen, wandb.log nicht -- dort steckt schon
    Rechenzeit im Lauf."""
    def on_start(daten):
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


@hydra.main(version_base="1.3", config_path="../configs",
            config_name="stufe2")
def main(dcfg: DictConfig) -> None:
    cfg: Stufe2Cfg = to_cfg(dcfg)
    run = wandb.init(
        project=PROJECT, group=slug(cfg.neighborhood), job_type="stufe2",
        name=f"{slug(cfg.kandidaten)}+{slug(cfg.transformer)}"
             f"_{cfg.solver.short()}",
        tags=[slug(cfg.pairs), slug(cfg.training)], config=asdict(cfg))
    run.define_metric("*", step_metric="epoch")
    print(f"W&B: {run.url}", flush=True)

    on_start, on_epoch = senken(run)
    try:
        pfad, kennzahlen = train_stufe2(cfg, on_start=on_start,
                                        on_epoch=on_epoch)
        run.summary.update(flatten({"final": kennzahlen}))
        run.summary["model_path"] = str(pfad)
        run.log({"epoch": cfg.transformer.epochs,
                 "train/confusion": konfusionstabelle(kennzahlen["train"]),
                 "holdout/confusion":
                     konfusionstabelle(kennzahlen["holdout"])})
        print(f"-> {pfad}", flush=True)
    finally:
        run.finish()


if __name__ == "__main__":
    main()