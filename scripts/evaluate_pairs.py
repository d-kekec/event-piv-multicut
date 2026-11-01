#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer eventpiv.kette.train.evaluate_pairs_on_dataset.

Liest configs/train.yaml, weil das Schema TrainCfg ist: bewertet wird ein
Netz, und dessen Identitaet besteht aus neighborhood + pairs + training.
Ein FREMDER Datensatz kommt ueber io.data_path herein.

Aufruf (allgemeine Bedienung: scripts/_cli.py). Die Gruppen waehlen das
Netz; ohne io.data_path wird gegen den eigenen Paar-Datensatz bewertet.
Kein sbatch-Skript.
  python scripts/evaluate_pairs.py io.data_path=<pfad>/pairs.npz
  # einen Paar-Datensatz aus val-Runs baut [2] mit eigener Quelle; den
  # Pfad meldet build_pair_dataset am Ende
  python scripts/build_pair_dataset.py pairs.split=val 'pairs.runs=[<runs>]'
  # batch1
  python scripts/evaluate_pairs.py datensatz=batch1 io.data_path=<pfad>
"""
from dataclasses import asdict

import hydra
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from _cli import konfusionstabelle, to_cfg
from eventpiv.bewerten.metrics import flatten
from eventpiv.kern.runconfig import TrainCfg, slug
from eventpiv.kette.train import evaluate_pairs_on_dataset

ConfigStore.instance().store(name="train_schema", node=TrainCfg)

PROJECT = "event-piv-multicut"


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(dcfg: DictConfig) -> None:
    cfg: TrainCfg = to_cfg(dcfg)

    run = wandb.init(
        project=PROJECT,
        group=slug(cfg.neighborhood),
        job_type="eval_pairs",
        name=f"eval+{slug(cfg.training)}",
        tags=[slug(cfg.pairs), slug(cfg.training)],
        config=asdict(cfg),
    )
    print(f"W&B: {run.url}", flush=True)

    try:
        k = evaluate_pairs_on_dataset(cfg)
        split = k["split"]
        # Praefix aus den Metadaten des Datensatzes: val/auc, test/auc, ...
        run.summary.update(flatten({split: k["metrics"], "data": k["data"]}))
        run.log({split + "/confusion": konfusionstabelle(k["metrics"]["cm"])})
    finally:
        run.finish()


if __name__ == "__main__":
    main()