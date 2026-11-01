#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer eventpiv.kette.labels -- Hydra + Aufruf. Stufe [1].

Aufruf (allgemeine Bedienung: scripts/_cli.py). Ohne selection.runs alle
Runs des Splits. Vorhandene Dateien mit passendem Labeling bleiben stehen,
abweichende brechen ab; io.overwrite=true baut sie neu. Auf batch1 ohne
labeling=batch1 entstehen still Labels mit den STELLA-Vorgaben.
  python scripts/build_event_labels.py
  python scripts/build_event_labels.py selection.split=val
  python scripts/build_event_labels.py 'selection.runs=[2,6,8]'
  # batch1
  python scripts/build_event_labels.py labeling=batch1
  # ueber slurm/labels.sbatch (Bedarf in dessen Kopf): SPLIT Pflicht, ohne
  # RUNS alle Runs des Splits, PRO_TASK Runs je Array-Task, also --array=0
  # bis ceil(Runs/PRO_TASK)-1; Overrides in EXTRA, fuer batch1
  # EXTRA="labeling=batch1"
  SPLIT=train RUNS="2 6 8" PRO_TASK=3 sbatch --array=0 slurm/labels.sbatch
"""
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from _cli import to_cfg
from eventpiv.kern.runconfig import LabelCfg
from eventpiv.kette.labels import build_labels

ConfigStore.instance().store(name="labels_schema", node=LabelCfg)


@hydra.main(version_base="1.3", config_path="../configs", config_name="labels")
def main(dcfg: DictConfig) -> None:
    cfg: LabelCfg = to_cfg(dcfg)
    failed = build_labels(cfg)
    if failed:
        raise SystemExit(f"{len(failed)} Run(s) fehlgeschlagen: {failed}")


if __name__ == "__main__":
    main()
