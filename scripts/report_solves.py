#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer eventpiv.report -- Hydra, W&B, Aufruf. Liest nur."""
import json
from dataclasses import asdict
from pathlib import Path

import hydra
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from _cli import to_cfg
from eventpiv.metrics import flatten
from eventpiv.report import report
from eventpiv.runconfig import ReportCfg, slug

ConfigStore.instance().store(name="report_schema", node=ReportCfg)

PROJECT = "event-piv-multicut"


@hydra.main(version_base="1.3", config_path="../configs", config_name="report")
def main(dcfg: DictConfig) -> None:
    cfg: ReportCfg = to_cfg(dcfg)

    # Ein eigener W&B-Lauf, obwohl hier nichts gerechnet wird: DIESE Zeilen
    # sind die Tabelle aus BA_NN_Kantenkosten.md §3.8, die am Ende in der
    # Arbeit steht. Ein Lauf je Netz, mit CROSSING und CLEAN getrennt in der
    # Runtabelle -- das ist die Ansicht, in der Arbeitspunkte verglichen
    # werden.
    run = wandb.init(
        project=PROJECT,
        group=slug(cfg.neighborhood),
        job_type="report",
        name=f"report+{slug(cfg.training)}+{cfg.report.split}",
        tags=[slug(cfg.pairs), slug(cfg.training)],
        config=asdict(cfg),
    )
    print(f"W&B: {run.url}", flush=True)

    try:
        z = report(cfg)
        run.summary.update(flatten(z))
        if cfg.io.json_out:
            Path(cfg.io.json_out).write_text(json.dumps(z, indent=1))
    finally:
        run.finish()


if __name__ == "__main__":
    main()