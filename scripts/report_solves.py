#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer eventpiv.report -- Hydra + Aufruf. Liest nur."""
import json
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from _cli import to_cfg
from eventpiv.report import report
from eventpiv.runconfig import ReportCfg

ConfigStore.instance().store(name="report_schema", node=ReportCfg)


@hydra.main(version_base="1.3", config_path="../configs", config_name="report")
def main(dcfg: DictConfig) -> None:
    cfg: ReportCfg = to_cfg(dcfg)
    z = report(cfg)
    if cfg.io.json_out:
        Path(cfg.io.json_out).write_text(json.dumps(z, indent=1))


if __name__ == "__main__":
    main()