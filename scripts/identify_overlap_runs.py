#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer eventpiv.bewerten.overlap -- Hydra + Aufruf."""
import json
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from _cli import to_cfg
from eventpiv.bewerten.overlap import analyze_split
from eventpiv.kern import config
from eventpiv.kern.runconfig import OverlapCfg, identity_meta

ConfigStore.instance().store(name="overlap_schema", node=OverlapCfg)


@hydra.main(version_base="1.3", config_path="../configs", config_name="overlap")
def main(dcfg: DictConfig) -> None:
    cfg: OverlapCfg = to_cfg(dcfg)
    results, cats = analyze_split(cfg)
    if cfg.io.json_out:
        Path(cfg.io.json_out).write_text(json.dumps(
            {"meta": json.loads(config.meta_json(
                stage="overlap", split=cfg.selection.split,
                **identity_meta(overlap=cfg.overlap))),
             "metrics": {str(k): v for k, v in results.items()},
             "categories": cats}, indent=2))
        print(f"-> {cfg.io.json_out}")


if __name__ == "__main__":
    main()