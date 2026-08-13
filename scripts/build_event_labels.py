#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer eventpiv.labels -- Hydra + Aufruf."""
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from _cli import to_cfg
from eventpiv.labels import build_labels
from eventpiv.kern.runconfig import LabelCfg

ConfigStore.instance().store(name="labels_schema", node=LabelCfg)


@hydra.main(version_base="1.3", config_path="../configs", config_name="labels")
def main(dcfg: DictConfig) -> None:
    cfg: LabelCfg = to_cfg(dcfg)
    failed = build_labels(cfg)
    if failed:
        raise SystemExit(f"{len(failed)} Run(s) fehlgeschlagen: {failed}")


if __name__ == "__main__":
    main()
