#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer eventpiv.train -- Hydra + Aufruf."""
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from _cli import to_cfg
from eventpiv.runconfig import TrainCfg
from eventpiv.train import train_edge_nn

ConfigStore.instance().store(name="train_schema", node=TrainCfg)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(dcfg: DictConfig) -> None:
    cfg: TrainCfg = to_cfg(dcfg)
    train_edge_nn(cfg)


if __name__ == "__main__":
    main()