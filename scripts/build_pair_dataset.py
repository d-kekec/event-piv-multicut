#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer eventpiv.pairs -- Hydra + Aufruf."""
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from _cli import to_cfg
from eventpiv.pairs import build_pair_dataset
from eventpiv.runconfig import PairCfg

ConfigStore.instance().store(name="pairs_schema", node=PairCfg)


@hydra.main(version_base="1.3", config_path="../configs", config_name="pairs")
def main(dcfg: DictConfig) -> None:
    cfg: PairCfg = to_cfg(dcfg)
    build_pair_dataset(cfg)


if __name__ == "__main__":
    main()