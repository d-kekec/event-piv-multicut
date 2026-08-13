#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer visualize.animate_tracklets -- Hydra + Aufruf.

Loest ein Zeitfenster und animiert Rohevents, RAMA-Cluster und Ground Truth
nebeneinander. Zweck: Reinheit und Fragmentierung SEHEN statt nur als Zahl
lesen. Framezahl = window.dt / anim.frame_ms -- ueber ~100 wird das GIF
unhandlich.

# Fragmentierung sehen: Trail an, funf Fenster, Zuschnitt
  WANDB_MODE=disabled python scripts/animate_tracklets.py selection.split=train selection.run_nr=32 window.t0=7050000 window.dt=50000 anim.frame_ms=0.5 anim.crop=true

# Verschmelzung sehen: Momentaufnahmen statt Trail, hoeherer Versatz
  WANDB_MODE=disabled python scripts/animate_tracklets.py selection.split=train selection.run_nr=32 window.t0=7050000 window.dt=50000 anim.frame_ms=0.5 solver.cost_offset=0.5

Beispielaufruf starke Überlappung: 
  WANDB_MODE=disabled python scripts/animate_tracklets.py selection.split=train selection.run_nr=76 window.t0=2700000 window.dt=50000 anim.frame_ms=0.5 anim.crop=true
"""
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from _cli import to_cfg
from eventpiv.kern.runconfig import AnimClustersCfg
from eventpiv.visualize import animate_tracklets

ConfigStore.instance().store(name="animclusters_schema", node=AnimClustersCfg)


@hydra.main(version_base="1.3", config_path="../configs",
            config_name="animate_clusters")
def main(dcfg: DictConfig) -> None:
    cfg: AnimClustersCfg = to_cfg(dcfg)
    if cfg.selection.run_nr is None:
        raise SystemExit("selection.run_nr fehlt -- z.B. selection.run_nr=3")
    animate_tracklets(cfg)


if __name__ == "__main__":
    main()