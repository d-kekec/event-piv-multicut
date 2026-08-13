#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer visualize.animate_clusters -- Hydra + Aufruf.

Loest ein Zeitfenster und animiert Rohevents, RAMA-Cluster und Ground Truth
nebeneinander. Zweck: Reinheit und Fragmentierung SEHEN statt nur als Zahl
lesen. Framezahl = window.dt / anim.frame_ms -- ueber ~100 wird das GIF
unhandlich.

  # Trail an, Farbe nach Cluster-ID: zeigt Fragmentierung
  python scripts/animate_clusters.py selection.split=train \\
      selection.run_nr=3 window.dt=100000 anim.frame_ms=2 anim.downscale=2

  # Momentaufnahmen, Farbe nach GT-Mehrheit: zeigt Verschmelzungen
  python scripts/animate_clusters.py selection.split=train \\
      selection.run_nr=32 anim.frame_ms=0.5 anim.trail=false anim.color_by=gt
"""
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from _cli import to_cfg
from eventpiv.kern.runconfig import AnimClustersCfg
from eventpiv.zeigen.visualize import animate_clusters

ConfigStore.instance().store(name="animclusters_schema", node=AnimClustersCfg)


@hydra.main(version_base="1.3", config_path="../configs",
            config_name="animate_clusters")
def main(dcfg: DictConfig) -> None:
    cfg: AnimClustersCfg = to_cfg(dcfg)
    if cfg.selection.run_nr is None:
        raise SystemExit("selection.run_nr fehlt -- z.B. selection.run_nr=3")
    animate_clusters(cfg)


if __name__ == "__main__":
    main()