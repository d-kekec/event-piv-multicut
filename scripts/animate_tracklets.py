#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer visualize.animate_tracklets -- Hydra + Aufruf.

Darstellung [C]: schreitet einen Ausschnitt in Stufe-1-Fenstern ab wie
[5a] und animiert Rohevents, Tracklets und Ground Truth nebeneinander.
Zweck: SEHEN, dass an jeder Fenstergrenze neue Tracklets beginnen. Schema
und Treiberdatei teilt es mit animate_clusters (AnimClustersCfg,
configs/animate_clusters.yaml). Framezahl = window.dt /
(anim.frame_ms * 1000).

Aufruf (allgemeine Bedienung: scripts/_cli.py). selection.run_nr ist
Pflicht. window.t0 und window.dt, in µs, schneiden den Ausschnitt, ohne
window.* das Vorgabefenster aus runconfig.Window; window.dt muss
mindestens zwei Stufe-1-Fenster umfassen (2 * neighborhood.alpha).
anim.panels waehlt aus roh, tracklets, gt. Braucht das Netz aus [3]; die
Treiberdatei schaltet keine Gruppen zu, es gelten die Vorgaben aus
kern/runconfig.py. Kein sbatch-Skript.
  python scripts/animate_tracklets.py selection.run_nr=8
  # Fragmentierung sehen: Trail an
  W="window.t0=1000000 window.dt=50000 anim.frame_ms=0.5"
  python scripts/animate_tracklets.py selection.run_nr=8 $W
  # Verschmelzung sehen: Momentaufnahmen statt Trail, hoeherer Versatz
  V="anim.trail=false solver.cost_offset=0.5"
  python scripts/animate_tracklets.py selection.run_nr=8 $W $V
  # ein Panel allein, breiter
  python scripts/animate_tracklets.py selection.run_nr=8 anim.panels=tracklets
"""
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from _cli import to_cfg
from eventpiv.kern.runconfig import AnimClustersCfg
from eventpiv.zeigen.visualize import animate_tracklets

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