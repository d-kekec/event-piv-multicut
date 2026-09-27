#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer visualize_trajektorien.animate_trajektorien.

Stellt Tracklets, Trajektorien und GT eines Runs nebeneinander. Gelesen
werden die Artefakte von (A), [6c] und fuer das fit-Panel [7] -- also
dieselben Gruppen und Overrides wie beim schreibenden Lauf. window
schneidet nur die Anzeige zu.

  python scripts/animate_trajektorien.py selection.split=val \\
      selection.run_nr=4

  # Fuer den Bericht: ein Panel, kurzer Ausschnitt
  python scripts/animate_trajektorien.py selection.split=val \\
      selection.run_nr=4 anim.panels=trajektorien \\
      window.t0=2000000 window.dt=1000000
"""
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from _cli import to_cfg
from eventpiv.kern.runconfig import AnimTrajCfg
from eventpiv.zeigen.visualize_trajektorien import animate_trajektorien

ConfigStore.instance().store(name="animtraj_schema", node=AnimTrajCfg)


@hydra.main(version_base="1.3", config_path="../configs",
            config_name="animate_trajektorien")
def main(dcfg: DictConfig) -> None:
    cfg: AnimTrajCfg = to_cfg(dcfg)
    if cfg.selection.run_nr is None:
        raise SystemExit("selection.run_nr fehlt -- z.B. selection.run_nr=4")
    animate_trajektorien(cfg)


if __name__ == "__main__":
    main()