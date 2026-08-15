#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer visualize_trajektorien.animate_trajektorien.

Loest Stufe 2 fuer einen Run und stellt Tracklets, Trajektorien und GT
nebeneinander. Der Solve laeuft ueber den GANZEN Run, window schneidet nur
die Anzeige zu.

  # Das Bild der Stufe: links flackert es, rechts stehen die Farben
  python scripts/animate_trajektorien.py selection.split=val \\
      selection.run_nr=4 solver.cost_offset=-0.5

  # Fuer den Bericht: ein Panel, gross, kurzer Ausschnitt
  python scripts/animate_trajektorien.py selection.split=val \\
      selection.run_nr=4 solver.cost_offset=-0.5 \\
      anim.panels=trajektorien anim.downscale=1 \\
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