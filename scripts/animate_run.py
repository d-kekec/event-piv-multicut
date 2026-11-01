#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer visualize.animate_run -- Hydra + Aufruf.

Darstellung [A]: animiert einen ganzen Run, Farbe = Polaritaet. Zweck:
visuell finden, WANN Ueberlappungen und Kreuzungen passieren.

Aufruf (allgemeine Bedienung: scripts/_cli.py). selection.run_nr ist
Pflicht, ausser bei io.list_runs=true: das listet nur die Runs aller
Splits. Ohne window.* der ganze Run, t0 und dt in µs. Liest nur Events,
kein Netz und keine Gruppen; kein sbatch-Skript.
  python scripts/animate_run.py io.list_runs=true
  python scripts/animate_run.py selection.run_nr=8
  python scripts/animate_run.py selection.split=val selection.run_nr=<run> \\
      window.t0=1500000 window.dt=500000 anim.frame_ms=5
"""
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from _cli import to_cfg
from eventpiv.kern import config
from eventpiv.kern.runconfig import AnimRunCfg
from eventpiv.zeigen.visualize import animate_run

ConfigStore.instance().store(name="animrun_schema", node=AnimRunCfg)


@hydra.main(version_base="1.3", config_path="../configs",
            config_name="animate_run")
def main(dcfg: DictConfig) -> None:
    cfg: AnimRunCfg = to_cfg(dcfg)
    if cfg.io.list_runs:
        for s in config.VALID_SPLITS:
            runs = config.available_runs(s)
            print(f"{s}: {len(runs)} Runs -> {runs}")
        return
    if cfg.selection.run_nr is None:
        raise SystemExit("selection.run_nr fehlt (oder io.list_runs=true)")
    animate_run(cfg)


if __name__ == "__main__":
    main()