#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer visualize_trajektorien.animate_trajektorien.

Darstellung [D]: stellt Tracklets, Trajektorien und GT eines Runs
nebeneinander. Gelesen werden die Artefakte von (A), [6c] und fuer die
Panels fit und geschw [7] -- also dieselben Gruppen und Overrides wie beim
schreibenden Lauf. window schneidet nur die Anzeige zu.

Aufruf (allgemeine Bedienung: scripts/_cli.py). selection.run_nr ist
Pflicht; ohne window.* der ganze Run, t0 und dt in µs. anim.panels waehlt
aus roh, tracklets, trajektorien, fit, gt, geschw; fit und geschw lesen
die Bahnen von [7] und brauchen dieselben fit.*-Werte wie dort.
anim.png=true schreibt statt des GIF ein PNG mit dem Stand am Ende von
window. Rechnet nichts nach: kein Netz, keine GPU.
  python scripts/animate_trajektorien.py selection.run_nr=8
  python scripts/animate_trajektorien.py selection.split=val \\
      selection.run_nr=<run>

  # ein Panel, kurzer Ausschnitt
  python scripts/animate_trajektorien.py selection.split=val \\
      selection.run_nr=<run> anim.panels=trajektorien \\
      window.t0=2000000 window.dt=1000000
  # mehrere Panels als EIN Wert, mit fit-Panel
  P="anim.panels='trajektorien,fit,gt'"
  python scripts/animate_trajektorien.py selection.run_nr=8 "$P"
  # Einzelbild: Ausschnitt in Sensorpixeln, vierfach vergroessert
  python scripts/animate_trajektorien.py selection.split=val \\
      selection.run_nr=<run> "anim.panels='fit,gt'" anim.png=true \\
      window.t0=2000000 window.dt=200000 anim.trail_ms=4 \\
      'anim.ausschnitt=[400,200,700,400]' anim.zoom=4
  # nur die Geschwindigkeit, Skala bis 1 px/ms
  python scripts/animate_trajektorien.py selection.split=val \\
      selection.run_nr=<run> anim.panels=geschw anim.png=true anim.u_max=1
  # batch1; mit fit- oder geschw-Panel zusaetzlich fitsatz=batch1
  E="datensatz=batch1 kandidatensatz=batch1 transformersatz=batch1"
  python scripts/animate_trajektorien.py selection.run_nr=32 $E
  # ueber slurm/animate_trajektorien.sbatch (Bedarf in dessen Kopf): SPLIT
  # und RUN Pflicht; PANELS ohne innere Anfuehrungszeichen, die setzt das
  # Skript; ohne PANELS mit fit-Panel, also erst nach [7]; Overrides in EXTRA
  SPLIT=train RUN=8 sbatch slurm/animate_trajektorien.sbatch
  F="$E fitsatz=batch1"
  SPLIT=train RUN=32 EXTRA="$F" sbatch slurm/animate_trajektorien.sbatch
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