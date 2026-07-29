#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer eventpiv.solve -- Hydra + Aufruf.

Hydra lebt in scripts/, so wie argparse es tat. Ab to_cfg() sieht kein Modul
mehr einen DictConfig (BA_Konfiguration.md §5).
"""
import json

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from _cli import to_cfg              # scripts/ liegt beim Start auf sys.path
from eventpiv import config
from eventpiv.runconfig import SolveCfg, identity_meta
from eventpiv.solve import solve_and_evaluate

# Schema registrieren: sonst nimmt Hydra falsch geschriebene Overrides
# klaglos an (gemessen 27.07.2026).
ConfigStore.instance().store(name="solve_schema", node=SolveCfg)


@hydra.main(version_base="1.3", config_path="../configs", config_name="solve")
def main(dcfg: DictConfig) -> None:
    cfg: SolveCfg = to_cfg(dcfg)
    if cfg.selection.run_nr is None:
        raise SystemExit("selection.run_nr fehlt -- z.B. selection.run_nr=3")

    # Der abgeleitete Ergebnispfad behauptet, das Netz stamme aus genau
    # diesem Zweig. Beim Notausgang io.model_path stimmt das nicht -- dann
    # muss das Ziel explizit genannt werden, statt eine falsche Herkunft
    # zu suggerieren.
    if cfg.io.model_path and not cfg.io.json_out:
        raise SystemExit(
            "io.model_path gesetzt (fremdes Netz) -- dann bitte auch "
            "io.json_out, sonst behauptet der Ergebnispfad eine Herkunft, "
            "die nicht stimmt.")

    info = solve_and_evaluate(cfg)

    out = (cfg.io.json_out or
           config.solve_file(cfg.neighborhood, cfg.pairs, cfg.training,
                             cfg.selection, cfg.window, cfg.solver))
    from pathlib import Path
    # Herkunft mitschreiben (config.py: "jede erzeugte Datei traegt ihre
    # Herkunft"). meta_json legt created/git_commit/host/slurm_job_id und die
    # uebergebenen Felder FLACH nebeneinander -- report.py liest weiterhin
    # info["purity_mean"] auf der obersten Ebene, nichts bricht.
    Path(out).write_text(config.meta_json(
        stage="solve",
        **identity_meta(neighborhood=cfg.neighborhood, pairs=cfg.pairs,
                        training=cfg.training, solver=cfg.solver,
                        window=cfg.window, selection=cfg.selection),
        **info))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
