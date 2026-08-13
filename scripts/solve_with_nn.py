#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer eventpiv.solve -- Hydra, W&B, Aufruf.

Hydra und W&B leben in scripts/. Ab to_cfg() sieht kein Modul mehr einen
DictConfig (BA_Konfiguration.md §5).
"""

from dataclasses import asdict
from pathlib import Path

import hydra
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from _cli import to_cfg              # scripts/ liegt beim Start auf sys.path
from eventpiv.bewerten.metrics import flatten
from eventpiv.kern import config
from eventpiv.kern.runconfig import SolveCfg, identity_meta, slug
from eventpiv.solve import info_groups, solve_and_evaluate

ConfigStore.instance().store(name="solve_schema", node=SolveCfg)

PROJECT = "event-piv-multicut"


@hydra.main(version_base="1.3", config_path="../configs", config_name="solve")
def main(dcfg: DictConfig) -> None:
    cfg: SolveCfg = to_cfg(dcfg)
    if cfg.selection.run_nr is None:
        raise SystemExit("selection.run_nr fehlt -- z.B. selection.run_nr=3")

    if cfg.io.model_path and not cfg.io.json_out:
        raise SystemExit(
            "io.model_path gesetzt (fremdes Netz) -- dann bitte auch "
            "io.json_out, sonst behauptet der Ergebnispfad eine Herkunft, "
            "die nicht stimmt.")

    out = (cfg.io.json_out or
           config.solve_file(cfg.neighborhood, cfg.pairs, cfg.training,
                             cfg.selection, cfg.window, cfg.solver, create=True))

    # name = der Dateiname des Ergebnisses, also genau das, was diesen Solve
    # von anderen desselben Netzes unterscheidet (Split, Run, Fenster, Modus).
    # Netz und Paar-Datensatz stehen als tags daneben: group ist der
    # Arbeitspunkt, und unter einem Arbeitspunkt liegen mehrere Netze --
    # ueber tags laesst sich darauf filtern, ohne group umzudeuten.
    run = wandb.init(
        project=PROJECT,
        group=slug(cfg.neighborhood),
        job_type="solve",
        name=Path(out).stem,
        tags=[slug(cfg.pairs), slug(cfg.training)],
        config=asdict(cfg),
    )
    print(f"W&B: {run.url}", flush=True)

    try:
        info = solve_and_evaluate(cfg)
        # Ein Solve ist EINE Messung, keine Reihe -> summary statt log.
        # Die Runtabelle zeigt summary, und genau dort will man Runs
        # nebeneinander sehen.
        run.summary.update(flatten(info_groups(info)))

        Path(out).write_text(config.meta_json(
            stage="solve",
            **identity_meta(neighborhood=cfg.neighborhood, pairs=cfg.pairs,
                            training=cfg.training, solver=cfg.solver,
                            window=cfg.window, selection=cfg.selection),
            **info))
        run.summary["json_out"] = str(out)
        print(f"-> {out}")
    finally:
        run.finish()


if __name__ == "__main__":
    main()