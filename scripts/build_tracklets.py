#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer eventpiv.solve.solve_run -- Hydra, W&B, Aufruf.

Stufe 1 ueber ganze Runs. Ein W&B-Lauf JE RUN, ein Fenster ist ein
History-Schritt -- bei rund 800 Fenstern je Run waeren fensterweise Laeufe
unbrauchbar.

Das Netz wird EINMAL geladen und ueber alle Runs benutzt: es ist dasselbe.

Beispielaufruf:
    python scripts/build_tracklets.py selection.split=train selection.runs=[32]
"""
from dataclasses import asdict, replace

import hydra
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from _cli import to_cfg
from eventpiv import config
from eventpiv.metrics import flatten
from eventpiv.model import load_checkpoint
from eventpiv.runconfig import SolveCfg, slug
from eventpiv.solve import (check_consistency, info_groups, schreibe_tracklets,
                            solve_run)

ConfigStore.instance().store(name="solve_schema", node=SolveCfg)

PROJECT = "event-piv-multicut"


def senke(run):
    """on_fenster fuer solve_run.

    'fenster' bleibt UNGRUPPIERT: es ist die x-Achse, und ein Praefix
    liesse define_metric ins Leere greifen.

    Fehlerpolitik wie im Training: ein fehlgeschlagenes log darf den Lauf
    nicht abbrechen, dort steckt schon Rechenzeit drin. Das Artefakt
    entsteht ohnehin unabhaengig von W&B.
    """
    def on_fenster(z):
        try:
            rest = {k: v for k, v in z.items() if k != "fenster"}
            run.log({"fenster": z["fenster"], **flatten(info_groups(rest))})
        except Exception as e:
            print(f"  WARN: W&B log fehlgeschlagen: {e}")
    return on_fenster


@hydra.main(version_base="1.3", config_path="../configs",
            config_name="tracklets")
def main(dcfg: DictConfig) -> None:
    cfg: SolveCfg = to_cfg(dcfg)

    split = cfg.selection.split
    runs = list(cfg.selection.runs) or (
        [cfg.selection.run_nr] if cfg.selection.run_nr is not None
        else config.available_runs(split))

    mpath = (config.model_file(cfg.neighborhood, cfg.pairs, cfg.training)
             if not cfg.io.model_path else cfg.io.model_path)
    model, mu, sig, feature_mode, ckpt = load_checkpoint(
        mpath, device=cfg.compute.device)
    check_consistency(ckpt, cfg.neighborhood, strict=cfg.io.strict)
    print(f"Netz {mpath}, {len(runs)} Run(s) im Split '{split}'", flush=True)

    for r in runs:
        rcfg = replace(cfg, selection=replace(cfg.selection, run_nr=r))
        run = wandb.init(
            project=PROJECT,
            group=slug(cfg.neighborhood),
            job_type="tracklets",
            name=f"{split}_run{r}_{cfg.solver.short()}",
            tags=[slug(cfg.pairs), slug(cfg.training)],
            config=asdict(rcfg),
        )
        # x-Achse ist das Fenster, nicht die Zeit.
        run.define_metric("*", step_metric="fenster")
        print(f"W&B: {run.url}", flush=True)
        try:
            tid, zeilen, lauf = solve_run(rcfg, model, mu, sig, feature_mode,
                                          on_fenster=senke(run))
            pfad = schreibe_tracklets(rcfg, tid, zeilen, lauf)
            run.summary.update(flatten({"lauf": lauf}))
            run.summary["tracklet_path"] = str(pfad)
            print(f"-> {pfad}", flush=True)
        finally:
            run.finish()


if __name__ == "__main__":
    main()