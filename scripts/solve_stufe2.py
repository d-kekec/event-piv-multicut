#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer eventpiv.kette.solve2 -- Hydra, W&B, Aufruf.

Ein W&B-Lauf JE RUN, ohne History: ein Solve ist ein Ereignis und keine
Reihe. Das Netz wird EINMAL geladen und ueber alle Runs benutzt.

Beispielaufruf:
    python scripts/solve_stufe2.py selection.split=val selection.runs=[1] \
        solver.cost_offset=-0.5
"""
from dataclasses import asdict, replace
from pathlib import Path

import hydra
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from _cli import to_cfg
from eventpiv.bewerten.metrics import flatten
from eventpiv.kern import config
from eventpiv.kern.runconfig import Solve2Cfg, slug
from eventpiv.kette.solve2 import schreibe_solve2, solve2_run
from eventpiv.kette.transformer import load_checkpoint

ConfigStore.instance().store(name="solve2_schema", node=Solve2Cfg)

PROJECT = "event-piv-multicut"


@hydra.main(version_base="1.3", config_path="../configs",
            config_name="solve2")
def main(dcfg: DictConfig) -> None:
    cfg: Solve2Cfg = to_cfg(dcfg)
    split = cfg.selection.split
    runs = list(cfg.selection.runs) or (
        [cfg.selection.run_nr] if cfg.selection.run_nr is not None
        else config.available_runs(split))

    mpath = (cfg.io.model_path or
             config.transformer_file(cfg.neighborhood, cfg.pairs,
                                     cfg.training, cfg.solver,
                                     cfg.kandidaten, cfg.transformer))
    # kandidaten.split/runs beschreiben den TRAININGSdatensatz und gehen in
    # den Slug ein; was geloest wird, sagt selection. Die Verwechslung
    # baut einen anderen Slug und faellt sonst erst als fehlende Datei auf.
    if cfg.kandidaten.split != "train":
        print(f"  WARN: kandidaten.split={cfg.kandidaten.split!r}. Das Feld "
              "beschreibt den Trainingsdatensatz des NETZES, nicht den zu "
              "loesenden Split -- dafuer ist selection.split da. Gemeint "
              f"war vermutlich selection.split={cfg.kandidaten.split!r}.",
              flush=True)
    if not Path(mpath).exists():
        raise SystemExit(
            f"{mpath} fehlt.\n"
            f"Erwartet: das Netz zu kandidaten={slug(cfg.kandidaten)} und "
            f"transformer={slug(cfg.transformer)}.\n"
            "Haeufigste Ursache: ein geaendertes kandidaten.* verschiebt "
            "den Slug. Zum Loesen eines anderen Splits NUR "
            "selection.split setzen.")
    model, norm, ckpt = load_checkpoint(mpath, device=cfg.compute.device)
    # Das Netz hat auf DIESER Skala gelernt. Ein stiller Wechsel waere die
    # gefaehrlichste Art Fehler: alles laeuft, die Zahlen sind falsch.
    if (norm["alpha"] != float(cfg.kandidaten.alpha)
            or norm["r_max"] != float(cfg.kandidaten.r_max)
            or norm["n_events"] != int(cfg.kandidaten.n_events)):
        raise SystemExit(
            f"Netz und Kandidaten passen nicht zusammen: Checkpoint "
            f"{norm}, Konfiguration alpha={cfg.kandidaten.alpha}, "
            f"r_max={cfg.kandidaten.r_max}, "
            f"n_events={cfg.kandidaten.n_events}.")
    print(f"Netz {mpath}, {len(runs)} Run(s) im Split '{split}', "
          f"Solver {cfg.solver2.short()}", flush=True)

    for r in runs:
        rcfg = replace(cfg, selection=replace(cfg.selection, run_nr=r))
        run = wandb.init(
            project=PROJECT, group=slug(cfg.neighborhood),
            job_type="solve2",
            name=f"solve2_{split}_run{r}_{cfg.solver2.short()}",
            tags=[slug(cfg.kandidaten), slug(cfg.transformer)],
            config=asdict(rcfg))
        try:
            _, lauf = solve2_run(rcfg, model, norm, r)
            pfad = schreibe_solve2(rcfg, r, lauf)
            run.summary.update(flatten({"lauf": lauf}))
            run.summary["solve2_path"] = str(pfad)
            print(f"-> {pfad}", flush=True)
        finally:
            run.finish()


if __name__ == "__main__":
    main()