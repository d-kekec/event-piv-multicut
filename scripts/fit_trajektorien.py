#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer eventpiv.kette.trajektorien -- Hydra, W&B, Aufruf.

Getrennt vom Solve: der Fit rechnet auf der CPU, der Solve auf der GPU,
und ein neuer Fit-Arbeitspunkt soll den Solve nicht wiederholen muessen.

Ein W&B-Lauf je Run, ohne History -- ein Fit ist ein Ereignis.

Beispielaufruf:
    python scripts/fit_trajektorien.py selection.split=train \\
        selection.runs=[32,40]
"""
import time
from dataclasses import asdict, replace

import hydra
import numpy as np
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from _cli import to_cfg
from eventpiv.bewerten.metrics import flatten
from eventpiv.kern import config
from eventpiv.kern.runconfig import FitCfg, slug
from eventpiv.kette.trajektorien import (fitte_run, gruppiere,
                                         lade_on_strom, schreibe_fit)

ConfigStore.instance().store(name="fit_schema", node=FitCfg)

PROJECT = "event-piv-multicut"


@hydra.main(version_base="1.3", config_path="../configs", config_name="fit")
def main(dcfg: DictConfig) -> None:
    cfg: FitCfg = to_cfg(dcfg)
    split = cfg.selection.split
    runs = list(cfg.selection.runs) or (
        [cfg.selection.run_nr] if cfg.selection.run_nr is not None
        else config.available_runs(split))
    print(f"{len(runs)} Run(s) im Split '{split}', Fit {slug(cfg.fit)}",
          flush=True)

    for r in runs:
        rcfg = replace(cfg, selection=replace(cfg.selection, run_nr=r))
        gruppen = (cfg.neighborhood, cfg.pairs, cfg.training, cfg.solver,
                   cfg.kandidaten, cfg.transformer, rcfg.selection,
                   cfg.solver2)
        out = config.fit_file(*gruppen, cfg.fit, create=True)
        # Uebersprungen wird HIER und nicht im Modul: ein Abbruch risse
        # die Schleife ab und die Runs danach kaemen nicht mehr dran.
        if out.exists() and not cfg.io.overwrite:
            print(f"-> {out} existiert, uebersprungen "
                  "(io.overwrite=true erzwingt)", flush=True)
            continue
        ppfad = config.trajektorien_file(*gruppen)
        if not ppfad.exists():
            print(f"  WARN: {ppfad.name} fehlt, run{r} uebersprungen. "
                  "Erst scripts/solve_stufe2.py.", flush=True)
            continue

        t0 = time.time()
        with np.load(ppfad) as d:
            tid = d["trajektorie_id"]
        t, x, y = lade_on_strom(split, r, cfg.neighborhood.polarity)
        tr = gruppiere(tid, len(t))
        spalten = fitte_run(tr, t, x, y, r_mess=cfg.fit.r_mess,
                            q_c=cfg.fit.q_c, c0=cfg.fit.c0, c1=cfg.fit.c1,
                            dt_aus=cfg.fit.dt_aus,
                            min_dauer_us=cfg.fit.min_dauer_us,
                            init_us=cfg.fit.init_us)
        lauf = {"run": r, "split": split,
                "n_bahnen_partition": len(tr),
                "n_bahnen_gefittet": int(len(np.unique(spalten[7]))),
                "n_punkte": int(len(spalten[0])),
                "sig_pos_median": float(np.median(spalten[5])),
                "sig_u_median": float(np.median(spalten[6]) * 1e3),
                "u_median_px_ms": float(
                    np.median(np.hypot(spalten[3], spalten[4])) * 1e3),
                "diag": {"sec_total": round(time.time() - t0, 1)}}

        run = wandb.init(
            project=PROJECT, group=slug(cfg.neighborhood), job_type="fit",
            name=f"fit_{split}_run{r}_{slug(cfg.fit)}",
            tags=[slug(cfg.kandidaten), slug(cfg.transformer)],
            config=asdict(rcfg))
        try:
            pfad = schreibe_fit(out, rcfg, r, spalten)
            run.summary.update(flatten({"lauf": lauf}))
            run.summary["fit_path"] = str(pfad)
            print(f"run{r}: {lauf['n_bahnen_gefittet']} von "
                  f"{lauf['n_bahnen_partition']} Bahnen, "
                  f"{lauf['n_punkte']:,} Punkte, "
                  f"|u| {lauf['u_median_px_ms']:.4f} px/ms, "
                  f"sig_pos {lauf['sig_pos_median']:.3f} px "
                  f"[{lauf['diag']['sec_total']:.0f}s]", flush=True)
            print(f"-> {pfad}", flush=True)
        finally:
            run.finish()


if __name__ == "__main__":
    main()