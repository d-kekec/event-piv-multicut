#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer eventpiv.kette.solve2 -- Hydra, W&B, Aufruf. Stufe [6c].

Ein W&B-Lauf JE RUN, ohne History: ein Solve ist ein Ereignis und keine
Reihe. Das Netz wird EINMAL geladen und ueber alle Runs benutzt.

Je Run entstehen ZWEI Dateien: die Kennzahlen als JSON und die Partition
als npz. Uebersprungen wird an der npz, weil sie die teure ist.

Aufruf (allgemeine Bedienung: scripts/_cli.py). Ohne selection.runs und
selection.run_nr alle Runs des Splits; io.overwrite=true loest
vorhandene neu. Gelesen werden das Netz aus [6b] und je Run (A) und [6a]
desselben Splits, alle mit denselben Gruppen. Einen anderen Split waehlt
nur selection.split: kandidaten.split benennt den Trainingsdatensatz des
Netzes und steht in dessen Pfad. Die train-Runs der Beispiele (8, 13, auf
batch1 32) liegen im Trainingssatz des Netzes; fuer Kennzahlen val nehmen.
Auf batch1 ohne transformersatz=batch1 nimmt es still das Netz mit den
Vorgaben, wenn eines gebaut ist.
  python scripts/solve_stufe2.py 'selection.runs=[8]'
  python scripts/solve_stufe2.py selection.split=val 'selection.runs=[<run>]'
  # batch1
  E="datensatz=batch1 kandidatensatz=batch1 transformersatz=batch1"
  python scripts/solve_stufe2.py $E 'selection.runs=[32]'
  # ueber slurm/solve2.sbatch (Bedarf in dessen Kopf): RUNS und SPLIT
  # Pflicht; OFFSET ist eine Liste fuer solver2.cost_offset, durch
  # Leerzeichen getrennt, je Wert ein Lauf; KD_RUNS ersetzt kandidaten.runs,
  # Overrides in EXTRA, EXTRA_SETS fuehrt mehrere Saetze nacheinander aus,
  # durch ';' getrennt
  RUNS="8 13" SPLIT=train sbatch slurm/solve2.sbatch
  RUNS="32" SPLIT=train EXTRA="$E" sbatch slurm/solve2.sbatch
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
from eventpiv.kette.solve2 import (schreibe_solve2, schreibe_trajektorien,
                                   solve2_run)
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
        tpfad = config.trajektorien_file(
            cfg.neighborhood, cfg.pairs, cfg.training, cfg.solver,
            cfg.kandidaten, cfg.transformer, rcfg.selection, cfg.solver2)
        # Uebersprungen wird HIER und nicht im Modul: ein Abbruch risse die
        # Schleife ab und die Runs danach kaemen nicht mehr dran.
        if tpfad.exists() and not cfg.io.overwrite:
            print(f"-> {tpfad} existiert, uebersprungen "
                  "(io.overwrite=true erzwingt)", flush=True)
            continue
        run = wandb.init(
            project=PROJECT, group=slug(cfg.neighborhood),
            job_type="solve2",
            name=f"solve2_{split}_run{r}_{cfg.solver2.short()}",
            tags=[slug(cfg.kandidaten), slug(cfg.transformer)],
            config=asdict(rcfg))
        try:
            lab_ev, lauf = solve2_run(rcfg, model, norm, r)
            pfad = schreibe_solve2(rcfg, r, lauf)
            tpfad = schreibe_trajektorien(rcfg, r, lab_ev)
            run.summary.update(flatten({"lauf": lauf}))
            run.summary["solve2_path"] = str(pfad)
            run.summary["trajektorien_path"] = str(tpfad)
            print(f"-> {pfad}", flush=True)
            print(f"-> {tpfad}", flush=True)
        finally:
            run.finish()


if __name__ == "__main__":
    main()