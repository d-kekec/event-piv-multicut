#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer eventpiv.tracklets.solve_run -- Hydra, W&B, Aufruf.

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
from eventpiv.kern import config
from eventpiv.kern.runconfig import SolveCfg, slug
from eventpiv.metrics import flatten
from eventpiv.model import load_checkpoint
from eventpiv.solve import check_consistency, info_groups
from eventpiv.tracklets import schreibe_tracklets, solve_run

ConfigStore.instance().store(name="solve_schema", node=SolveCfg)

PROJECT = "event-piv-multicut"


def kanten_tabelle(lauf):
    """Konfusionsmatrix der KANTEN als 2x2-Tabelle, ueber den ganzen Run.

    Zeilen sind die Vorhersage des NETZES (Vorzeichen der Kosten), Spalten
    die GT. Nicht der Schnitt des Solvers: Multicut darf von der
    Kantenpraeferenz abweichen, wenn die Bilanz es verlangt.

    Randsummen stehen mit drin, weil sie im Gespraech sonst im Kopf
    gerechnet werden muessen -- und weil die Spaltensumme rechts die
    Klassenverteilung im Einsatz ist.

    None, wenn der Run keine GT hat.
    """
    tp, fp = lauf.get("edge_tp"), lauf.get("edge_fp")
    fn, tn = lauf.get("edge_fn"), lauf.get("edge_tn")
    if tp is None or (tp + fp + fn + tn) == 0:
        return None
    t = wandb.Table(columns=["", "GT: gleiches Partikel",
                             "GT: verschiedene Partikel", "Summe"])
    t.add_data("Kosten >= 0 (verbinden)", tp, fp, tp + fp)
    t.add_data("Kosten < 0 (schneiden)", fn, tn, fn + tn)
    t.add_data("Summe", tp + fn, fp + tn, tp + fp + fn + tn)
    return t


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
            name=f"solve_tracklets_{split}_run{r}_{cfg.solver.short()}",
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
            tab = kanten_tabelle(lauf)
            if tab is not None:
                # run.log, nicht run.summary: nur der Log-Eintrag traegt den
                # Deskriptor, aus dem die Oberflaeche ein Panel baut.
                # Dieselbe Begruendung wie bei pairs/per_run.
                run.log({"lauf/kanten_konfusion": tab})
            print(f"-> {pfad}", flush=True)
        finally:
            run.finish()


if __name__ == "__main__":
    main()