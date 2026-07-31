#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer eventpiv.pairs -- Hydra, W&B, Aufruf."""
from dataclasses import asdict

import hydra
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from _cli import to_cfg
from eventpiv.metrics import flatten
from eventpiv.pairs import build_pair_dataset
from eventpiv.runconfig import PairCfg, slug

ConfigStore.instance().store(name="pairs_schema", node=PairCfg)

PROJECT = "event-piv-multicut"


def ausbeute_tabelle(je_run):
    """Ausbeute je Run als W&B-Tabelle.

    Eine Tabelle und keine Kurve: die Runs haben keine Ordnung in der Zeit.
    ROH neben BEHALTEN, damit sichtbar ist, welcher Run am Deckel haengt --
    ein gedeckelter Run sieht im Ergebnis sonst aus wie einer, der zufaellig
    so viel liefert.
    """
    t = wandb.Table(columns=["run", "Klasse", "roh", "behalten",
                             "gedeckelt", "Sekunden"])
    for z in je_run:
        t.add_data(z["run"], z["kind"], z["n_roh"], z["n_behalten"],
                   z["gedeckelt"], z["sec"])
    return t


@hydra.main(version_base="1.3", config_path="../configs", config_name="pairs")
def main(dcfg: DictConfig) -> None:
    cfg: PairCfg = to_cfg(dcfg)

    run = wandb.init(
        project=PROJECT,
        group=slug(cfg.neighborhood),
        job_type="pairs",
        name=slug(cfg.pairs),
        config=asdict(cfg),
    )
    print(f"W&B: {run.url}", flush=True)

    def on_summary(k):
        # Die Tabelle nicht mitverflachen -- sie ist ein W&B-Objekt, kein Wert.
        skalare = {kk: v for kk, v in k.items() if kk != "per_run"}
        try:
            run.summary.update(flatten({"pairs": skalare}))
            run.summary["pairs/per_run"] = ausbeute_tabelle(k["per_run"])
        except Exception as e:
            print(f"  WARN: W&B summary fehlgeschlagen: {e}")

    try:
        build_pair_dataset(cfg, on_summary=on_summary)
    finally:
        run.finish()


if __name__ == "__main__":
    main()