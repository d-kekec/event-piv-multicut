#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer eventpiv.pairs -- Hydra, W&B, Aufruf."""
from dataclasses import asdict

import hydra
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from _cli import to_cfg
from eventpiv.kern.runconfig import PairCfg, slug
from eventpiv.metrics import flatten
from eventpiv.pairs import build_pair_dataset

ConfigStore.instance().store(name="pairs_schema", node=PairCfg)

PROJECT = "event-piv-multicut"


def ausbeute_tabelle(je_run):
    """Ausbeute je Run als W&B-Tabelle.

    Eine Tabelle und keine Kurve: die Runs haben keine Ordnung in der Zeit.
    ROH neben BEHALTEN je Klasse: JA wird schon beim Sammeln geduennt
    (ja_frac), NEIN erst am Deckel. Erst beide Zahlen sagen, ob das
    Klassenverhaeltnis aus den DATEN folgt oder aus max_yes/max_no.
    """
    t = wandb.Table(columns=["run", "Fenster", "ohne NEIN",
                             "JA roh", "JA behalten", "NEIN roh",
                             "NEIN behalten", "JA gedeckelt",
                             "NEIN gedeckelt", "Sekunden"])
    for z in je_run:
        t.add_data(z["run"], z["n_fenster"], z["n_fenster_ohne_nein"],
                   z["n_roh_ja"], z["n_ja"], z["n_roh_nein"], z["n_nein"],
                   z["gedeckelt_ja"], z["gedeckelt_nein"], z["sec"])
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
            # run.log, nicht run.summary: nur der Log-Eintrag traegt den
            # Deskriptor, aus dem die Oberflaeche ein Panel baut.
            run.log({"pairs/per_run": ausbeute_tabelle(k["per_run"])})
        except Exception as e:
            print(f"  WARN: W&B summary fehlgeschlagen: {e}")

    try:
        build_pair_dataset(cfg, on_summary=on_summary)
    finally:
        run.finish()


if __name__ == "__main__":
    main()