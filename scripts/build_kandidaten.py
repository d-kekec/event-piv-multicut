#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer eventpiv.kette.kandidaten -- Hydra, W&B, Aufruf.

Liest das Tracklet-Artefakt EINES Runs und den zugehoerigen Eventstrom,
schreibt Artefakt [6a]. Rechnet den Solve NICHT neu.

Ein W&B-Lauf je Run, OHNE History: die Knotentabelle hat keine Zeitachse,
nur eine Zusammenfassung.
Anders als bei build_tracklets, wo ein Fenster ein Schritt ist.

Beispielaufruf:
    python scripts/build_kandidaten.py selection.split=train selection.runs=[32]
"""
from dataclasses import asdict, replace

import hydra
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from _cli import to_cfg
from eventpiv.bewerten.metrics import flatten
from eventpiv.kern import config
from eventpiv.kern.runconfig import KandidatenCfg, identity_meta, slug
from eventpiv.kette.kandidaten import kandidaten_fuer_run, schreibe_kandidaten

ConfigStore.instance().store(name="kandidaten_schema", node=KandidatenCfg)

PROJECT = "event-piv-multicut"


def kandidaten_tabelle(lauf):
    """Die Bilanz der Knotenmenge als W&B-Tabelle.

    Aus den Zahlen gebaut, nicht aus den Rohdaten: die Frage ist, WAS der
    Filter entfernt hat und wie viele Kandidaten uebrig bleiben -- vier
    Zeilen, keine Millionen Zeilen.

    Ueber run.log geloggt, nicht ueber run.summary: nur der Log-Eintrag
    traegt den Deskriptor, aus dem die Oberflaeche ein Panel baut.
    GEMESSEN an wandb 0.28.1, dieselbe Begruendung wie in _cli.
    """
    t = wandb.Table(columns=["Groesse", "Wert", "Anteil"])
    t.add_data("Tracklets roh", lauf["n_tracklets_roh"], 1.0)
    t.add_data("Knoten ueber min_events", lauf["n_knoten"],
               lauf["knoten_frac"])
    t.add_data("davon ohne Kandidat", lauf.get("n_knoten_ohne_kandidat", 0),
               lauf.get("knoten_ohne_kandidat_frac", float("nan")))
    t.add_data("Kandidatenpaare", lauf["n_kandidaten"], float("nan"))
    if "n_gueltig" in lauf:
        t.add_data("davon mit Label", lauf["n_gueltig"], lauf["gueltig_frac"])
    return t


@hydra.main(version_base="1.3", config_path="../configs",
            config_name="kandidaten")
def main(dcfg: DictConfig) -> None:
    cfg: KandidatenCfg = to_cfg(dcfg)
    split = cfg.selection.split
    runs = list(cfg.selection.runs) or (
        [cfg.selection.run_nr] if cfg.selection.run_nr is not None
        else config.available_runs(split))
    kd = cfg.kandidaten
    print(f"{len(runs)} Run(s) im Split '{split}', "
          f"alpha={kd.alpha}, r_max={kd.r_max}, geom={kd.geom}, "
          f"min_events={kd.min_events}, n_events={kd.n_events}", flush=True)

    for r in runs:
        rcfg = replace(cfg, selection=replace(cfg.selection, run_nr=r))
        out = config.kandidaten_file(cfg.neighborhood, cfg.pairs,
                                     cfg.training, rcfg.selection,
                                     cfg.solver, kd, create=True)
        if out.exists() and not cfg.io.overwrite:
            print(f"-> {out} existiert, uebersprungen "
                  "(io.overwrite=true erzwingt)", flush=True)
            continue

        run = wandb.init(
            project=PROJECT, group=slug(cfg.neighborhood),
            job_type="kandidaten",
            name=f"kandidaten_{split}_run{r}_{cfg.solver.short()}",
            tags=[slug(cfg.pairs), slug(cfg.training), slug(kd)],
            config=asdict(rcfg))
        print(f"W&B: {run.url}", flush=True)
        try:
            daten, lauf, quelle = kandidaten_fuer_run(rcfg, r)
            pfad = schreibe_kandidaten(rcfg, r, daten, lauf, quelle)
            run.summary.update(flatten({"lauf": lauf}))
            run.summary["kandidaten_path"] = str(pfad)
            run.log({"lauf/bilanz": kandidaten_tabelle(lauf)})
            print(f"-> {pfad}", flush=True)
        finally:
            run.finish()


if __name__ == "__main__":
    main()