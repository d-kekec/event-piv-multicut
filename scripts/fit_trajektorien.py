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
import resource
import time
from dataclasses import asdict, replace

import hydra
import numpy as np
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from _cli import to_cfg
from eventpiv.bewerten.evaluate_trajektorien import trajektorien_metriken
from eventpiv.bewerten.metrics import flatten
from eventpiv.kern import config
from eventpiv.kern.data import load_bbox
from eventpiv.kern.graph import polarity_mask
from eventpiv.kern.runconfig import FitCfg, slug
from eventpiv.kette.trajektorien import (fitte_run, gruppiere,
                                         lade_on_strom, schreibe_fit)

ConfigStore.instance().store(name="fit_schema", node=FitCfg)

PROJECT = "event-piv-multicut"


def _zuordnung(cfg, run_nr, tr):
    """{trajektorie_id: (gt_track_id, reinheit)} aus der Label-Datei.

    None ohne Labels -- das ist der Einsatzfall echter Messdaten und kein
    Fehler. Dann entstehen Bahnen, aber keine Guetezahlen.

    Mehrheitspartikel ueber die SAUBEREN Events der Bahn: mehrdeutige
    tragen selbst eine geschaetzte track_id und wuerden die Reinheit
    verwaessern.
    """
    lab = config.label_file(cfg.selection.split, run_nr)
    if not lab.exists():
        return None
    with np.load(lab) as g:
        m = polarity_mask(g["p"], cfg.neighborhood.polarity)
        gt = g["track_id"][m]
        sauber = g["status"][m] == 0
    aus = {}
    for i in range(len(tr)):
        idx = tr.reihenfolge[tr.grenzen[i]:tr.grenzen[i + 1]]
        gs = gt[idx][sauber[idx]]
        gs = gs[gs >= 0]
        if len(gs) == 0:
            continue
        werte, anzahl = np.unique(gs, return_counts=True)
        aus[int(tr.trajektorie_id[i])] = (int(werte[anzahl.argmax()]),
                                          float(anzahl.max() / len(gs)))
    return aus


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
                            beta=cfg.fit.beta,
                            dt_aus=cfg.fit.dt_aus,
                            min_dauer_us=cfg.fit.min_dauer_us,
                            init_us=cfg.fit.init_us,
                            glaetten=cfg.fit.glaetten,
                            a_unbekannt=cfg.fit.a_unbekannt,
                            spline_eps=cfg.fit.spline_eps)
        lauf = {"run": r, "split": split,
                "n_bahnen_partition": len(tr),
                "n_bahnen_gefittet": int(len(np.unique(spalten[5]))),
                "n_punkte": int(len(spalten[0])),
                "u_median_px_ms": float(
                    np.median(np.hypot(spalten[3], spalten[4])) * 1e3)}

        # Bewertung nur mit Ground Truth. Ohne sie entstehen Bahnen, aber
        # keine Guetezahlen -- derselbe Einsatzfall wie bei solve2_run.
        zuo = _zuordnung(rcfg, r, tr)
        if zuo is not None and len(spalten[0]):
            lauf["gt"] = trajektorien_metriken(
                spalten, load_bbox(split, r), zuo,
                v_norm=cfg.neighborhood.v_max * 1e3)
        # Spitzenwert des PROZESSES bis hierher, nicht je Run: ru_maxrss
        # faellt nie. Bei mehreren Runs in einem Aufruf traegt jeder den
        # hoechsten Wert der Runs davor mit -- wie in [6c].
        lauf["diag"] = {"sec_total": round(time.time() - t0, 1),
                        "max_rss_mb": round(
                            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                            / 1024, 1)}

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
                  f"|u| {lauf['u_median_px_ms']:.4f} px/ms "
                  f"[{lauf['diag']['sec_total']:.0f}s, max_rss "
                  f"{lauf['diag']['max_rss_mb']:,.0f} MB]", flush=True)
            if "gt" in lauf:
                g = lauf["gt"]
                print(f"  MedAE_xy {g['medae_xy']:.3f} px  "
                      f"MedAE_u {g['medae_u']:.5f} px/ms "
                      f"({g['medae_u_rel']*100:.2f} % von v_max)  |  "
                      f"rein: {g.get('medae_xy_rein', float('nan')):.3f} px, "
                      f"{g.get('medae_u_rein', float('nan')):.5f} px/ms "
                      f"({g.get('n_bahnen_rein', 0)} von "
                      f"{g['n_bahnen_bewertet']} Bahnen)", flush=True)
                print(f"  du {g['du_signiert']:+.5f} px/ms, "
                      f"{g['du_neg_frac']*100:.0f} % der Bahnen zu langsam",
                      flush=True)
            print(f"-> {pfad}", flush=True)
        finally:
            run.finish()


if __name__ == "__main__":
    main()