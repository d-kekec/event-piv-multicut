#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer die Merkmalstabelle -- Hydra, W&B, Aufruf.

Abgeleiteter Schritt: liest das Tracklet-Artefakt EINES Runs und den
zugehoerigen Label-Strom, schreibt Artefakt (B). Rechnet den Solve NICHT
neu -- eine Aenderung an einem Merkmal kostet damit Sekunden je Run.

Ein W&B-Lauf je Run, ohne History: die Tabelle hat keine Zeitachse, nur
eine Zusammenfassung.

Beispielaufruf:
    python scripts/build_features.py selection.split=train selection.runs=[32]
"""
from dataclasses import asdict, replace

import hydra
import numpy as np
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from pathlib import Path

from _cli import to_cfg
from eventpiv.bewerten.metrics import flatten
from eventpiv.features import off_zuordnung, tracklet_features
from eventpiv.kern import config
from eventpiv.kern.graph import polarity_mask
from eventpiv.kern.status import STATUS_NOISE
from eventpiv.kern.runconfig import FeatureCfg, identity_meta, slug

ConfigStore.instance().store(name="feature_schema", node=FeatureCfg)

PROJECT = "event-piv-multicut"


def _lade(cfg: FeatureCfg, run_nr: int):
    """Tracklet-Artefakt und Label-Strom EINES Runs, ausgerichtet.

    DER VERTRAG: tracklet_id hat einen Eintrag je ON-Event des Runs, in der
    Reihenfolge des polaritaetsgefilterten Label-Stroms. Ein Versatz um
    einen Eintrag waere still und verfaelschte jedes Merkmal -- deshalb der
    Laengenvergleich, und deshalb bricht er ab statt zu warnen.

    Bewusst NICHT ueber data.load_events: die Label-Datei traegt den
    vollstaendigen Eventstrom, und der Umweg ueber die .dat verlangte
    zusaetzlich align_gt_window. Bewusst auch nicht ueber
    solve.lade_fenster: das wendet polarity_mask an und verwirft damit die
    OFF-Events, die hier gebraucht werden. Wer beides zusammenlegen will,
    muss lade_fenster einen Rueckgabepfad fuer beide Polaritaeten geben.
    """
    sel = replace(cfg.selection, run_nr=run_nr)
    art = config.tracklet_file(cfg.neighborhood, cfg.pairs, cfg.training,
                               sel, cfg.solver)
    if not art.exists():
        raise SystemExit(
            f"{art} fehlt. Erst scripts/build_tracklets.py mit denselben "
            "Parametern laufen lassen.")
    with np.load(art) as d:
        tid_on, fenster_t0 = d["tracklet_id"], d["fenster_t0"]
        

    lab = (config.label_file(cfg.selection.split, run_nr)
           if cfg.io.label_dir is None
           else Path(cfg.io.label_dir) / f"run{run_nr}_eventlabels.npz")
    if not Path(lab).exists():
        # Wie in lade_fenster: keine Labels ist kein Fehler. Echte
        # Messdaten haben keine Boxen, und die Tabelle traegt dann eben
        # keine gt_-Felder. Ohne die Events geht es allerdings nicht --
        # die stehen nur in der Label-Datei.
        raise SystemExit(
            f"{lab} fehlt. Ohne Label-Datei liegt der Eventstrom nicht in "
            "der Reihenfolge des Artefakts vor; fuer Runs ohne GT braucht "
            "es einen eigenen Lesepfad ueber data.load_events.")
    with np.load(lab) as d:
        t, x, y, p = d["t"], d["x"], d["y"], d["p"]
        gt, status = d["track_id"], d["status"]

    m_on = polarity_mask(p, cfg.neighborhood.polarity)
    if int(m_on.sum()) != len(tid_on):
        raise SystemExit(
            f"VERTRAG VERLETZT: {int(m_on.sum()):,} ON-Events im Labelstrom, "
            f"{len(tid_on):,} im Artefakt {art.name}. Andere Polaritaet oder "
            "andere Label-Datei.")
    return art, tid_on, fenster_t0, m_on, t, x, y, gt, status


def _tabelle(cfg: FeatureCfg, tid_on, fenster_t0, m_on, t, x, y, gt, status):
    """Alle Fenster eines Runs zu einer Tabelle.

    Fenstergrenzen ueber searchsorted statt ueber eine Maske je Fenster: t
    ist sortiert, und 800 boolesche Masken ueber 16 Mio. Events waeren der
    Engpass.
    """
    alpha = float(cfg.neighborhood.alpha)
    m_off = ~m_on
    t_on, x_on, y_on = t[m_on], x[m_on], y[m_on]
    t_off, x_off, y_off = t[m_off], x[m_off], y[m_off]
    gt_on, rausch = gt[m_on], status[m_on] == STATUS_NOISE
    hat_gt = bool((gt_on >= 0).any())

    # EXAKT die Rechnung aus solve_run, nicht eine gleichwertige.
    # fenster_t0 im Artefakt ist die Zeit des ERSTEN EVENTS im Fenster
    # (info["t0"] = teil.t[0]), NICHT der Rasterpunkt. Damit zu schneiden
    # zieht die ersten Events des Folgefensters herein, und deren Tracklets
    # bekommen eine zweite Zeile.
    raster = np.arange(t_on[0], t_on[-1], alpha)
    grenzen = np.append(np.searchsorted(t_on, raster), len(t_on))

    teile, j = [], 0
    for k, (lo, hi) in enumerate(zip(grenzen[:-1], grenzen[1:])):
        if hi - lo < 2:
            continue                      # uebersprungen wie in solve_run
        # Vertrag: unsere Fenstergrenze muss die des Artefakts sein. Ein
        # Versatz waere still und verteilte jedes Tracklet auf zwei Zeilen.
        if j >= len(fenster_t0) or abs(t_on[lo] - fenster_t0[j]) > 0.5:
            raise SystemExit(
                f"Fenstergrenze weicht ab: Fenster {k} beginnt bei "
                f"{t_on[lo]:.0f} µs, das Artefakt meldet "
                f"{fenster_t0[j] if j < len(fenster_t0) else float('nan'):.0f}.")
        j += 1

        o_hi = raster[k] + alpha if k + 1 < len(raster) else np.inf
        b0 = int(np.searchsorted(t_off, raster[k], "left"))
        b1 = int(np.searchsorted(t_off, o_hi, "left"))

        zu, kon = off_zuordnung(
            t_on[lo:hi], x_on[lo:hi], y_on[lo:hi], tid_on[lo:hi],
            t_off[b0:b1], x_off[b0:b1], y_off[b0:b1],
            s=cfg.features.off_s, k=cfg.features.off_k)
        z = tracklet_features(
            t_on[lo:hi], x_on[lo:hi], y_on[lo:hi], tid_on[lo:hi],
            t_off[b0:b1], x_off[b0:b1], y_off[b0:b1], zu, kon,
            fenster_idx=k,
            track_id=gt_on[lo:hi] if hat_gt else None,
            status_noise=rausch[lo:hi] if hat_gt else None)
        if z:
            teile.append(z)

    if j != len(fenster_t0):
        raise SystemExit(
            f"{j} Fenster rekonstruiert, {len(fenster_t0)} im Artefakt.")
    if not teile:
        return {}
    return {k: np.concatenate([z[k] for z in teile]) for k in teile[0]}


def _kennzahlen(tab: dict) -> dict:
    """Zusammenfassung der Tabelle. Dieselben Zahlen, an denen die
    Merkmale entschieden wurden -- damit ein spaeterer Lauf sofort zeigt,
    ob sich etwas verschoben hat."""
    n = len(tab["tracklet_id"])
    d = np.hypot(tab["x_on"] - tab["x_off"], tab["y_on"] - tab["y_off"])
    ok = np.isfinite(d)
    out = {
        "n_tracklets": n,
        # Sind einmal auseinandergelaufen, nur zur Pruefung.
        "n_tracklet_ids": int(len(np.unique(tab["tracklet_id"]))),
        "n_fenster": int(tab["fenster_idx"].max()) + 1,
        "n_on_median": float(np.median(tab["n_on"])),
        "klein_frac": float((tab["n_on"] < 15).mean()),
        "klein_event_frac": float(tab["n_on"][tab["n_on"] < 15].sum()
                                  / tab["n_on"].sum()),
        "ohne_off_frac": float((~ok).mean()),
        "d_median": float(np.median(d[ok])) if ok.any() else float("nan"),
        "off_konsens_median": float(np.nanmedian(tab["off_konsens"])),
    }
    if "n_gt" in tab:
        out["rausch_frac"] = float((tab["n_gt"] == 0).mean())
        out["merge_frac_strikt"] = float((tab["n_gt"] >= 2).mean())
        out["gt_anteil_lt90_frac"] = float(
            (tab["gt_anteil"][tab["n_gt"] >= 1] < 0.9).mean())
    return out


@hydra.main(version_base="1.3", config_path="../configs",
            config_name="features")
def main(dcfg: DictConfig) -> None:
    cfg: FeatureCfg = to_cfg(dcfg)
    split = cfg.selection.split
    runs = list(cfg.selection.runs) or (
        [cfg.selection.run_nr] if cfg.selection.run_nr is not None
        else config.available_runs(split))
    print(f"{len(runs)} Run(s) im Split '{split}', "
          f"off_s={cfg.features.off_s}, off_k={cfg.features.off_k}", flush=True)

    for r in runs:
        rcfg = replace(cfg, selection=replace(cfg.selection, run_nr=r))
        out = config.feature_file(cfg.neighborhood, cfg.pairs, cfg.training,
                                  rcfg.selection, cfg.solver, cfg.features,
                                  create=True)
        if out.exists() and not cfg.io.overwrite:
            print(f"-> {out} existiert, uebersprungen "
                  "(io.overwrite=true erzwingt)", flush=True)
            continue

        run = wandb.init(
            project=PROJECT, group=slug(cfg.neighborhood),
            job_type="features",
            name=f"features_{split}_run{r}_{cfg.solver.short()}",
            tags=[slug(cfg.pairs), slug(cfg.training), slug(cfg.features)],
            config=asdict(rcfg))
        try:
            art, tid_on, f_t0, m_on, t, x, y, gt, st = _lade(rcfg, r)
            tab = _tabelle(rcfg, tid_on, f_t0, m_on, t, x, y, gt, st)
            if not tab:
                print(f"run{r}: keine Tracklets", flush=True)
                continue
            kz = _kennzahlen(tab)
            np.savez_compressed(
                out, **tab,
                meta=config.meta_json(
                    stage="features", quelle=str(art),
                    **identity_meta(neighborhood=cfg.neighborhood,
                                    pairs=cfg.pairs, training=cfg.training,
                                    solver=cfg.solver, features=cfg.features,
                                    selection=rcfg.selection),
                    **kz))
            run.summary.update(flatten({"lauf": kz}))
            run.summary["feature_path"] = str(out)
            print(f"run{r}: {kz['n_tracklets']:,} Tracklets, "
                  f"{kz['klein_frac']*100:.1f} % unter 15 Events "
                  f"({kz['klein_event_frac']*100:.2f} % der Events), "
                  f"|D| {kz['d_median']:.2f} px", flush=True)
            print(f"-> {out}", flush=True)
        finally:
            run.finish()


if __name__ == "__main__":
    main()