#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline.py -- Mehr-Run-Bewertung und Parameterstudien.

Ersetzt evaluate_on_val.py und search_parameters.py. Weil die Solve-Logik
jetzt importierbar ist, entfallen die JSON-Dateien als Datenaustausch
zwischen Prozessen -- ausser dort, wo die Subprozess-Isolation BEWUSST
gewollt ist:

  Die Isolation schuetzt gegen VRAM-Abstuerze (cudaErrorMemoryAllocation /
  cudaErrorIllegalAddress). Ein Absturz killt dann nur den Wegwerf-Subprozess,
  und "nicht loesbar" wird zum ERGEBNIS statt zum Abbruch der Studie
  (BA_NN_Kantenkosten.md §3.8: die Spalten OV_ok / CL_ok sind genau das).
  Fuer die Bewertung ist isolated=True daher der Default.

  Datensatzbau und Training laufen dagegen IN-PROCESS -- sie stuerzen nicht
  ab, sie brauchen nur viel RAM. Wer auf Nummer sicher gehen will, ruft die
  scripts/ statt der Funktionen auf.

BEWERTUNGSLOGIK (uebernommen aus evaluate_on_val.py):
  CROSSING/OVERLAP-Runs = Schnitt-Test (fordern das Trennen)
  CLEAN-Runs            = Kontrolle (duerfen NICHT fragmentieren)
  Beide Gruppen getrennt mitteln -- ein guter Mittelwert ueber beide koennte
  sonst schlechte Trennung durch gute Kontrolle kaschieren.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from eventpiv import config
from eventpiv.pairs import build_pair_dataset, win_dt_for
from eventpiv.solve import solve_and_evaluate
from eventpiv.train import train_edge_nn

T0_DEFAULT = 1_700_000      # µs Fensterstart der Bewertung
DT_DEFAULT = 5_000          # µs Bewertungsfenster (einheitlich = fair)


def _solve_isolated(split, run_nr, model_tag, *, alpha, r_max, t0, dt,
                    model_dir, label_dir, strict, mode="PD"):
    """Ein Run in einem eigenen Prozess. Rueckgabe: info-dict oder None.

    None bedeutet "nicht loesbar" -- Absturz, OOM oder leeres Fenster. Das
    ist ein legitimes Ergebnis, kein Fehler.
    """
    skript = config.PROJECT_ROOT / "scripts" / "solve_with_nn.py"
    with tempfile.TemporaryDirectory() as tmp:
        jf = Path(tmp) / "result.json"
        cmd = [sys.executable, str(skript),
               "--split", split, "--run_nr", str(run_nr),
               "--model_tag", model_tag,
               "--alpha", str(alpha), "--r_max", str(r_max),
               "--t0", str(t0), "--dt", str(dt), "--mode", mode,
               "--json_out", str(jf)]
        if model_dir:
            cmd += ["--model_dir", str(model_dir)]
        if label_dir:
            cmd += ["--label_dir", str(label_dir)]
        if not strict:
            cmd += ["--no_strict"]
        r = subprocess.run(cmd)
        if r.returncode != 0 or not jf.exists():
            return None
        info = json.loads(jf.read_text())
    return info if info.get("purity_mean") is not None else None


def evaluate_runs(split, runs, model_tag, *, alpha, r_max,
                  t0=T0_DEFAULT, dt=DT_DEFAULT, isolated=True,
                  model_dir=None, label_dir=None, strict=True, verbose=True, mode="PD"):
    """Bewertet mehrere Runs. Rueckgabe: Liste aus info-dicts und None."""
    ergebnisse = []
    for r in runs:
        if isolated:
            info = _solve_isolated(split, r, model_tag, alpha=alpha,
                                   r_max=r_max, t0=t0, dt=dt,
                                   model_dir=model_dir, label_dir=label_dir,
                                   strict=strict, mode=mode)
        else:
            try:
                info = solve_and_evaluate(split, r, model_tag, alpha=alpha,
                                          r_max=r_max, t0=t0, dt=dt,
                                          model_dir=model_dir,
                                          label_dir=label_dir, strict=strict,
                                          verbose=False, mode=mode)
                if "purity_mean" not in info:
                    info = None
            except Exception as e:
                print(f"  run{r}: {type(e).__name__}: {e}")
                info = None
        if verbose:
            if info:
                print(f"  run{r}: Rein {info['purity_mean']:.3f}  "
                      f"Frag {info['frag_mean']:.2f}  "
                      f"({info['n_clusters']} Cl / {info['n_gt']} GT, "
                      f"{info['n_edges']:,} Kanten)")
            else:
                print(f"  run{r}: NICHT LOESBAR (Absturz / VRAM / kein GT)")
        ergebnisse.append(info)
    return ergebnisse


def summarize(name, infos, verbose=True) -> dict:
    """Mittelt ueber die loesbaren Runs einer Gruppe."""
    gueltig = [i for i in infos if i]
    if not gueltig:
        if verbose:
            print(f"\n{name}: kein Run loesbar.")
        return {"n_ok": 0, "n_total": len(infos), "purity": float("nan"),
                "frag": float("nan"), "frag_max": None}
    z = {"n_ok": len(gueltig), "n_total": len(infos),
         "purity": float(np.mean([i["purity_mean"] for i in gueltig])),
         "frag": float(np.mean([i["frag_mean"] for i in gueltig])),
         "frag_max": int(max(i["frag_max"] for i in gueltig))}
    if verbose:
        print(f"\n{name} ({z['n_ok']}/{z['n_total']} Runs gemittelt):")
        print(f"  Reinheit       {z['purity']:.3f}")
        print(f"  Fragmentierung {z['frag']:.2f}  (max {z['frag_max']})")
    return z


def evaluate_on_val(model_tag, *, alpha, r_max, crossing=(), clean=(),
                    split="val", t0=T0_DEFAULT, dt=DT_DEFAULT, isolated=True,
                    model_dir=None, label_dir=None, strict=True, mode="PD") -> dict:
    """Bewertung eines Netzes ueber CROSSING- und CLEAN-Runs."""
    print(f"Bewertung: alpha={alpha} r_max={r_max} Netz={model_tag} "
          f"Split={split}")
    print(f"Fenster [{t0/1000:.0f}ms +{dt/1000:.0f}ms]\n")

    gem = dict(alpha=alpha, r_max=r_max, t0=t0, dt=dt, isolated=isolated,
               model_dir=model_dir, label_dir=label_dir, strict=strict, mode=mode)
    print("CROSSING/OVERLAP-Runs (Schnitt-Test):")
    cross = evaluate_runs(split, crossing, model_tag, **gem)
    print("\nCLEAN-Runs (Kontrolle, darf nicht fragmentieren):")
    cl = evaluate_runs(split, clean, model_tag, **gem)

    print("\n" + "=" * 52)
    z_cross = summarize("CROSSING", cross)
    z_clean = summarize("CLEAN", cl)
    print("=" * 52)
    return {"crossing": z_cross, "clean": z_clean,
            "crossing_runs": cross, "clean_runs": cl}


def parameter_sweep(alphas, r_maxs, clean_runs, overlap_runs, *,
                    val_overlap, val_clean, feature_mode="dr",
                    imbalance="sample", train_split="train", val_split="val",
                    solve_dt=DT_DEFAULT, t0=T0_DEFAULT, seed=0, mode="PD",
                    out_json=None) -> list[dict]:
    """Faehrt (alpha, r_max)-Konfigurationen durch und vergleicht sie.

    Kette je Konfiguration: Paar-Datensatz -> Training -> Bewertung.
    WIN_DT wird pro Konfiguration aus der Formel bestimmt, damit die
    NEIN-Ausbeute vergleichbar bleibt (pairs.win_dt_for).

    METRIK-HIERARCHIE (BA_NN_Kantenkosten.md §3.8):
    Fragmentierung > Reinheit >> AUC. AUC steigt bei kleinem r_max
    truegerisch an, waehrend die Cluster zerfallen -- NICHT zur
    Parameterwahl verwenden.
    """
    t_all = time.time()
    # grosse r_max zuerst: vielversprechend und schnell, wichtige Ergebnisse
    # frueh, falls ein kleiner-r_max-Lauf zaeh wird.
    konfigs = [(a, r) for r in sorted(r_maxs, reverse=True) for a in alphas]
    print(f"SWEEP ueber {len(konfigs)} Konfigurationen: "
          f"alpha={list(alphas)}, r_max={list(r_maxs)}")

    tag = f"{feature_mode}_{imbalance}"
    ergebnisse = []
    for alpha, r_max in konfigs:
        wdt = win_dt_for(alpha, r_max)
        print(f"\n{'='*60}\nKONFIG  alpha={alpha}  r_max={r_max}  "
              f"(WIN_DT={wdt})\n{'='*60}")
        try:
            build_pair_dataset(clean_runs, overlap_runs, split=train_split,
                               alpha=alpha, r_max=r_max, win_dt=wdt, seed=seed)
            train_edge_nn(feature_mode=feature_mode, imbalance=imbalance,
                          seed=seed, verbose=True)
        except Exception as e:
            print(f"  Konfiguration uebersprungen: {type(e).__name__}: {e}")
            continue

        z = evaluate_on_val(tag, alpha=alpha, r_max=r_max,
                            crossing=val_overlap, clean=val_clean,
                            split=val_split, t0=t0, dt=solve_dt, mode=mode)
        ergebnisse.append({"alpha": alpha, "r_max": r_max, "win_dt": wdt,
                           "mode": mode, "ov": z["crossing"], "cl": z["clean"]})

    _sweep_tabelle(ergebnisse, len(konfigs), time.time() - t_all)
    if out_json:
        Path(out_json).write_text(json.dumps(
            {"meta": json.loads(config.meta_json(stage="sweep")),
             "results": ergebnisse}, indent=2))
        print(f"-> {out_json}")
    return ergebnisse


def _sweep_tabelle(ergebnisse, n_konfigs, dauer):
    print("\n\n" + "#" * 72)
    print(f"# SWEEP-ERGEBNIS  ({len(ergebnisse)}/{n_konfigs} Konfigs, "
          f"{dauer:.0f}s gesamt)")
    print("#" * 72)
    if not ergebnisse:
        print("Keine erfolgreichen Laeufe.")
        return
    # primaer niedrige Fragmentierung auf den Overlap-Runs, dann Reinheit
    ergebnisse.sort(key=lambda r: (r["ov"]["frag"] if r["ov"]["n_ok"] else 9e9,
                                   -r["ov"]["purity"] if r["ov"]["n_ok"] else 0))
    kopf = (f"{'alpha':>6} {'r_max':>5} | {'OV_rein':>8} {'OV_frag':>8} "
            f"{'OV_ok':>6} | {'CL_rein':>8} {'CL_frag':>8} {'CL_ok':>6}")
    print("\n" + kopf)
    print("-" * len(kopf))
    for r in ergebnisse:
        ov, cl = r["ov"], r["cl"]
        print(f"{r['alpha']:>6} {r['r_max']:>5} | "
              f"{ov['purity']:>8.3f} {ov['frag']:>8.2f} "
              f"{ov['n_ok']}/{ov['n_total']:>4} | "
              f"{cl['purity']:>8.3f} {cl['frag']:>8.2f} "
              f"{cl['n_ok']}/{cl['n_total']:>4}")
    print("\nOV = Overlap-Runs (Schnitt-Test), CL = Clean (Kontrolle).")
    print("'ok' = wieviele Runs loesbar (Rest = VRAM-Absturz, eigenes Ergebnis).")
    print("Gut: beide frag nahe 1.0, hohe Reinheit, viele loesbar.")
    bester = ergebnisse[0]
    print(f"\n-> Bester Kandidat: alpha={bester['alpha']}, "
          f"r_max={bester['r_max']}")
    
def run_pipeline(*, alpha, r_max, clean_runs, overlap_runs,
                 val_crossing=(), val_clean=(), feature_mode="dr",
                 imbalance="sample", train_split="train", val_split="val",
                 win_dt=None, solve_dt=DT_DEFAULT, t0=T0_DEFAULT, seed=0, mode="PD",
                 steps=("labels", "dataset", "train", "evaluate")):
    """Die komplette Kette mit EINEM Satz Parameter.

    Der zentrale Zweck (unveraendert gegenueber run_pipeline_alpha1.py):
    alpha/r_max werden an EINER Stelle gesetzt und an Datensatzbau UND
    Bewertung durchgereicht -> Training und Einsatz sind konsistent.
    Zusaetzlich prueft solve.check_consistency das jetzt hart gegen die
    Metadaten der .pt -- die Konvention ist zur Zusicherung geworden.

    steps: welche Stufen laufen sollen. 'labels' ist meist schon erledigt
    (haengt NICHT von alpha/r_max ab) und wird uebersprungen, wenn die
    Dateien existieren.
    """
    from eventpiv.labels import build_labels

    t_all = time.time()
    wdt = win_dt_for(alpha, r_max) if win_dt is None else win_dt
    tag = f"{feature_mode}_{imbalance}"
    print(f"PIPELINE  alpha={alpha}µs  r_max={r_max}px  WIN_DT={wdt}µs  "
          f"feature={feature_mode}  imbalance={imbalance}")
    print(f"Schritte: {list(steps)}\n")

    if "labels" in steps:
        print("=" * 60 + "\nSCHRITT: labels\n" + "=" * 60)
        build_labels(sorted(set(clean_runs) | set(overlap_runs)),
                     split=train_split)
        val_runs = sorted(set(val_crossing) | set(val_clean))
        if val_runs:
            build_labels(val_runs, split=val_split)

    if "dataset" in steps:
        print("\n" + "=" * 60 + "\nSCHRITT: dataset\n" + "=" * 60)
        build_pair_dataset(clean_runs, overlap_runs, split=train_split,
                           alpha=alpha, r_max=r_max, win_dt=wdt, seed=seed)

    if "train" in steps:
        print("\n" + "=" * 60 + "\nSCHRITT: train\n" + "=" * 60)
        train_edge_nn(feature_mode=feature_mode, imbalance=imbalance,
                      seed=seed)

    ergebnis = None
    if "evaluate" in steps:
        print("\n" + "=" * 60 + "\nSCHRITT: evaluate\n" + "=" * 60)
        ergebnis = evaluate_on_val(tag, alpha=alpha, r_max=r_max,
                                   crossing=val_crossing, clean=val_clean,
                                   split=val_split, t0=t0, dt=solve_dt, mode=mode)

    print("\n" + "=" * 60)
    print(f"PIPELINE KOMPLETT in {time.time()-t_all:.0f}s")
    print("=" * 60)
    return ergebnis
