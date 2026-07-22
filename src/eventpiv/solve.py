#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
solve.py -- BAUSTEIN [4]: NN-Kantenkosten -> RAMA -> Cluster.

Die Vorgaengerversion mischte VIER Zustaendigkeiten in einer Funktion:
Graphbau, NN-Auswertung, RAMA-Aufruf und GT-Abgleich. Hier getrennt:

  graph.neighborhood_pairs  Kanten            (eigenes Modul, mit pairs geteilt)
  edge_costs                Kosten je Kante   (hier)
  solve_multicut            RAMA              (hier)
  evaluate.cluster_metrics  GT-Abgleich       (eigenes Modul)
  solve_and_evaluate        setzt es zusammen (hier)

Nutzen: evaluate_on_val und search_parameters koennen die Teile einzeln
importieren, statt das ganze Skript als Subprozess zu starten. (Die
Subprozess-Isolation gegen VRAM-Abstuerze bleibt moeglich -- sie ist dann
eine bewusste Entscheidung, kein Zwang aus der Codestruktur.)

POINTER-PFAD-REGELN (BA_RAMA_Verstaendnis.md §3.6, hart):
  i,j int32 | costs float32 | node_labels int32-PUFFER selbst anlegen
  (die Funktion schreibt hinein, gibt nichts zurueck) | alles .contiguous()
  | torch.cuda.synchronize() vor UND nach (asynchron!) | num_nodes korrekt
  | leere Eingaben abfangen, sonst cudaErrorIllegalAddress.
PREIS: der Pointer-Pfad gibt die untere Schranke NICHT zurueck -- kein
Optimalitaetszertifikat. Bei Bedarf stichprobenartig ueber rama_cuda.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from eventpiv import config
from eventpiv.data import load_events
from eventpiv.evaluate import align_gt_window, cluster_metrics
from eventpiv.graph import edge_deltas, neighborhood_pairs
from eventpiv.model import SWAP_RAW6, build_features, is_symmetric


def edge_costs(t, x, y, i, j, model, mu, sig, feature_mode, device="cuda"):
    """Logit je Kante = Multicut-Kantenkosten. Rueckgabe: float32-GPU-Tensor.

    Der Logit wird DIREKT als Kosten verwendet, ohne Nachskalierung -- durch
    Sigmoid+BCE ist er die Log-Odds, also genau die von RAMA erwartete
    Kalibrierung (BA_RAMA_Verstaendnis.md §1.4).

    Die Features entstehen ueber dieselbe build_features wie im Training,
    nur mit xp=torch statt xp=numpy.
    """
    dt_np, dx_np, dy_np = edge_deltas(t, x, y, i, j)
    zu = lambda a: torch.as_tensor(np.ascontiguousarray(a),
                                   dtype=torch.float32, device=device)
    dt, dx, dy = zu(dt_np), zu(dx_np), zu(dy_np)

    extra = {}
    if feature_mode == "raw6":
        extra = {"ti": zu(t[i]), "xi": zu(x[i]), "yi": zu(y[i])}

    feats = build_features(dt, dx, dy, mode=feature_mode, xp=torch, **extra)
    feats = (feats - mu) / sig

    with torch.no_grad():
        logit = model(feats).squeeze(1)
        if not is_symmetric(feature_mode):
            # c_ij muss gleich c_ji sein -> ueber den Tausch mitteln
            roh = feats * sig + mu
            getauscht = (roh[:, SWAP_RAW6] - mu) / sig
            logit = 0.5 * (logit + model(getauscht).squeeze(1))
    return logit


def solve_multicut(i, j, costs_gpu, num_nodes, device_index=0, mode="PD"):
    """RAMA ueber den GPU-Pointer-Pfad. Rueckgabe: node_labels (numpy int32).

    Import von rama_py bewusst LOKAL: so laesst sich solve.py importieren
    (und testen), ohne dass RAMA installiert sein muss.
    """
    import rama_py

    num_edges = len(i)
    if num_edges == 0 or num_nodes == 0:
        raise ValueError("Leere Instanz -- der Pointer-Pfad crasht darauf "
                         "(cudaErrorIllegalAddress), vorher abfangen.")

    i_t = torch.as_tensor(np.ascontiguousarray(i), device="cuda").to(torch.int32)
    j_t = torch.as_tensor(np.ascontiguousarray(j), device="cuda").to(torch.int32)
    c_t = costs_gpu.contiguous().to(torch.float32)
    node_labels = torch.zeros(num_nodes, device="cuda",
                              dtype=torch.int32).contiguous()

    opts = rama_py.multicut_solver_options(mode)
    opts.sanitize_graph = True      # lueckenhafte Indizes / Doppelkanten erlaubt
    opts.verbose = False

    torch.cuda.synchronize()
    rama_py.rama_cuda_gpu_pointers(
        i_t.data_ptr(), j_t.data_ptr(), c_t.data_ptr(),
        node_labels.data_ptr(), num_nodes, num_edges,
        device_index, opts)
    torch.cuda.synchronize()
    return node_labels.cpu().numpy()


def check_consistency(ckpt, alpha, r_max, strict=True):
    """Prueft, ob das Netz mit DIESEN alpha/r_max trainiert wurde.

    Der Kern der Stolperfalle aus BA_NN §3.7: bei geaenderten alpha/r_max
    sieht das Netz im Einsatz eine andere Kantenverteilung als im Training.
    Bisher war die Konsistenz eine Konvention (der Orchestrator reichte beides
    durch) -- jetzt eine Zusicherung.
    """
    meta = ckpt.get("meta", {}) or {}
    trainiert = (meta.get("alpha"), meta.get("r_max"))
    if trainiert == (None, None):
        print("  WARN: .pt enthaelt keine alpha/r_max-Metadaten "
              "(vor der Modularisierung erzeugt) -- Konsistenz UNGEPRUEFT.")
        return
    passt = (trainiert[0] == alpha) and (abs(trainiert[1] - r_max) < 1e-9)
    if passt:
        return
    text = (f"Netz wurde mit alpha={trainiert[0]}, r_max={trainiert[1]} "
            f"trainiert, Einsatz mit alpha={alpha}, r_max={r_max}. "
            "Training/Einsatz inkonsistent -- Datensatz und Netz neu bauen.")
    if strict:
        raise ValueError(text)
    print(f"  WARN: {text}")


def solve_window(split, run_nr, model, mu, sig, feature_mode, *,
                 alpha, r_max, t0, dt, use_knn=False, k_max=24,
                 verbose=True):
    """Ein Zeitfenster eines Runs loesen. Rueckgabe: (labels, events, info).

    events ist ein dict mit t, x, y (float64); info enthaelt Kennzahlen zum
    Lauf selbst (Kantenzahl, Anteil negativer Kosten, Zeiten).
    """
    ev = load_events(split, run_nr, t0=t0, dt=dt)
    t = ev["t"].astype(np.float64)
    x = ev["x"].astype(np.float64)
    y = ev["y"].astype(np.float64)
    n = len(t)
    if verbose:
        print(f"  {n:,} Events")
    if n == 0:
        return None, {"t": t, "x": x, "y": y}, {"n_events": 0}

    tic = time.time()
    i, j = neighborhood_pairs(t, x, y, alpha, r_max,
                              use_knn=use_knn, k_max=k_max, dedup=use_knn)
    t_graph = time.time() - tic
    if verbose:
        print(f"  {len(i):,} Kanten [{t_graph:.1f}s]")
    if len(i) == 0:
        return None, {"t": t, "x": x, "y": y}, {"n_events": n, "n_edges": 0}

    tic = time.time()
    costs = edge_costs(t, x, y, i, j, model, mu, sig, feature_mode)
    frac_neg = (costs < 0).float().mean().item()
    t_costs = time.time() - tic
    if verbose:
        print(f"  NN-Kosten ({feature_mode}): {frac_neg*100:.1f}% negativ "
              f"[{t_costs:.1f}s]")

    tic = time.time()
    labels = solve_multicut(i, j, costs, num_nodes=n)
    t_rama = time.time() - tic
    if verbose:
        print(f"  RAMA: {len(np.unique(labels))} Cluster [{t_rama:.1f}s]")

    info = {"n_events": n, "n_edges": int(len(i)), "frac_neg": frac_neg,
            "t_graph": t_graph, "t_costs": t_costs, "t_rama": t_rama}
    return labels, {"t": t, "x": x, "y": y}, info


def solve_and_evaluate(split, run_nr, model_tag, *, alpha, r_max,
                       t0=1_700_000, dt=5_000, use_knn=False, k_max=24,
                       model_dir=None, label_dir=None, strict=True,
                       verbose=True):
    """Vollstaendige Kette fuer EINEN Run inklusive GT-Abgleich.

    Rueckgabe: dict mit Lauf-Kennzahlen und Cluster-Metriken (leer, wenn
    keine Bewertung moeglich war).
    """
    from pathlib import Path

    from eventpiv.model import load_checkpoint

    mdir = config.model_dir(create=False) if model_dir is None else Path(model_dir)
    model, mu, sig, feature_mode, ckpt = load_checkpoint(
        mdir / f"edge_nn_{model_tag}.pt", device="cuda")
    check_consistency(ckpt, alpha, r_max, strict=strict)

    if verbose:
        print(f"run{run_nr} ({split}), Fenster [{t0/1000:.0f}ms "
              f"+{dt/1000:.0f}ms], Netz '{model_tag}', "
              f"k-NN={'AN' if use_knn else 'AUS'}")

    labels, ev, info = solve_window(split, run_nr, model, mu, sig,
                                    feature_mode, alpha=alpha, r_max=r_max,
                                    t0=t0, dt=dt, use_knn=use_knn,
                                    k_max=k_max, verbose=verbose)
    info["run"] = run_nr
    if labels is None:
        return info

    gt_path = (config.label_file(split, run_nr) if label_dir is None
               else Path(label_dir) / f"run{run_nr}_eventlabels.npz")
    if not gt_path.exists():
        if verbose:
            print(f"  keine GT-Datei ({gt_path.name}) -> kein Abgleich")
        return info

    gt = np.load(gt_path)
    sl = align_gt_window(gt["t"], ev["t"])
    metriken = cluster_metrics(labels, gt["track_id"][sl], gt["status"][sl])
    info.update(metriken)

    if verbose and metriken:
        print(f"\n  GT-Abgleich (nur saubere Events):")
        print(f"    Cluster: {metriken['n_clusters']}  vs  "
              f"GT-Partikel: {metriken['n_gt']}")
        print(f"    Reinheit  Mittel {metriken['purity_mean']:.3f}, "
              f"{metriken['purity_lt90']} Cluster < 0.90")
        print(f"    Fragmentierung Mittel {metriken['frag_mean']:.2f}, "
              f"max {metriken['frag_max']} (1.0 = perfekt)")
    return info
