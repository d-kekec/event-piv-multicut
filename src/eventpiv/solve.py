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


def edge_costs(t, x, y, i, j, model, mu, sig, feature_mode, device="cuda",
               batch=4_000_000):
    """Logit je Kante = Multicut-Kantenkosten. Rueckgabe: float32-GPU-Tensor.

    Der Logit wird DIREKT als Kosten verwendet, ohne Nachskalierung -- durch
    Sigmoid+BCE ist er die Log-Odds, also genau die von RAMA erwartete
    Kalibrierung (BA_RAMA_Verstaendnis.md §1.4).

    Die Features entstehen ueber dieselbe build_features wie im Training,
    nur mit xp=torch statt xp=numpy.

    STAPELWEISE, und das ist bei grossen Fenstern kein Detail: die versteckte
    Schicht ist (n, hidden) gross. Bei 80 M Kanten und hidden=8 sind das
    2.5 GB -- pro Zwischenergebnis, und Linear und Tanh legen je eines an.
    Auf einen Schlag gerechnet sprengt das die Karte, bevor RAMA ueberhaupt
    startet. Mit batch=4M bleibt der Spitzenbedarf bei ~150 MB.
    """
    n_edges = len(i)
    out = torch.empty(n_edges, dtype=torch.float32, device=device)

    for s in range(0, n_edges, batch):
        sl = slice(s, min(s + batch, n_edges))
        ii, jj = i[sl], j[sl]

        zu = lambda a: torch.as_tensor(np.ascontiguousarray(a, np.float32),
                                       device=device)
        # edge_deltas statt inline: die Delta-Vorschrift bleibt an EINER
        # Stelle (graph.py), auch wenn hier nur ein Stapel gerechnet wird.
        dt, dx, dy = (zu(a) for a in edge_deltas(t, x, y, ii, jj))

        extra = {}
        if feature_mode == "raw6":
            extra = {"ti": zu(t[ii]), "xi": zu(x[ii]), "yi": zu(y[ii])}

        feats = build_features(dt, dx, dy, mode=feature_mode, xp=torch, **extra)
        feats = (feats - mu) / sig

        with torch.no_grad():
            logit = model(feats).squeeze(1)
            if not is_symmetric(feature_mode):
                # c_ij muss gleich c_ji sein -> ueber den Tausch mitteln
                roh = feats * sig + mu
                getauscht = (roh[:, SWAP_RAW6] - mu) / sig
                logit = 0.5 * (logit + model(getauscht).squeeze(1))
        out[sl] = logit

    return out


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

    # int32-Cast auf der CPU: torch.as_tensor(int64_array) legt sonst erst
    # ein int64-Array auf der GPU an (bei 80 M Kanten 636 MB je Richtung)
    # und castet danach.
    i_t = torch.as_tensor(np.ascontiguousarray(i, np.int32), device="cuda")
    j_t = torch.as_tensor(np.ascontiguousarray(j, np.int32), device="cuda")
    c_t = costs_gpu.contiguous().to(torch.float32)
    
    # Mit -1 vorbelegen, nicht mit 0: 0 ist eine gueltige Cluster-ID. Schreibt
    # RAMA kantenlose Knoten nicht (sanitize_graph entfernt sie), landeten sie
    # sonst still in Cluster 0. -1 heisst "keinem Cluster zugeordnet" und wird
    # von cluster_metrics korrekt herausgefiltert.
    node_labels = torch.full((num_nodes,), -1, device="cuda",
                             dtype=torch.int32).contiguous()

    opts = rama_py.multicut_solver_options(mode)
    opts.sanitize_graph = True      # lueckenhafte Indizes / Doppelkanten erlaubt
    opts.verbose = False

    # PyTorchs Caching-Allocator gibt freigegebenen Speicher NICHT an CUDA
    # zurueck. RAMA allokiert per cudaMalloc und sieht den Cache nicht --
    # ohne diese Zeile scheitert es mit bad_alloc auf einer rechnerisch
    # halbleeren Karte.
    torch.cuda.empty_cache()

    torch.cuda.synchronize()
    rama_py.rama_cuda_gpu_pointers(
        i_t.data_ptr(), j_t.data_ptr(), c_t.data_ptr(),
        node_labels.data_ptr(), num_nodes, num_edges,
        device_index, opts)
    torch.cuda.synchronize()
    return node_labels.cpu().numpy()


def check_consistency(ckpt, alpha, r_max, geom="ellipsoid", strict=True):
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
    passt = (trainiert[0] == alpha) and (abs(trainiert[1] - r_max) < 1e-9) \
        and (meta.get("geom", "ellipsoid") == geom)
    if passt:
        return
    text = (f"Netz wurde mit alpha={trainiert[0]}, r_max={trainiert[1]}, "
            f"geom={meta.get('geom', 'ellipsoid')} trainiert, Einsatz mit "
            f"alpha={alpha}, r_max={r_max}, geom={geom}. "
            "Training/Einsatz inkonsistent -- Datensatz und Netz neu bauen.")
    if strict:
        raise ValueError(text)
    print(f"  WARN: {text}")


def solve_window(split, run_nr, model, mu, sig, feature_mode, *,
                 alpha, r_max, t0, dt, use_knn=False, k_max=24, geom="ellipsoid",
                 mode="PD", verbose=True):
    """Ein Zeitfenster eines Runs loesen. Rueckgabe: (labels, events, info).

    events ist ein dict mit t, x, y (float64); info enthaelt Kennzahlen zum
    Lauf selbst (Kantenzahl, Anteil negativer Kosten, Zeiten).
    """
    # Nur um Peak Memory je Fenster besser zu sehen.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
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
    i, j = neighborhood_pairs(t, x, y, alpha, r_max, geom=geom,
                              use_knn=use_knn, k_max=k_max, dedup=use_knn)
    t_graph = time.time() - tic
    if verbose:
        print(f"  {len(i):,} Kanten [{t_graph:.1f}s, {geom}]")
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
    labels = solve_multicut(i, j, costs, num_nodes=n, mode=mode)
    t_rama = time.time() - tic
    if verbose:
        print(f"  RAMA ({mode}): {len(np.unique(labels))} Cluster "
              f"[{t_rama:.1f}s, Torch-Peak "
              f"{torch.cuda.max_memory_allocated()/2**30:.1f} GB]")

    info = {"n_events": n, "n_edges": int(len(i)), "frac_neg": frac_neg,
            "mode": mode, "geom": geom, "t_graph": t_graph,
            "t_costs": t_costs, "t_rama": t_rama}
    return labels, {"t": t, "x": x, "y": y}, info


def solve_and_evaluate(split, run_nr, model_tag, *, alpha, r_max,
                       t0=1_700_000, dt=5_000, use_knn=False, k_max=24, geom="ellipsoid", mode="PD",
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
    check_consistency(ckpt, alpha, r_max, geom=geom, strict=strict)

    if verbose:
        print(f"run{run_nr} ({split}), Fenster [{t0/1000:.0f}ms "
              f"+{dt/1000:.0f}ms], Netz '{model_tag}', "
              f"k-NN={'AN' if use_knn else 'AUS'}")

    labels, ev, info = solve_window(split, run_nr, model, mu, sig,
                                    feature_mode, alpha=alpha, r_max=r_max,
                                    t0=t0, dt=dt, use_knn=use_knn,
                                    k_max=k_max, geom=geom, mode=mode, verbose=verbose)
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
