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

Nutzen: Auswertung und Visualisierung koennen die Teile einzeln importieren.
Die Prozess-Isolation gegen VRAM-Abstuerze liegt jetzt in den sbatch-Skripten
(ein Python-Prozess je Lauf), nicht mehr im Python-Code.

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
from eventpiv.runconfig import Neighborhood, SolveCfg, canonical, same_neighborhood


def edge_costs(t, x, y, i, j, model, mu, sig, feature_mode, cp):
    """Logit je Kante = Multicut-Kantenkosten. Rueckgabe: float32-GPU-Tensor.

    cp : runconfig.Compute -- device und cost_batch waren hartkodiert
         (device="cuda", batch=4_000_000). Auf tujestpolin mit bis zu sechs
         Karten ist device_index keine Konstante (Befund 27.07.2026).
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
    device, batch = cp.device, cp.cost_batch
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


def solve_multicut(i, j, costs_gpu, num_nodes, cp, mode="PD"):
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
    i_t = torch.as_tensor(np.ascontiguousarray(i, np.int32), device=cp.device)
    j_t = torch.as_tensor(np.ascontiguousarray(j, np.int32), device=cp.device)
    c_t = costs_gpu.contiguous().to(torch.float32)
    
    # Mit -1 vorbelegen, nicht mit 0: 0 ist eine gueltige Cluster-ID. Schreibt
    # RAMA kantenlose Knoten nicht (sanitize_graph entfernt sie), landeten sie
    # sonst still in Cluster 0. -1 heisst "keinem Cluster zugeordnet" und wird
    # von cluster_metrics korrekt herausgefiltert.
    node_labels = torch.full((num_nodes,), -1, device=cp.device,
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
        cp.device_index, opts)
    torch.cuda.synchronize()
    return node_labels.cpu().numpy()



def check_consistency(ckpt, nb: Neighborhood, strict: bool = True) -> None:
    """Prueft, ob das Netz mit DIESER Nachbarschaft trainiert wurde.

    Verglichen wird die kanonische Form der ganzen Gruppe, nicht Feld fuer
    Feld: ein spaeter hinzugefuegtes Feld wird damit automatisch mitgeprueft.
    Eine handgepflegte Liste haette es uebersehen -- belegt an v_max, das in
    beiden Metadatenbloecken fehlte (BA_Konfiguration.md §2). Der frueher
    noetige Float-Vergleich entfaellt; die kanonische Form rundet, 3 und 3.0
    koennen nicht auseinanderlaufen.

    Dateien von VOR dem 27.07.2026 tragen die alte flache Struktur. Sie
    werden bewusst NICHT gelesen: eine Rekonstruktion muesste fehlende Felder
    (geom, v_max) raten, und geratene Identitaet ist genau das, was dieses
    Schema abschafft. Solche Netze laufen mit Warnung und UNGEPRUEFT.
    """
    gruppe = (ckpt.get("meta") or {}).get("neighborhood")
    if not isinstance(gruppe, dict):
        print("  WARN: .pt enthaelt keinen Metadatenblock 'neighborhood' "
              "(vor dem 27.07.2026 erzeugt) -- Konsistenz UNGEPRUEFT.")
        return
    if same_neighborhood(Neighborhood(**gruppe), nb):
        return
    text = (f"Netz trainiert mit {canonical(Neighborhood(**gruppe))}, "
            f"Einsatz mit {canonical(nb)}. Training/Einsatz inkonsistent -- "
            "Datensatz und Netz neu bauen.")
    if strict:
        raise ValueError(text)
    print(f"  WARN: {text}")


def solve_window(cfg: SolveCfg, model, mu, sig, feature_mode, verbose=True):
    """Ein Zeitfenster eines Runs loesen. Rueckgabe: (labels, events, info).

    Nimmt die Konfiguration statt zwoelf Einzelargumenten entgegen. Gelesen
    werden cfg.selection, cfg.window, cfg.neighborhood, cfg.solver, cfg.compute
    -- geschrieben wird NIE in cfg (BA_Konfiguration.md §5).
    """
    nb, sv, cp = cfg.neighborhood, cfg.solver, cfg.compute
    sel, win = cfg.selection, cfg.window

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    ev = load_events(sel.split, sel.run_nr, t0=win.t0, dt=win.dt)
    t = ev["t"].astype(np.float64)
    x = ev["x"].astype(np.float64)
    y = ev["y"].astype(np.float64)
    n = len(t)
    if verbose:
        print(f"  {n:,} Events")
    if n == 0:
        return None, {"t": t, "x": x, "y": y}, {"n_events": 0}

    tic = time.time()
    i, j = neighborhood_pairs(t, x, y, nb.alpha, nb.r_max, geom=nb.geom,
                              v_max=nb.v_max, use_knn=sv.use_knn,
                              k_max=sv.k_max, dedup=sv.use_knn)
    t_graph = time.time() - tic
    if verbose:
        print(f"  {len(i):,} Kanten [{t_graph:.1f}s, {nb.geom}]")
    if len(i) == 0:
        return None, {"t": t, "x": x, "y": y}, {"n_events": n, "n_edges": 0}

    tic = time.time()
    costs = edge_costs(t, x, y, i, j, model, mu, sig, feature_mode, cp)
    frac_neg = (costs < 0).float().mean().item()
    t_costs = time.time() - tic
    if verbose:
        print(f"  NN-Kosten ({feature_mode}): {frac_neg*100:.1f}% negativ "
              f"[{t_costs:.1f}s]")

    tic = time.time()
    labels = solve_multicut(i, j, costs, num_nodes=n, cp=cp, mode=sv.mode)
    t_rama = time.time() - tic

    # np.unique(labels) zaehlte den -1-Eimer als Cluster mit -- also um
    # genau EINS zu hoch, sobald es kantenlose Knoten gibt, und die gibt es
    # zu Tausenden (13.611 von ~18.500 Rauschevents in run32, BA_NN §4.2).
    # Getrennt gefuehrt, weil beides eine eigene Frage beantwortet:
    #   n_clusters_rama       wie viele Cluster hat der Solver gebildet
    #   n_nodes_unassigned    wie viele Knoten hatten gar keine Kante
    # Die spaeter dazukommende n_clusters aus cluster_metrics ist eine DRITTE
    # Zahl: nur Cluster mit mindestens einem sauberen Event. Genau diese
    # Luecke (1142 gegen 56) war bisher unbenannt.
    zugeordnet = labels >= 0
    n_clusters_rama = int(len(np.unique(labels[zugeordnet])))
    n_unassigned = int((~zugeordnet).sum())
    vram = (torch.cuda.max_memory_allocated() / 2**30
            if torch.cuda.is_available() else 0.0)
    if verbose:
        print(f"  RAMA ({sv.mode}): {n_clusters_rama} Cluster, "
              f"{n_unassigned:,} Knoten ohne Kante "
              f"[{t_rama:.1f}s, Torch-Peak {vram:.1f} GB]")

    info = {"n_events": n, "n_edges": int(len(i)), "frac_neg": frac_neg,
            "n_clusters_rama": n_clusters_rama,
            "n_nodes_unassigned": n_unassigned,
            "dt_ms": win.dt / 1000.0 if win.dt else float("nan"),
            "events_per_ms": n / (win.dt / 1000.0) if win.dt else float("nan"),
            "t_graph": t_graph, "t_costs": t_costs, "t_rama": t_rama,
            "vram_peak_gb": vram}
    return labels, {"t": t, "x": x, "y": y}, info


def solve_and_evaluate(cfg: SolveCfg, verbose=True) -> dict:
    """Vollstaendige Kette fuer EINEN Run inklusive GT-Abgleich.

    Der Modellpfad wird ABGELEITET (config.model_file aus neighborhood, pairs,
    training), nicht uebergeben. cfg.io.model_path ist der Notausgang fuer
    fremde Netze -- bewusst ein sichtbarer Akt.
    """
    from pathlib import Path

    from eventpiv.model import load_checkpoint

    mpath = (Path(cfg.io.model_path) if cfg.io.model_path
             else config.model_file(cfg.neighborhood, cfg.pairs, cfg.training,
                                    create=False))
    if not mpath.exists():
        raise FileNotFoundError(
            f"{mpath} fehlt -- erst trainieren, oder io.model_path setzen.")

    model, mu, sig, feature_mode, ckpt = load_checkpoint(
        mpath, device=cfg.compute.device)
    check_consistency(ckpt, cfg.neighborhood, strict=cfg.io.strict)

    # feature_mode steht im Checkpoint UND in der Konfiguration. Bisher wurde
    # der Checkpoint stillschweigend bevorzugt -- dann behauptet der
    # abgeleitete Modellpfad etwas anderes, als gerechnet wird.
    if feature_mode != cfg.training.feature_mode:
        text = (f"Checkpoint hat feature_mode={feature_mode!r}, Konfiguration "
                f"{cfg.training.feature_mode!r} -- Pfad passt nicht zum Modell.")
        if cfg.io.strict:
            raise ValueError(text)
        print(f"  WARN: {text}")

    sel, win = cfg.selection, cfg.window
    if verbose:
        print(f"run{sel.run_nr} ({sel.split}), Fenster [{win.t0/1000:.0f}ms "
              f"+{win.dt/1000:.0f}ms], Netz {mpath.name}, "
              f"k-NN={'AN' if cfg.solver.use_knn else 'AUS'}")

    labels, ev, info = solve_window(cfg, model, mu, sig, feature_mode, verbose)
    info["run"] = sel.run_nr
    if labels is None:
        return info

    gt_path = (Path(cfg.io.label_dir) / f"run{sel.run_nr}_eventlabels.npz"
               if cfg.io.label_dir
               else config.label_file(sel.split, sel.run_nr))
    if not gt_path.exists():
        if verbose:
            print(f"  keine GT-Datei ({gt_path.name}) -> kein Abgleich")
        return info

    gt = np.load(gt_path)
    sl = align_gt_window(gt["t"], ev["t"])
    tic = time.time()
    info.update(cluster_metrics(labels, gt["track_id"][sl], gt["status"][sl],
                                t=ev["t"]))
    info["t_eval"] = time.time() - tic

    if verbose and "purity_cluster_mean" in info:
        print(f"\n  GT-Abgleich (nur saubere Events):")
        print(f"    Cluster: {info['n_clusters']} bewertet von "
              f"{info['n_clusters_rama']} (RAMA)  vs  "
              f"GT-Partikel: {info['n_gt']}")
        print(f"    Reinheit  clustergewichtet {info['purity_cluster_mean']:.3f}, "
              f"eventgewichtet {info['purity_event_mean']:.3f}, "
              f"{info['purity_lt90']} Cluster < 0.90 "
              f"({info['purity_lt90_frac']*100:.1f}%)")
        print(f"    Fragmentierung Mittel {info['frag_mean']:.2f}, "
              f"max {info['frag_max']} (1.0 = perfekt)")
        print(f"    Clustergroessen: min {info['cluster_size_min']}, "
              f"median {info['cluster_size_median']:.0f}, "
              f"max {info['cluster_size_max']}, "
              f"{info['n_singletons']} Singletons")
        print(f"    GT je Partikel: median "
              f"{info['gt_events_per_particle_median']:.0f} Events, "
              f"{info.get('gt_visible_ms_median', float('nan')):.2f} ms sichtbar")
    
    return info
