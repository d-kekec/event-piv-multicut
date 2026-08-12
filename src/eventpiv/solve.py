#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
solve.py -- BAUSTEIN [4]: NN-Kantenkosten -> Partition eines Fensters.

  lade_fenster        Events + ausgerichtete GT, polaritaetsgefiltert
  edge_costs          Kosten je Kante aus dem Netz
  solve_window        EIN geladenes Fenster loesen
  solve_and_evaluate  ein Fenster laden, loesen, bewerten

Die Solver selbst stehen in loeser.py, hinter solver.backend austauschbar.
Einen ganzen Run abschreiten und das Tracklet-Artefakt schreiben: siehe
tracklets.py.

solve_window laedt nichts und laeuft ohne Ground Truth wie mit -- also
auch auf echten Messdaten. Die Prozess-Isolation gegen VRAM-Abstuerze
liegt in den sbatch-Skripten, nicht hier.
"""
from __future__ import annotations

import time
import resource
from dataclasses import dataclass

import numpy as np
import torch
from pathlib import Path

from eventpiv import config
from eventpiv.data import load_events
from eventpiv.evaluate import (align_gt_window, cluster_metrics,
                               edge_gt_stats, merge_metrics)
from eventpiv.graph import edge_deltas, neighborhood_pairs, polarity_mask
from eventpiv.loeser import SOLVER_FUNKTIONEN
from eventpiv.model import build_features
from eventpiv.runconfig import Neighborhood, SolveCfg, identity_meta


def edge_costs(t, x, y, i, j, model, mu, sig, feature_mode, cp):
    """Logit je Kante = Multicut-Kantenkosten. Rueckgabe: float32-GPU-Tensor.

    cp : runconfig.Compute. device_index ist keine Konstante -- auf
         tujestpolin stecken bis zu sechs Karten in einem Knoten.

    Der Logit wird DIREKT als Kosten verwendet, ohne Nachskalierung: durch
    Sigmoid+BCE ist er die Log-Odds, also genau die von RAMA erwartete
    Kalibrierung. Die Features entstehen ueber dieselbe build_features wie
    im Training, nur mit xp=torch statt xp=numpy.

    STAPELWEISE, und das ist bei grossen Fenstern kein Detail: die versteckte
    Schicht ist (n, hidden) gross, bei 80 Mio. Kanten und hidden=8 also
    2,5 GB -- je Zwischenergebnis, und Linear und Tanh legen je eines an.
    Auf einen Schlag gerechnet sprengt das die Karte, BEVOR RAMA ueberhaupt
    startet. Mit batch=4M bleibt die Spitze bei ~150 MB.
    """
    device, batch = cp.device, cp.cost_batch
    n_edges = len(i)
    out = torch.empty(n_edges, dtype=torch.float32, device=device)

    for s in range(0, n_edges, batch):
        sl = slice(s, min(s + batch, n_edges))
        ii, jj = i[sl], j[sl]

        zu = lambda a: torch.as_tensor(np.ascontiguousarray(a, np.float32),
                                       device=device)
        dt, dx, dy = (zu(a) for a in edge_deltas(t, x, y, ii, jj))

        feats = build_features(dt, dx, dy, mode=feature_mode, xp=torch)
        feats = (feats - mu) / sig

        with torch.no_grad():
            out[sl] = model(feats).squeeze(1)

    return out


def check_consistency(ckpt, nb: Neighborhood, strict: bool = True) -> None:
    """Prueft, ob das Netz mit DIESER Nachbarschaft trainiert wurde.

    Verglichen wird die normierte Form der GANZEN Gruppe gegen die, die beim
    Schreiben in die Datei ging -- nicht Feld fuer Feld. Ein spaeter
    hinzugefuegtes Feld wird damit automatisch mitgeprueft, und 3 gegen 3.0
    kann nicht auseinanderlaufen.

    NICHT ueber Neighborhood(**gruppe): fehlt im gespeicherten Block ein
    Feld, fuellte der Konstruktor still den heutigen Default ein, und ein
    Netz aus der Zeit vor `geom` verglichen sich als passend zu einer
    cylinder-Konfiguration.

    Netze ohne Metadatenblock laufen mit Warnung und UNGEPRUEFT.
    """
    gruppe = (ckpt.get("meta") or {}).get("neighborhood")
    if not isinstance(gruppe, dict):
        print("  WARN: .pt enthaelt keinen Metadatenblock 'neighborhood' "
              "-- Konsistenz UNGEPRUEFT.")
        return
    jetzt = identity_meta(neighborhood=nb)["neighborhood"]
    if gruppe == jetzt:
        return
    text = (f"Netz trainiert mit {gruppe}, Einsatz mit {jetzt}. "
            "Training/Einsatz inkonsistent -- Datensatz und Netz neu bauen.")
    if strict:
        raise ValueError(text)
    print(f"  WARN: {text}")


def info_groups(info: dict) -> dict:
    """Ordnet die flachen Kennzahlen eines Solves in semantische Gruppen.

    Dieselbe Rolle wie train._epochenzeile: das MODUL entscheidet, welche
    Frage eine Zahl beantwortet, das Skript verflacht nur. info selbst bleibt
    flach -- es ist das Dateiformat der solves/*.json.

    REGELBASIERT statt handgepflegte Liste: ein spaeter hinzugefuegtes Feld
    landet automatisch irgendwo, statt still zu verschwinden.
      gt_*                       -> gt/    Verteilungsstatistik der GT
      t_*, vram_peak_gb          -> diag/  Ressourcen, nie Ergebnis
      alles andere               -> solve/ Instanz und Partition

    n_gt landet damit unter solve/, nicht unter gt/ -- absichtlich: es ist
    eine INSTANZGROESSE wie n_events, n_edges und n_clusters, also Teil der
    Frage "wie gross ist eine Sequenz". Die gt_-Felder sind dagegen
    Verteilungen ueber die Partikel.
    """
    gruppen: dict[str, dict] = {"solve": {}, "gt": {}, "diag": {}}
    for k, v in info.items():
        if k.startswith("gt_"):
            gruppen["gt"][k[3:]] = v
        elif k.startswith("t_") or k in ("vram_peak_gb", "gpu_name",
                                         "max_rss_mb"):
            gruppen["diag"][k] = v
        else:
            gruppen["solve"][k] = v
    return gruppen


@dataclass
class Fenster:
    """Ein geladenes Zeitfenster: Events und, falls vorhanden, die dazu
    AUSGERICHTETE Ground Truth.

    Warum ein eigener Typ und kein dict: die Ausrichtung der GT auf den
    Eventstrom ist die fehleranfaelligste Stelle dieser Stufe --
    align_gt_window existiert nur deswegen, und ein gefilterter Strom
    laeuft dort sofort auseinander. Ein Typ macht "ausgerichtet oder gar
    nicht" zu einer Eigenschaft des Objekts statt zu einer Zusage im Kopf.

    track_id und status sind None, wenn es keine Label-Datei gibt -- der
    Normalfall auf echten Messdaten. Sie sind NIE einzeln None.

    t, x, y in float64, weil der KDTree das erwartet. p bleibt in seinem
    Speicherformat: es geht in keine Rechnung ein.
    """
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    track_id: np.ndarray | None = None
    status: np.ndarray | None = None

    @property
    def hat_gt(self) -> bool:
        return self.track_id is not None

    def __len__(self) -> int:
        return len(self.t)


def lade_fenster(sel, win, *, polarity, gt_path=None) -> Fenster:
    """Laedt ein Zeitfenster eines Runs und richtet die GT darauf aus.

    Vom Loesen getrennt: Laden ist I/O ueber NFS, Loesen ist Rechnen.
    solve_window ist dadurch frei von Pfaden und auf vorgerechneten Arrays
    pruefbar.

    Die vollen GT-Arrays (rund 190 MB je Run) leben nur innerhalb des
    with-Blocks; was das Fenster behaelt, ist fenstergross.

    polarity ist Pflicht und ohne Vorgabewert: ein vergessener Filter waere
    im Einsatz eine andere Knotenmenge als im Training, und das faellt an
    keiner Kennzahl auf.

    gt_path = None oder eine fehlende Datei heisst: keine GT. Kein Fehler.
    """
    ev = load_events(sel.split, sel.run_nr, t0=win.t0, dt=win.dt)
    t = ev["t"].astype(np.float64)
    x = ev["x"].astype(np.float64)
    y = ev["y"].astype(np.float64)

    # Erst auf dem UNGEFILTERTEN Strom ausrichten, dann dieselbe Maske auf
    # beide Seiten -- umgekehrt bekaeme align_gt_window eine Teilfolge.
    tid = sta = None
    if gt_path is not None and Path(gt_path).exists():
        with np.load(gt_path) as g:
            sl = align_gt_window(g["t"], t)
            tid = g["track_id"][sl]
            sta = g["status"][sl]

    m = polarity_mask(ev["p"], polarity)
    return Fenster(t=t[m], x=x[m], y=y[m], p=ev["p"][m],
                   track_id=None if tid is None else tid[m],
                   status=None if sta is None else sta[m])


def solve_window(cfg: SolveCfg, fenster: Fenster, model, mu, sig,
                 feature_mode, verbose=True, dt_us=None):
    """Ein GELADENES Zeitfenster loesen. Rueckgabe: (labels, info).

    dt_us : nominale Fensterlaenge in µs, nur fuer die Diagnosefelder
        dt_ms und events_per_ms. None heisst cfg.window.dt. Der Treiber
        reicht alpha herein: beim Abschreiten hat jedes Fenster diese
        Laenge, und cfg.window beschreibt dann den ganzen Run.

    Laedt nichts. Das Fenster kommt fertig herein (lade_fenster), und
    damit ist die Funktion frei von Pfaden und I/O -- pruefbar auf
    vorgerechneten Arrays, ohne Datensatz.

    Die Kantenstatistik gegen die GT wird hier gerechnet, wenn das Fenster
    eine traegt. Frueher kam sie ueber eine Senke herein, weil solve_window
    die GT nicht kannte -- diese Bedingung traegt jetzt das Fenster selbst
    (hat_gt), und eine Senke mit genau einer moeglichen Implementierung
    waere Umweg ohne Wahl. Die Funktion laeuft weiterhin auf Fenstern ohne
    GT, also auf echten Messdaten.

    Nimmt die Konfiguration statt zwoelf Einzelargumenten entgegen; gelesen
    werden window, neighborhood, solver und compute. Geschrieben wird NIE
    in cfg.
    """
    nb, sv, cp = cfg.neighborhood, cfg.solver, cfg.compute
    _dt = cfg.window.dt if dt_us is None else dt_us

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t, x, y = fenster.t, fenster.x, fenster.y
    n = len(fenster)
    if verbose:
        print(f"  {n:,} Events")
    if n == 0:
        return None, {"n_events": 0}

    tic = time.time()
    i, j = neighborhood_pairs(t, x, y, nb.alpha, nb.r_max, geom=nb.geom,
                              v_max=nb.v_max, use_knn=sv.use_knn,
                              k_max=sv.k_max, dedup=sv.use_knn)
    t_graph = time.time() - tic
    if verbose:
        print(f"  {len(i):,} Kanten [{t_graph:.1f}s, {nb.geom}]")
    if len(i) == 0:
        return None, {"n_events": n, "n_edges": 0}

    tic = time.time()
    costs = edge_costs(t, x, y, i, j, model, mu, sig, feature_mode, cp)
    # Getrennt von edge_costs: dort steht, was das Netz rechnet, hier die
    # Entscheidung des Solvers. Beide Anteile werden protokolliert.
    frac_neg_roh = (costs < 0).float().mean().item()
    if sv.cost_offset != 0.0:
        costs += sv.cost_offset
    frac_neg = (costs < 0).float().mean().item()
    t_costs = time.time() - tic
    # Vor dem Solver: RAMAs Message-Passing reparametrisiert die Kosten im
    # selben Puffer. Was wir bewerten wollen, ist die Vorhersage des NETZES.
    # Nur das Vorzeichen, ein bool je Kante -- 1 MB statt 12.
    kosten_pos = (costs >= 0).cpu().numpy()
    if verbose:
        versatz = ("" if sv.cost_offset == 0.0
                   else f", Versatz {sv.cost_offset:+g} "
                        f"(roh {frac_neg_roh*100:.1f}%)")
        print(f"  NN-Kosten ({feature_mode}): {frac_neg*100:.1f}% negativ"
              f"{versatz} "
              f"[{t_costs:.1f}s]")

    tic = time.time()
    labels = SOLVER_FUNKTIONEN[sv.backend](i, j, costs, num_nodes=n, cp=cp,
                                           mode=sv.mode)
    t_rama = time.time() - tic
          
    # NACH dem Solve: die Statistik legt CPU-Arrays in Kantengroesse an.
    kanten = {}
    if fenster.hat_gt:
        tic = time.time()
        kanten = edge_gt_stats(i, j, t, x, y,
                               fenster.track_id, fenster.status,
                               pos=kosten_pos)
        kanten["t_edges"] = time.time() - tic
        if verbose and "edge_ja_frac" in kanten:
            print(f"  Kanten gegen GT: {kanten['edge_ja_frac']*100:.2f}% "
                  f"gleiches Partikel "
                  f"({kanten['edge_bewertbar_frac']*100:.1f}% bewertbar), "
                  f"dt {kanten['dt_mean']/1000:.2f}ms, "
                  f"dr {kanten['dr_mean']:.2f}px")
            
    # DREI Clusterzahlen, drei Fragen -- die Luecke dazwischen ist gross:
    #   n_clusters_rama       wie viele Cluster hat der Solver gebildet
    #   n_nodes_unassigned    wie viele Knoten hatten gar keine Kante
    #   n_clusters (spaeter)  wie viele davon enthalten ein SAUBERES Event
    # np.unique(labels) zaehlte den -1-Eimer mit, also um eins zu hoch.
    zugeordnet = labels >= 0
    n_clusters_rama = int(len(np.unique(labels[zugeordnet])))
    n_unassigned = int((~zugeordnet).sum())
    vram = (torch.cuda.max_memory_allocated() / 2**30
            if torch.cuda.is_available() else 0.0)
    gpu_name = (torch.cuda.get_device_name(cp.device_index)
            if torch.cuda.is_available() else "cpu")
    if verbose:
        wer = f"RAMA ({sv.mode})" if sv.backend == "rama" else "CC-Baseline"
        print(f"  {wer}: {n_clusters_rama} Cluster, "
              f"{n_unassigned:,} Knoten ohne Kante "
              f"[{t_rama:.1f}s, Torch-Peak {vram:.1f} GB]")

    info = {"n_events": n, "n_edges": int(len(i)), "frac_neg": frac_neg,
            "frac_neg_roh": frac_neg_roh, "cost_offset": sv.cost_offset,
            "backend": sv.backend, "mode": sv.mode,
            "n_clusters_rama": n_clusters_rama,
            "n_nodes_unassigned": n_unassigned,
            "dt_ms": _dt / 1000.0 if _dt else float("nan"),
            "events_per_ms": n / (_dt / 1000.0) if _dt else float("nan"),
            "t_graph": t_graph, "t_costs": t_costs, "t_rama": t_rama,
            "vram_peak_gb": vram, "gpu_name": gpu_name,
            # ru_maxrss ist unter Linux KILObytes. Die Spitze liegt im
            # KDTree, also im RAM, nicht auf der Karte.
            "max_rss_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            **kanten}
    return labels, info


def solve_and_evaluate(cfg: SolveCfg, verbose=True) -> dict:
    """Vollstaendige Kette fuer EINEN Run inklusive GT-Abgleich.

    Der Modellpfad wird ABGELEITET (config.model_file aus neighborhood, pairs,
    training), nicht uebergeben. cfg.io.model_path ist der Notausgang fuer
    fremde Netze -- bewusst ein sichtbarer Akt.
    """

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

    # feature_mode steht im Checkpoint UND in der Konfiguration: weichen
    # sie ab, behauptet der Modellpfad etwas anderes, als gerechnet wird.
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

    gt_path = (Path(cfg.io.label_dir) / f"run{sel.run_nr}_eventlabels.npz"
               if cfg.io.label_dir
               else config.label_file(sel.split, sel.run_nr))

    fenster = lade_fenster(sel, win, polarity=cfg.neighborhood.polarity,
                           gt_path=gt_path)

    labels, info = solve_window(cfg, fenster, model, mu, sig, feature_mode,
                                verbose)
    info["run"] = sel.run_nr

    if labels is None:
        return info
    if not fenster.hat_gt:
        if verbose:
            print(f"  keine GT-Datei ({gt_path.name}) -> kein Abgleich")
        return info

    tic = time.time()
    info.update(cluster_metrics(labels, fenster.track_id, fenster.status,
                                t=fenster.t))
    # Zweite Maske, bewusst verschieden: die Merge-Kennzahlen schliessen
    # mehrdeutige Events ein, weil dort verschmolzen wird.
    info.update(merge_metrics(labels, fenster.track_id, fenster.status))
    info["t_eval"] = time.time() - tic

    if verbose and "purity_cluster_mean" in info:
        print(f"\n  GT-Abgleich (nur saubere Events):")
        print(f"    Cluster: {info['n_clusters']} bewertet von "
              f"{info['n_clusters_rama']} ({info.get('backend', 'rama')})"
              f"  vs  GT-Partikel: {info['n_gt']}")
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
        if "merge_event_frac" in info:
            print(f"    Merge ueber {info['merge_n_events']:,} Events "
                  f"(saubere UND mehrdeutige; die Kennzahlen darueber "
                  f"ruhen auf {info['n_events']:,} sauberen): "
                  f"strikt {info['merge_frac_strikt']*100:.1f}%, "
                  f"ab 10% {info['merge_frac_10']*100:.1f}%, "
                  f"Events {info['merge_event_frac']*100:.2f}%")
            print(f"    Groesstes Fragment je Partikel: median "
                  f"{info['frag_groesster_anteil_median']*100:.0f}% "
                  f"seiner Events")
    
    return info