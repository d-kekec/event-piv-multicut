#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
solve.py -- BAUSTEIN [4]: NN-Kantenkosten -> RAMA -> Cluster.

lade_fenster        Events + ausgerichtete GT, polaritaetsgefiltert
  edge_costs          Kosten je Kante aus dem Netz
  solve_multicut      RAMA          \  ueber solver.backend
  solve_cc            CC-Baseline   /
  solve_window        EIN geladenes Fenster loesen
  solve_and_evaluate  ein Fenster laden, loesen, bewerten
  solve_run           einen ganzen Run in Tracklet-Fenstern abschreiten

solve_window laedt nichts und laeuft ohne Ground Truth wie mit -- also
auch auf echten Messdaten. Die Prozess-Isolation gegen VRAM-Abstuerze
liegt in den sbatch-Skripten, nicht hier.

POINTER-PFAD-REGELN, hart:
  i,j int32 | costs float32 | node_labels int32-PUFFER selbst anlegen
  (die Funktion schreibt hinein, gibt nichts zurueck) | alles .contiguous()
  | torch.cuda.synchronize() vor UND nach (asynchron!) | num_nodes korrekt
  | leere Eingaben abfangen, sonst cudaErrorIllegalAddress.
PREIS: der Pointer-Pfad gibt die untere Schranke NICHT zurueck -- kein
Optimalitaetszertifikat. Bei Bedarf stichprobenartig ueber rama_cuda.
"""
from __future__ import annotations

import contextlib
import os
import sys
import time
import resource
from dataclasses import dataclass

import numpy as np
import torch
from pathlib import Path
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components

from eventpiv import config
from eventpiv.data import load_events
from eventpiv.labels import STATUS_AMBIG
from eventpiv.evaluate import (NAH_PX, abstand_im_fenster, align_gt_window,
                               cluster_metrics, edge_gt_stats, merge_metrics,
                               min_abstand_je_partikel, nah_maske)
from eventpiv.graph import edge_deltas, neighborhood_pairs, polarity_mask
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


def solve_multicut(i, j, costs_gpu, num_nodes, cp, mode="PD"):
    """RAMA ueber den GPU-Pointer-Pfad. Rueckgabe: node_labels (numpy int32).

    Import von rama_py bewusst LOKAL: so laesst sich solve.py importieren
    und testen, ohne dass RAMA installiert ist.
    """
    import rama_py

    num_edges = len(i)
    if num_edges == 0 or num_nodes == 0:
        raise ValueError("Leere Instanz -- der Pointer-Pfad crasht darauf "
                         "(cudaErrorIllegalAddress), vorher abfangen.")

    # int32-Cast auf der CPU: torch.as_tensor(int64_array) legt sonst erst
    # ein int64-Array auf der GPU an und castet danach -- bei 80 Mio. Kanten
    # 636 MB je Richtung umsonst.
    i_t = torch.as_tensor(np.ascontiguousarray(i, np.int32), device=cp.device)
    j_t = torch.as_tensor(np.ascontiguousarray(j, np.int32), device=cp.device)
    c_t = costs_gpu.contiguous().to(torch.float32)
    
    # -1 statt 0 vorbelegen: 0 ist eine gueltige Cluster-ID. Kantenlose
    # Knoten schreibt RAMA nicht und landeten sonst still in Cluster 0.
    node_labels = torch.full((num_nodes,), -1, device=cp.device,
                             dtype=torch.int32).contiguous()

    opts = rama_py.multicut_solver_options(mode)
    opts.sanitize_graph = True      # lueckenhafte Indizes / Doppelkanten erlaubt
    opts.verbose = False

    # PyTorchs Caching-Allocator gibt Speicher nicht an CUDA zurueck, und
    # RAMA allokiert per cudaMalloc: ohne diese Zeile bad_alloc auf einer
    # rechnerisch halbleeren Karte.
    torch.cuda.empty_cache()

    torch.cuda.synchronize()
    rama_py.rama_cuda_gpu_pointers(
        i_t.data_ptr(), j_t.data_ptr(), c_t.data_ptr(),
        node_labels.data_ptr(), num_nodes, num_edges,
        cp.device_index, opts)
    torch.cuda.synchronize()
    return node_labels.cpu().numpy()


def solve_cc(i, j, costs, num_nodes, cp=None, mode=None):
    """Zusammenhangskomponenten des nicht-negativen Teilgraphen.

    Vergleichssolver zu solve_multicut, mit derselben Signatur -- der
    Aufrufer tauscht die Funktion, nicht seinen Code. cp und mode werden
    ignoriert und stehen nur da, damit die Signaturen austauschbar bleiben.

    Alle Kanten mit NEGATIVEN Kosten loeschen, vom Rest die
    Zusammenhangskomponenten. Kosten von exakt 0 bleiben erhalten, wie bei
    solve/frac_neg.

    CC kennt die Zyklusbedingung nicht und darf deshalb nie eine positive
    Kante schneiden: zwei Wolken verschmelzen, sobald EINE Kante zwischen
    ihnen positiv ist. Der Abstand zu RAMA ist damit die Zahl, die den
    Solver rechtfertigt -- diese Funktion ist ein Ergebnis, kein
    Hilfsmittel.

    Kantenlose Knoten bekommen -1 wie bei RAMA, sonst waeren
    n_clusters_rama und n_nodes_unassigned nicht vergleichbar. Ein Knoten
    mit ausschliesslich negativen Kanten bekommt dagegen eine eigene
    Cluster-ID: er ist zugeordnet, nur allein.

    Laeuft auf der CPU, braucht weder GPU noch RAMA. Rueckgabe:
    node_labels (int32) der Laenge num_nodes.
    """
    c = costs.detach().cpu().numpy() if hasattr(costs, "detach") else np.asarray(costs)
    i, j = np.asarray(i), np.asarray(j)
    if len(i) == 0 or num_nodes == 0:
        raise ValueError("Leere Instanz -- wie bei solve_multicut vorher abfangen.")

    behalten = c >= 0.0
    a = coo_array((np.ones(int(behalten.sum()), np.int8),
                   (i[behalten], j[behalten])), shape=(num_nodes, num_nodes))
    _, labels = connected_components(a, directed=False)
    labels = labels.astype(np.int32)

    hat_kante = np.zeros(num_nodes, dtype=bool)
    hat_kante[i] = True
    hat_kante[j] = True
    labels[~hat_kante] = -1
    return labels


SOLVER_FUNKTIONEN = {"rama": solve_multicut, "cc": solve_cc}


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
                               fenster.track_id, fenster.status)
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


@contextlib.contextmanager
def _stumm():
    """Unterdrueckt Ausgaben auf Dateideskriptor 1.

    RAMA meldet je Aufruf "Running purely primal solver"; bei 800 Fenstern
    je Run ist das 800-mal dieselbe Zeile. Die Meldung kommt aus dem
    C++-Teil, deshalb reicht contextlib.redirect_stdout nicht -- das ersetzt
    nur sys.stdout, nicht den Deskriptor darunter.

    stderr bleibt unberuehrt: Fehlermeldungen gehen dorthin und sollen
    sichtbar bleiben. Was hier verloren geht, sind ausschliesslich
    Statuszeilen auf stdout.
    """
    sys.stdout.flush()
    sicherung = os.dup(1)
    leer = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(leer, 1)
        yield
    finally:
        os.dup2(sicherung, 1)
        os.close(leer)
        os.close(sicherung)


def _nah_fern(zeilen):
    """Fasst die Merge-Rate getrennt fuer nahe und ferne Partikel zusammen.

    DIE Frage von Stufe 1: verschmilzt der Solver dort, wo Partikel sich
    begegnen -- und laesst er die uebrigen in Ruhe?

    Getrennt wird je PARTIKEL, nicht je Fenster. Ein Fenster mit 49 weit
    getrennten Partikeln und einem nahen Paar ist weder das eine noch das
    andere; ein Fenster-Etikett muesste sich fuer eine Seite entscheiden
    und waere fuer beide falsch.

    Aufsummiert werden absolute Zahlen, nicht Mittelwerte von Anteilen:
    sonst zaehlte ein Fenster mit zwanzig bewertbaren Events so viel wie
    eines mit zwanzigtausend.

    `fern_merge_event_frac` ist zugleich die Kontrolle. Sie sollte nahe
    null liegen -- verschmilzt dort etwas, sind zwei Partikel
    zusammengefallen, zwischen denen es gar keine Kanten geben duerfte.

    merge_event_frac ist dieselbe Rechnung ohne Trennung. Sie steht daneben,
    weil NAH und FERN allein nicht sagen, wie gross der Fehler insgesamt
    ist -- 8 % auf 7 % der Events sind etwas anderes als 8 % auf allen.
    """
    out = {"nah_schwelle_px": NAH_PX}
    mi_ges = 0
    for pre in ("nah", "fern"):
        ev = sum(z.get(f"{pre}_n_events", 0) for z in zeilen)
        mi = sum(z.get(f"{pre}_n_minderheit", 0) for z in zeilen)
        mi_ges += mi
        out[f"{pre}_n_events"] = ev
        out[f"{pre}_merge_event_frac"] = (mi / ev) if ev else float("nan")
    ges = out["nah_n_events"] + out["fern_n_events"]
    out["nah_event_anteil"] = (out["nah_n_events"] / ges) if ges else float("nan")
    # Die ungetrennte Zahl, aus denselben Summen -- exakt, nicht gemittelt.
    out["merge_event_frac"] = (mi_ges / ges) if ges else float("nan")
    return out


def _gewichtet(zeilen, key, gewicht):
    """Einen Anteil ueber die Fenster eines Runs mitteln, GEWICHTET.

    Ein Mittel von Anteilen zaehlte ein Fenster mit zwanzig Kanten so viel
    wie eines mit zwanzig Millionen -- dieselbe Begruendung wie in
    _nah_fern. Dort liegen die Zaehler vor und die Summe ist exakt; hier
    liegt nur der Anteil vor, also wird er mit seinem Nenner
    zurueckgewichtet.

    Fenster ohne den Schluessel bleiben aussen vor. Fehlt er ueberall, ist
    das Ergebnis nan -- eine 0 sahe wie eine Messung aus.
    """
    paare = [(z[key], z[gewicht]) for z in zeilen
             if key in z and gewicht in z]
    g = sum(w for _, w in paare)
    return (sum(v * w for v, w in paare) / g) if g else float("nan")


def solve_run(cfg: SolveCfg, model, mu, sig, feature_mode, *,
              verbose=True, on_fenster=None, stumm=True):
    """Schreitet EINEN Run in anstossenden Fenstern der Laenge alpha ab.

    Je Fenster unabhaengig geloest, NICHT gestitcht: ein an einer
    Fenstergrenze zerschnittenes Partikel ergibt zwei Tracklets, und das
    ist die beabsichtigte Ausgabe. Verbinden ist Aufgabe von Stufe 2.

    EINMAL geladen, danach mit searchsorted zerlegt -- bei rund 800
    Fenstern je Run traegt das die Laufzeit.

    TRACKLET-IDs SIND GLOBAL: RAMA nummeriert je Fenster ab 0, der Treiber
    addiert einen Versatz. -1 bleibt -1.

    on_fenster : optional Callable(zeile) je Fenster. Dieselbe Loesung wie
        on_epoch im Training -- ein Test haengt an eine Liste an, das Skript
        reicht wandb.log durch.
    
    stumm : RAMAs Statuszeilen unterdruecken. Bei 800 Fenstern je Run ist
        dieselbe Meldung 800-mal keine Information. Zum Nachsehen, was der
        Solver meldet, mit stumm=False aufrufen.

    Rueckgabe: (tracklet_id, zeilen, lauf).
      tracklet_id  int32 je ON-Event des Runs, in der Reihenfolge des
                   gefilterten Stroms. -1 = keinem Tracklet zugeordnet.
      zeilen       eine dict-Zeile je Fenster.
      lauf         Kennzahlen ueber den ganzen Run.
    """

    nb, sel = cfg.neighborhood, cfg.selection
    gt_path = (Path(cfg.io.label_dir) / f"run{sel.run_nr}_eventlabels.npz"
               if cfg.io.label_dir
               else config.label_file(sel.split, sel.run_nr))

    t_start = time.time()
    # cfg.window ist der AUSSCHNITT (None/None = ganzer Run). Die
    # Fensterlaenge des Abschreitens ist davon unberuehrt: die ist alpha.
    ganz = lade_fenster(sel, cfg.window, polarity=nb.polarity,
                        gt_path=gt_path)
    t_laden = time.time() - t_start
    n_ev = len(ganz)
    if verbose:
        ausschnitt = ("ganzer Run" if cfg.window.t0 is None
                      and cfg.window.dt is None
                      else f"AUSSCHNITT {cfg.window.short()}")
        print(f"run{sel.run_nr} ({sel.split}): {n_ev:,} Events "
              f"(polarity={nb.polarity}, {ausschnitt}), GT "
              f"{'ja' if ganz.hat_gt else 'nein'} [{t_laden:.1f}s]")
    if n_ev < 2:
        raise ValueError(f"run{sel.run_nr} hat {n_ev} Events -- nichts zu tun.")

    grenzen = np.searchsorted(ganz.t, np.arange(ganz.t[0], ganz.t[-1], nb.alpha))
    grenzen = np.append(grenzen, n_ev)

    tracklet_id = np.full(n_ev, -1, np.int32)
    versatz = 0
    zeilen = []

    raster_t = gt_ids = d_part = None
    if ganz.hat_gt:
        from eventpiv.data import load_bbox
        raster_t, gt_ids, d_part = min_abstand_je_partikel(
            load_bbox(sel.split, sel.run_nr))

    for k, (lo, hi) in enumerate(zip(grenzen[:-1], grenzen[1:])):
        if hi - lo < 2:
            continue
        teil = Fenster(
            t=ganz.t[lo:hi], x=ganz.x[lo:hi], y=ganz.y[lo:hi],
            p=ganz.p[lo:hi],
            track_id=None if not ganz.hat_gt else ganz.track_id[lo:hi],
            status=None if not ganz.hat_gt else ganz.status[lo:hi])

        with (_stumm() if stumm else contextlib.nullcontext()):
            labels, info = solve_window(cfg, teil, model, mu, sig,
                                        feature_mode, verbose=False,
                                        dt_us=nb.alpha)
        info["fenster"] = k
        info["t0"] = float(teil.t[0])

        if labels is not None:
            gueltig = labels >= 0
            tracklet_id[lo:hi][gueltig] = labels[gueltig] + versatz
            # max()+1 statt der Clusterzahl: RAMA garantiert keine
            # luecklose Nummerierung, und verschenkte IDs kosten nichts.
            versatz += int(labels.max()) + 1
            if teil.hat_gt:
                info.update(cluster_metrics(labels, teil.track_id,
                                            teil.status, t=teil.t))
                info.update(merge_metrics(labels, teil.track_id, teil.status))
                # Cluster werden je Teilmenge neu gebildet -- unbedenklich,
                # weil zwei verschmolzene Partikel einander per Definition
                # nah sind, also beide in derselben Teilmenge liegen.
                nah = nah_maske(teil.track_id, gt_ids,
                                abstand_im_fenster(raster_t, gt_ids, d_part,
                                                   teil.t[0],
                                                   teil.t[0] + nb.alpha))
                for pre, m in (("nah", nah), ("fern", ~nah)):
                    mm = merge_metrics(labels, teil.track_id, teil.status,
                                       extra_maske=m)
                    info[f"{pre}_n_events"] = mm.get("merge_n_events", 0)
                    info[f"{pre}_n_minderheit"] = mm.get("merge_n_minderheit", 0)

        if "edge_ja_frac" in info:
            info["n_nein"] = int(round((1.0 - info["edge_ja_frac"])
                                       * info["edge_n_bewertbar"]))
        if teil.hat_gt:
            info["n_ambig"] = int((teil.status == STATUS_AMBIG).sum())
            info["ambig_frac"] = info["n_ambig"] / len(teil)
        zeilen.append(info)
        if on_fenster is not None:
            on_fenster(info)
        if verbose and (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(grenzen)-1} Fenster, "
                  f"{versatz:,} Tracklets bisher")

    n_tracklets = int((np.unique(tracklet_id) >= 0).sum())
    lauf = {
        "run": sel.run_nr, "split": sel.split,
        "n_events": n_ev,
        "n_fenster": len(zeilen),
        # Default -1 wie im Artefakt: ohne GT ist n_nein nicht bestimmbar.
        "n_fenster_ohne_nein": sum(1 for z in zeilen
                                   if z.get("n_nein", -1) == 0),
        "n_tracklets": n_tracklets,
        "n_events_unassigned": int((tracklet_id < 0).sum()),
        "tracklets_je_fenster": n_tracklets / max(len(zeilen), 1),
        "events_je_tracklet": ((tracklet_id >= 0).sum()
                               / max(n_tracklets, 1)),
        "frac_neg": _gewichtet(zeilen, "frac_neg", "n_edges"),
        "frag_mean": _gewichtet(zeilen, "frag_mean", "n_gt"),
        # _mittel, nicht _median: ein gewichtetes Mittel von Medianen ist
        # kein Median.
        "frag_groesster_anteil_mittel": _gewichtet(
            zeilen, "frag_groesster_anteil_median", "n_gt"),
        "merge_frac_strikt": _gewichtet(zeilen, "merge_frac_strikt",
                                        "merge_n_clusters"),
        "merge_frac_10": _gewichtet(zeilen, "merge_frac_10",
                                    "merge_n_clusters"),
        **_nah_fern(zeilen),
        "diag": {
            "sec_laden": round(t_laden, 1),
            "sec_total": round(time.time() - t_start, 1),
            # EINMAL je Run: ru_maxrss ist eine Hochwassermarke des Prozesses.
            "max_rss_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        },
    }
    if verbose:
        print(f"  {lauf['n_fenster']} Fenster, {n_tracklets:,} Tracklets, "
              f"{lauf['events_je_tracklet']:.0f} Events je Tracklet, "
              f"{lauf['n_fenster_ohne_nein']} Fenster ohne NEIN "
              f"[{lauf['diag']['sec_total']:.0f}s, "
              f"{lauf['diag']['max_rss_mb']:.0f} MB]")
        if lauf.get("nah_n_events"):
            print(f"  NAH (<{NAH_PX:.0f}px Nachbar, "
                  f"{lauf['nah_event_anteil']*100:.0f}% der Events): "
                  f"Merge {lauf['nah_merge_event_frac']*100:.2f}%   |   "
                  f"FERN: Merge "
                  f"{lauf['fern_merge_event_frac']*100:.2f}%")
    return tracklet_id, zeilen, lauf


def _spalte(zeilen, key, default=0):
    """Eine Kennzahl aus allen Fensterzeilen als Array."""
    return np.array([z.get(key, default) for z in zeilen])


def schreibe_tracklets(cfg: SolveCfg, tracklet_id, zeilen, lauf):
    """Artefakt (A): Event -> Tracklet, eine Datei je Run.

    INDIZIERUNG, und ohne diesen Satz ist die Datei unbrauchbar:
    `tracklet_id` hat EINEN Eintrag je ON-Event des Runs, in der Reihenfolge
    des polaritaetsgefilterten Label-Stroms. Ein Leser stellt die Zuordnung
    so her:

        with np.load(config.label_file(split, run)) as d:
            m = polarity_mask(d["p"], nb.polarity)
            t, x, y = d["t"][m], d["x"][m], d["y"][m]

    Danach gehoert `t[k]` zu `tracklet_id[k]`. -1 heisst KEIN Tracklet
    (kantenloser Knoten), nicht Tracklet 0.

    Warum nicht ueber alle Events: OFF-Events sind keine Knoten und
    bekaemen durchweg -1 -- das verdoppelte die Datei fuer nichts. Ihre
    Zuordnung zu Tracklets ist eine eigene, noch offene Frage.

    Daneben je Fenster fuenf Kennzahlen. `fenster_n_nein` ist die
    Volkszaehlung der Begegnungen: sie sagt, welche Fenster ueberhaupt eine
    Trennung verlangen, und ist die Grundlage der CROSSING/CLEAN-Trennung
    bei der Bewertung. -1 heisst "keine GT, also nicht bestimmbar".

    Komprimiert gespeichert: das Array ist ueberwiegend aus kurzen Laeufen
    gleicher IDs aufgebaut und schrumpft dabei erheblich.
    """
    out = (Path(cfg.io.out_path) if cfg.io.out_path
           else config.tracklet_file(cfg.neighborhood, cfg.pairs,
                                     cfg.training, cfg.selection,
                                     cfg.solver, create=True))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        tracklet_id=np.asarray(tracklet_id, np.int32),
        fenster_t0=_spalte(zeilen, "t0", float("nan")).astype(np.float64),
        fenster_n_events=_spalte(zeilen, "n_events").astype(np.int32),
        fenster_n_edges=_spalte(zeilen, "n_edges").astype(np.int64),
        fenster_n_nein=_spalte(zeilen, "n_nein", -1).astype(np.int64),
        fenster_n_ambig=_spalte(zeilen, "n_ambig", -1).astype(np.int64),
        fenster_n_clusters=_spalte(zeilen, "n_clusters_rama", -1).astype(np.int32),
        meta=config.meta_json(
            stage="tracklets",
            **identity_meta(neighborhood=cfg.neighborhood, pairs=cfg.pairs,
                            training=cfg.training, solver=cfg.solver,
                            selection=cfg.selection),
            **lauf))
    return Path(out)