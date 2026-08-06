#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
solve.py -- BAUSTEIN [4]: NN-Kantenkosten -> RAMA -> Cluster.

Vier getrennte Zustaendigkeiten:

  graph.neighborhood_pairs  Kanten            (eigenes Modul, mit pairs geteilt)
  edge_costs                Kosten je Kante   (hier)
  solve_multicut            RAMA              (hier)
  evaluate.cluster_metrics  GT-Abgleich       (eigenes Modul)
  solve_and_evaluate        setzt es zusammen (hier)

Nutzen: Auswertung und Visualisierung importieren die Teile einzeln, und
solve_window laeuft dadurch auch ohne Ground Truth. Die Prozess-Isolation
gegen VRAM-Abstuerze liegt in den sbatch-Skripten (ein Python-Prozess je
Lauf), nicht im Python-Code.

POINTER-PFAD-REGELN, hart:
  i,j int32 | costs float32 | node_labels int32-PUFFER selbst anlegen
  (die Funktion schreibt hinein, gibt nichts zurueck) | alles .contiguous()
  | torch.cuda.synchronize() vor UND nach (asynchron!) | num_nodes korrekt
  | leere Eingaben abfangen, sonst cudaErrorIllegalAddress.
PREIS: der Pointer-Pfad gibt die untere Schranke NICHT zurueck -- kein
Optimalitaetszertifikat. Bei Bedarf stichprobenartig ueber rama_cuda.
"""
from __future__ import annotations

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
from eventpiv.evaluate import align_gt_window, cluster_metrics, edge_gt_stats
from eventpiv.graph import edge_deltas, neighborhood_pairs
from eventpiv.model import SWAP_RAW6, build_features, is_symmetric
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


def solve_cc(i, j, costs, num_nodes, cp=None, mode=None):
    """Zusammenhangskomponenten des nicht-negativen Teilgraphen.

    Vergleichssolver zu solve_multicut, mit derselben Signatur -- der
    Aufrufer tauscht die Funktion, nicht seinen Code. cp und mode werden
    ignoriert und stehen nur da, damit die Signaturen austauschbar bleiben.

    VERFAHREN: alle Kanten mit NEGATIVEN Kosten loeschen, vom Rest die
    Zusammenhangskomponenten. Kosten von exakt 0 bleiben also erhalten --
    dieselbe Lesart wie solve/frac_neg, das 0 als nicht-negativ zaehlt.

    Was CC nicht kann: die Zyklusbedingung. Multicut darf eine positive
    Kante schneiden, wenn die Bilanz es verlangt; CC nie. Zwei Wolken
    verschmelzen hier deshalb, sobald EINE Kante zwischen ihnen positiv
    ist. Genau das ist der Fehler, den Stufe 1 nicht machen darf, und
    deshalb ist diese Funktion ein Ergebnis und kein Hilfsmittel.

    Kantenlose Knoten bekommen -1 wie bei RAMA (dort ueber
    sanitize_graph). Ohne diese Angleichung waeren n_clusters_rama und
    n_nodes_unassigned zwischen den beiden Solvern nicht vergleichbar --
    und der Vergleich ist der ganze Zweck. Ein Knoten, der Kanten hat, aber
    nur negative, bekommt dagegen eine echte eigene Cluster-ID: er ist
    zugeordnet, nur allein.

    scipy statt eigenem Union-Find: dieselbe Rechnung, aber in C. Eine
    Python-Schleife ueber 12,5 Mio. Kanten waere Minuten.

    Laeuft auf der CPU und braucht weder GPU noch RAMA. Rueckgabe:
    node_labels (numpy int32) der Laenge num_nodes.
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


# Die eine Stelle, an der aus solver.backend eine Funktion wird. Als Tabelle
# und nicht als if-Kette: ein dritter Solver kostet eine Zeile, und die
# Pruefung des Werts sitzt schon in Solver.__post_init__.
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


def lade_fenster(sel, win, gt_path=None) -> Fenster:
    """Laedt ein Zeitfenster eines Runs und richtet die GT darauf aus.

    Vom Loesen getrennt, weil beides verschiedene Kosten hat: Laden ist
    I/O ueber NFS, Loesen ist Rechnen. solve_window ist dadurch frei von
    Pfaden und laesst sich auf vorgerechneten Arrays pruefen.

    Die GT wird EINMAL geladen und sofort geschnitten. Die vollen Arrays
    (rund 190 MB je Run) leben nur innerhalb des with-Blocks; was das
    Fenster behaelt, ist fenstergross. Vorher geschah dasselbe zweimal --
    einmal fuer die Kantenstatistik, einmal fuer den Cluster-Abgleich.

    Die Spitze im Speicher aendert sich dadurch NICHT: ru_maxrss ist ein
    Hoechststand, und die GT-Spitze lag vorher nur hinter der des KDTrees
    statt davor.

    gt_path = None oder eine fehlende Datei heisst: keine GT. Kein Fehler.
    """
    ev = load_events(sel.split, sel.run_nr, t0=win.t0, dt=win.dt)
    f = Fenster(t=ev["t"].astype(np.float64),
                x=ev["x"].astype(np.float64),
                y=ev["y"].astype(np.float64),
                p=ev["p"])
    if gt_path is not None and Path(gt_path).exists():
        with np.load(gt_path) as g:
            sl = align_gt_window(g["t"], f.t)
            f.track_id = g["track_id"][sl]
            f.status = g["status"][sl]
    return f


def solve_window(cfg: SolveCfg, fenster: Fenster, model, mu, sig,
                 feature_mode, verbose=True):
    """Ein GELADENES Zeitfenster loesen. Rueckgabe: (labels, info).

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
    win = cfg.window

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
    frac_neg = (costs < 0).float().mean().item()
    t_costs = time.time() - tic
    if verbose:
        print(f"  NN-Kosten ({feature_mode}): {frac_neg*100:.1f}% negativ "
              f"[{t_costs:.1f}s]")

    tic = time.time()
    labels = SOLVER_FUNKTIONEN[sv.backend](i, j, costs, num_nodes=n, cp=cp,
                                           mode=sv.mode)
    t_rama = time.time() - tic

    # NACH dem Solve gerechnet: die Statistik legt CPU-Arrays in
    # Kantengroesse an, die nicht waehrend des Solves im Speicher liegen
    # sollen. Die Reihenfolge ist der Grund, nicht der Aufrufer.
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
    #   n_clusters (spaeter aus cluster_metrics)
    #                         wie viele davon enthalten ein SAUBERES Event
    # np.unique(labels) zaehlte frueher den -1-Eimer als Cluster mit, also um
    # genau eins zu hoch, sobald es kantenlose Knoten gibt -- und die gibt es
    # zu Tausenden.
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
            "backend": sv.backend,
            "n_clusters_rama": n_clusters_rama,
            "n_nodes_unassigned": n_unassigned,
            "dt_ms": win.dt / 1000.0 if win.dt else float("nan"),
            "events_per_ms": n / (win.dt / 1000.0) if win.dt else float("nan"),
            "t_graph": t_graph, "t_costs": t_costs, "t_rama": t_rama,
            "vram_peak_gb": vram, "gpu_name": gpu_name,
            # Die Spitze liegt im KDTree, also im RAM, nicht auf der Karte.
            # ru_maxrss ist unter Linux KILObytes; SLURM buchhaltet Speicher
            # auf diesem Cluster nicht.
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

    # feature_mode steht im Checkpoint UND in der Konfiguration. Weichen sie
    # ab, behauptet der abgeleitete Modellpfad etwas anderes, als gerechnet
    # wird.
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

    fenster = lade_fenster(sel, win, gt_path)

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
    
    return info
