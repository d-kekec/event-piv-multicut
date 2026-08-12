#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
loeser.py -- die austauschbaren Solver-Rueckseiten.

  solve_multicut      RAMA ueber den GPU-Pointer-Pfad
  solve_cc            Zusammenhangskomponenten als Vergleichssolver
  SOLVER_FUNKTIONEN   solver.backend -> Funktion

Gleiche Signatur, gleiche Rueckgabe (node_labels, int32, -1 fuer kantenlose
Knoten): der Aufrufer tauscht die Funktion, nicht seinen Code. Eine Tabelle
statt einer if-Kette, weil die Pruefung des Werts schon in
Solver.__post_init__ sitzt.

Dieses Modul kennt das Projekt nicht -- es importiert nichts aus eventpiv
und nimmt nur Arrays und einen Compute-Block entgegen.

POINTER-PFAD-REGELN, hart:
  i,j int32 | costs float32 | node_labels int32-PUFFER selbst anlegen
  (die Funktion schreibt hinein, gibt nichts zurueck) | alles .contiguous()
  | torch.cuda.synchronize() vor UND nach (asynchron!) | num_nodes korrekt
  | leere Eingaben abfangen, sonst cudaErrorIllegalAddress.
PREIS: der Pointer-Pfad gibt die untere Schranke NICHT zurueck -- kein
Optimalitaetszertifikat. Bei Bedarf stichprobenartig ueber rama_cuda.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components


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