#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
knoten.py -- BAUSTEIN [6a]: aus dem Tracklet-Artefakt die KNOTEN von Stufe 2.

  knoten_bauen  Tracklets als Zeilen: Zentroid, Zeitspanne, GT-Label
  stichprobe    n_events Rohevents je Knoten, plus Maske

Ein Tracklet ist in Stufe 2 EIN Knoten. Was Stufe 1 als Event -> Tracklet
ablegt, wird hier zur Knotenmenge: gefiltert, kompakt nummeriert, mit
Zentroid fuer die Kandidatensuche und einer Eventstichprobe fuer den
Transformer.

NICHT ueber die Merkmalstabelle (B): gelesen wird (A) plus der Eventstrom,
gerechnet wird nur, was Stufe 2 braucht. Die OFF-Zuordnung und alles, was
daran haengt, bleibt aussen vor -- deshalb haengt das Artefakt unter
tracklets/ und nicht unter tracklets/features/.

DER VERTRAG zu (A) gilt unveraendert: tracklet_id hat EINEN Eintrag je
ON-Event des Runs, in der Reihenfolge des polaritaetsgefilterten Stroms.
Geprueft wird er an der Aufrufstelle -- diese Funktionen sehen nur Arrays.

WARUM DIE STICHPROBE INS ARTEFAKT GEHOERT: 93 000 Knoten je Run mal 32
Events sind rund 39 MB, dieselbe Groessenordnung wie (A) selbst. Damit ist
die Auswahl EINGEFROREN: derselbe Knoten traegt in jedem Paar dieselben
Events, im Training wie im Einsatz, und die Reproduzierbarkeit haengt
nicht daran, dass jemand denselben Seed setzt. PREIS: Unterabtasten je
Epoche als Augmentierung ist so nicht moeglich.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eventpiv.kette.features import mehrheit

# Spalten der Knotentabelle, die IMMER da sind. Die drei gt_-Spalten
# kommen nur mit Ground Truth dazu -- wie bei tracklet_features.
SPALTEN = ("tracklet_id", "n_on", "x_c", "y_c", "t_min", "t_max")


@dataclass
class Knoten:
    """Die Knoten eines Runs, plus der Weg zu ihren Events.

    tabelle     Spaltenarrays der Laenge m, in aufsteigender tracklet_id
    reihenfolge Indizes in den ON-Strom, nach Knoten gruppiert
    grenzen     m+1 Offsets in reihenfolge

    Warum ein Typ und nicht drei Rueckgabewerte: die drei gehoeren
    zusammen und sind einzeln wertlos -- grenzen ohne reihenfolge zeigt
    ins Leere. Dieselbe Ueberlegung wie bei Fenster.
    """
    tabelle: dict
    reihenfolge: np.ndarray
    grenzen: np.ndarray

    @property
    def hat_gt(self) -> bool:
        return "gt_track_id" in self.tabelle

    def __len__(self) -> int:
        return len(self.grenzen) - 1


def _leer() -> Knoten:
    return Knoten({k: np.empty(0, np.float64) for k in SPALTEN},
                  np.empty(0, np.int64), np.zeros(1, np.int64))


def knoten_bauen(t_on, x_on, y_on, tracklet_id, *, min_events,
                 track_id=None, status_noise=None) -> Knoten:
    """Knotentabelle EINES Runs. Reine Funktion.

    Erwartet den ON-Strom des ganzen Runs und die zugehoerige
    tracklet_id. Events mit tracklet_id < 0 (kantenlose Knoten) fallen
    heraus, ebenso Knoten unter min_events.

    min_events ist der Kandidatenfilter, und er ist kein Kompromiss: die
    Groessenverteilung ist zweigipflig, unter 15 Events liegt das Mittel
    bei 1,8, darueber bei 166,1, und dazwischen liegt fast nichts. Jede
    Schwelle in dieser Luecke liefert dasselbe (Journal 14.08.2026).

    track_id und status_noise sind optional und nur beim Training
    vorhanden. Ohne sie fehlen die gt_-Spalten, alles andere bleibt --
    dieselbe Regel wie bei tracklet_features.

    NUMMERIERUNG: die Tabelle ist nach tracklet_id aufsteigend sortiert,
    und die IDs von solve_run wachsen fensterweise monoton. Damit ist der
    Zeilenindex zugleich eine Ordnung nach Fenstern, und die kanonische
    Paar-Ordnung braucht kein eigenes Kriterium dafuer.
    """
    tid = np.asarray(tracklet_id)
    gilt = tid >= 0
    if not gilt.any():
        return _leer()
    idx_gilt = np.flatnonzero(gilt)
    werte, kompakt = np.unique(tid[gilt], return_inverse=True)
    m0 = len(werte)

    t = np.asarray(t_on, np.float64)[gilt]
    x = np.asarray(x_on, np.float64)[gilt]
    y = np.asarray(y_on, np.float64)[gilt]

    n_on = np.bincount(kompakt, minlength=m0).astype(np.int64)
    x_c = np.bincount(kompakt, weights=x, minlength=m0) / n_on
    y_c = np.bincount(kompakt, weights=y, minlength=m0) / n_on
    t_min = np.full(m0, np.inf);  np.minimum.at(t_min, kompakt, t)
    t_max = np.full(m0, -np.inf); np.maximum.at(t_max, kompakt, t)

    roh = {"tracklet_id": werte.astype(np.int64), "n_on": n_on,
           "x_c": x_c, "y_c": y_c, "t_min": t_min, "t_max": t_max}

    if track_id is not None and status_noise is not None:
        gt = np.asarray(track_id)[gilt]
        echt = ~np.asarray(status_noise)[gilt]
        sieger, anteil, n_versch = mehrheit(kompakt[echt], m0, gt[echt])
        roh["gt_track_id"] = sieger
        roh["gt_anteil"] = anteil
        roh["n_gt"] = n_versch.astype(np.int32)

    behalten = n_on >= int(min_events)
    m = int(behalten.sum())
    if m == 0:
        return _leer()
    tabelle = {k: v[behalten] for k, v in roh.items()}

    # Nach dem Filter neu durchnummerieren. -1 fuer die Verworfenen, damit
    # ein Zugriff darauf auffaellt statt still auf Knoten 0 zu zeigen.
    neu = np.full(m0, -1, np.int64)
    neu[behalten] = np.arange(m, dtype=np.int64)

    ok_ev = behalten[kompakt]
    grp = neu[kompakt[ok_ev]]
    quelle = idx_gilt[ok_ev]
    # kind="stable": t_on ist aufsteigend sortiert und quelle damit
    # ebenfalls, also bleibt die Zeitordnung INNERHALB eines Knotens
    # erhalten. Fuer den Transformer ist sie belanglos (Attention ist
    # permutationsinvariant), fuer das Nachsehen von Hand nicht.
    ordn = np.argsort(grp, kind="stable")
    reihenfolge = quelle[ordn].astype(np.int64)
    grenzen = np.concatenate(
        [[0], np.cumsum(np.bincount(grp, minlength=m))]).astype(np.int64)
    return Knoten(tabelle, reihenfolge, grenzen)


def stichprobe(knoten: Knoten, t_on, x_on, y_on, *, n_events, seed):
    """n_events Events je Knoten, ohne Zuruecklegen. Reine Funktion.

    Rueckgabe: (ev, maske)
      ev     (m, n_events, 3) float64 -- t, x, y je Token, ROH
      maske  (m, n_events) bool -- True = echtes Event

    FLOAT64 UND NICHT FLOAT32, obwohl das Artefakt float32 ablegt: float32
    hat 24 signifikante Bits und stellt ganze Zahlen exakt bis 2^24 =
    16.777.216 dar. Ein Run ist 8.000.000 µs lang, der Abstand zur Grenze
    betraegt also Faktor 2. Das reicht heute -- die Reserve ist aber
    kleiner als eine Groessenordnung, und ueberschritten wuerde sie STILL:
    jeder Zeitstempel raste auf gerade Vielfache, ohne Meldung. Der
    Aufrufer subtrahiert t_min und castet danach; dann liegen alle Werte
    unter alpha, wo float32 exakt ist. VERSICHERUNG, keine Reparatur.

    Knoten mit weniger als n_events Events fuellen die vorderen Spalten
    und lassen den Rest auf 0 stehen; die Maske traegt die Unterscheidung.
    Beim Transformer wird sie zu src_key_padding_mask -- ohne sie zoegen
    die Fuell-Token echte Aufmerksamkeit, und ein kurzer Knoten bekaeme
    systematisch andere Ausgaben als ein langer.

    ROH und nicht normiert: die Normierung ist JE PAAR (Ursprung auf den
    Zentroid des frueheren Knotens) und kann hier noch nicht stattfinden
    -- ein Knoten steht in vielen Paaren.

    Vektorisiert ueber einen Zufallsschluessel je Event und lexsort: das
    ist eine Ziehung ohne Zuruecklegen je Gruppe, ohne Schleife ueber
    93 000 Knoten. Die Reihenfolge INNERHALB einer Zeile ist damit
    zufaellig, und das ist unschaedlich -- waere sie es nicht, waere das
    Modell falsch gebaut.
    """
    m = len(knoten)
    n_events = int(n_events)
    ev = np.zeros((m, n_events, 3), np.float64)
    maske = np.zeros((m, n_events), bool)
    if m == 0 or n_events == 0:
        return ev, maske

    grenzen, reihenfolge = knoten.grenzen, knoten.reihenfolge
    n = np.diff(grenzen)
    grp = np.repeat(np.arange(m, dtype=np.int64), n)

    rng = np.random.default_rng(seed)
    r = rng.random(len(reihenfolge))
    # grp ist bereits aufsteigend, lexsort mit grp als Hauptschluessel
    # laesst die Bloecke also an Ort und Stelle und mischt nur innerhalb.
    ordn = np.lexsort((r, grp))
    rang = np.arange(len(reihenfolge), dtype=np.int64) - np.repeat(
        grenzen[:-1], n)
    nimm = rang < np.minimum(n, n_events)[grp]

    quelle = reihenfolge[ordn[nimm]]
    zeile, spalte = grp[nimm], rang[nimm]
    ev[zeile, spalte, 0] = np.asarray(t_on, np.float64)[quelle]
    ev[zeile, spalte, 1] = np.asarray(x_on, np.float64)[quelle]
    ev[zeile, spalte, 2] = np.asarray(y_on, np.float64)[quelle]
    maske[zeile, spalte] = True
    return ev, maske