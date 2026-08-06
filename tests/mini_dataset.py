#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetischer Mini-Datensatz -- schreibt .dat und _bbox.npy wie der echte.

Zweck: der Integrationstest ueber [1] bis [4] braucht einen echten
Schreib- UND Lesepfad. Ein Fixture, das nur Arrays liefert, umgeht
data.load_events -- und genau dort hat ein Doku-Commit am 04.08. eine
return-Zeile entfernt, ohne dass pytest es bemerkte.

Warum der Schreibweg HIER liegt: psee_io.dat_events_tools.write_header ist
defekt (referenziert EV_STRINGS statt EV_STRING, oeffnet im Textmodus bei
binaerem Inhalt). psee_io ist Fremdcode und bleibt unangetastet.

DAT-FORMAT, gelesen aus psee_io/dat_events_tools.py:
  Kopf      Zeilen "% ...", danach zwei Bytes ev_type=0 und ev_size=8
  Event     4 B Zeitstempel u4 + 4 B Nutzlast i4
  Nutzlast  Bit 0-13 x, Bit 14-27 y, Bit 28 Polaritaet

DIE SZENE bildet nach, was am Datensatz gemessen ist (Datensatz.md §3):
zwei Sicheln je Partikel (ON in Bewegungsrichtung davor, OFF gleich weit
dahinter, |D| = 6 px), Geschwindigkeit im gemessenen Bereich, rund 5 %
gleichverteiltes Rauschen. Die Sichel ist als Gaussverteilung GENAEHERT --
geprueft werden Trennbarkeit und Polaritaetsversatz, nicht die Bogenform.

VIER RUNS, nicht einer: train.split_runs_stratified braucht mindestens zwei
Runs je Klasse, sonst verliert der Holdout eine Klasse und AUC/TPR werden
NaN. run1/run2 kreuzen (NEIN-Quelle), run3/run4 sind sauber (JA-Quelle).

DIE KREUZUNG ist echt: die Relativgeschwindigkeit betraegt 1,0e-4 px/µs in
y, aus 50 px Abstand bei t=0 wird 0 bei t=500 ms und wieder 50 px bei
t=1 s. Die Szene durchlaeuft damit BEIDE Regime. Der echte Datensatz tut
das in keinem untersuchten Run -- run3 kreuzt nie (19,2 px Mindestabstand),
run32 ist dauerhaft verschmolzen.

Die Laufdauer ist mit 1 s bewusst kuerzer als die 8 s des echten
Datensatzes. Folge: die Vorgabewerte von PairSampling.win_offsets
(1,5 / 3,5 / 5,5 s) liegen ausserhalb, und wer hier Paare sammelt, muss sie
setzen. Das entfaellt, sobald Stufe 1 den ganzen Run abschreitet.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

# Dateiformat des echten Datensatzes (Datensatz.md §2).
BBOX_DTYPE = np.dtype([("t", "<i8"), ("x", "<f4"), ("y", "<f4"),
                       ("w", "<f4"), ("h", "<f4"), ("class_id", "<u4"),
                       ("track_id", "<u4"), ("class_confidence", "<f4")])

# Endianness explizit: das Format ist Little-Endian, nicht "was die
# Maschine gerade macht".
_EV_DTYPE = np.dtype([("t", "<u4"), ("_", "<i4")])

BOX_W = 11.0                    # px, Konstante des Datensatzes
RASTER_DT = 1e6 / 60.0          # µs, das 60-Hz-Boxraster

# Sichelgeometrie. VERSATZ_ON=3 ergibt |D| = 6 px zwischen den beiden
# Zentroiden -- gemessen 6,07 px auf run3.
VERSATZ_ON = 3.0
SIG_LAENGS = 1.2                # laengs der Bewegung duenn
SIG_QUER   = 2.0                # quer dazu ausgedehnt
R_CLIP     = 7.0                # px, unter r_assign = 8: Signalevents
                                # duerfen nicht zufaellig zu RAUSCH werden

BREITE, HOEHE = 1280, 720
T_ENDE = 1_000_000              # µs
RATE_PRO_US = 6e-3              # Events je µs und Bahn -> 6000 je Bahn
RAUSCH_ANTEIL = 0.05


class Bahn(NamedTuple):
    """Geradlinige Partikelbahn. vx/vy in px/µs.

    |v| = 7,8e-5 px/µs = 0,078 px/ms liegt im gemessenen Bereich
    (0,025 bis 0,102 px/ms). Schneller waere schaedlich: ab etwa
    0,1 px/ms uebersteigt der Rasterfehler der 60-Hz-Naeherung (bis
    8,33 ms) die Schwelle r_ambig = 2 px, und die Fixture erzeugte
    Mehrdeutigkeit aus der Naeherung statt aus der Kreuzung.
    """
    track_id: int
    x0: float
    y0: float
    vx: float
    vy: float


def _raster(t_ende: int) -> np.ndarray:
    """60-Hz-Zeitraster. Die Abstaende wechseln 16667/16666 wie im echten."""
    n = int(t_ende // RASTER_DT) + 1
    return np.rint(np.arange(n) * RASTER_DT).astype(np.int64)


def bbox_array(bahnen, t_ende: int = T_ENDE) -> np.ndarray:
    """Boxen je Rasterpunkt und Bahn. (x, y) ist die ECKE oben links."""
    rt = _raster(t_ende)
    b = np.zeros(len(rt) * len(bahnen), dtype=BBOX_DTYPE)
    k = 0
    for t in rt:
        for bahn in bahnen:
            b[k] = (t, bahn.x0 + bahn.vx * t - BOX_W / 2.0,
                    bahn.y0 + bahn.vy * t - BOX_W / 2.0,
                    BOX_W, BOX_W, 0, bahn.track_id, 1.0)
            k += 1
    return b


def _events_bahn(rng, bahn: Bahn, t_ende: int, n: int):
    """n Events um eine Bahn, als zwei Sicheln je Polaritaet."""
    t = np.sort(rng.integers(0, t_ende, n))
    cx = bahn.x0 + bahn.vx * t
    cy = bahn.y0 + bahn.vy * t

    v = float(np.hypot(bahn.vx, bahn.vy))
    ux, uy = (bahn.vx / v, bahn.vy / v) if v > 0 else (1.0, 0.0)

    p = rng.integers(0, 2, n)          # 1 = ON (vorne), 0 = OFF (hinten)
    a = np.where(p == 1, VERSATZ_ON, -VERSATZ_ON) + rng.normal(0.0, SIG_LAENGS, n)
    q = rng.normal(0.0, SIG_QUER, n)
    # Radial klemmen statt verwerfen: eine Ablehnungsstichprobe machte die
    # Eventzahl vom Zufall abhaengig, und die Fixture waere nicht mehr
    # deterministisch in ihrer GROESSE.
    s = np.minimum(1.0, R_CLIP / np.maximum(np.hypot(a, q), 1e-9))
    a, q = a * s, q * s
    return t, cx + a * ux - q * uy, cy + a * uy + q * ux, p


def _rauschen(rng, t_ende: int, n: int):
    """Gleichverteilt ueber die Sensorflaeche -- wie auf run3 gemessen."""
    t = np.sort(rng.integers(0, t_ende, n))
    return (t, rng.uniform(0, BREITE, n), rng.uniform(0, HOEHE, n),
            rng.integers(0, 2, n))


def events_array(bahnen, seed: int, t_ende: int = T_ENDE):
    """Alle Events eines Runs, nach t sortiert. Rueckgabe: (t, x, y, p)."""
    rng = np.random.default_rng(seed)
    n_bahn = int(RATE_PRO_US * t_ende)
    teile = [_events_bahn(rng, b, t_ende, n_bahn) for b in bahnen]
    # RAUSCH_ANTEIL ist der Anteil AM GANZEN Strom, nicht am Signal.
    n_rausch = int(round(RAUSCH_ANTEIL / (1.0 - RAUSCH_ANTEIL)
                         * n_bahn * len(bahnen)))
    teile.append(_rauschen(rng, t_ende, n_rausch))

    t = np.concatenate([s[0] for s in teile])
    x = np.concatenate([s[1] for s in teile])
    y = np.concatenate([s[2] for s in teile])
    p = np.concatenate([s[3] for s in teile])
    o = np.argsort(t, kind="stable")   # collect_pairs verlangt sortiertes t
    return t[o], x[o], y[o], p[o]


def schreibe_dat(pfad, t, x, y, p) -> None:
    """Schreibt einen Eventstrom im Prophesee-DAT-Format.

    x und y werden gerundet und geklemmt: der echte Datensatz enthaelt
    ganzzahlige Pixel, und genau daher ist dr quantisiert -- eine Fixture
    mit Subpixelkoordinaten haette einen anderen Merkmalsraum.

    Kopf und Daten in EINEM write: np.ndarray.tofile auf ein bereits
    beschriebenes Dateiobjekt mischt zwei Pufferungen.
    """
    xi = np.clip(np.rint(x), 0, BREITE - 1).astype(np.int64)
    yi = np.clip(np.rint(y), 0, HOEHE - 1).astype(np.int64)
    ev = np.empty(len(t), dtype=_EV_DTYPE)
    ev["t"] = np.asarray(t, dtype=np.uint32)
    ev["_"] = (xi | (yi << 14) | (np.asarray(p, np.int64) << 28)).astype(np.int32)

    kopf = (b"% Data file containing Event2D events.\n"
            b"% Version 2\n"
            b"% Date 2026-08-06 00:00:00\n"
            + f"% Height {HOEHE}\n% Width {BREITE}\n".encode()
            + bytes([0, 8]))          # ev_type = 0, ev_size = 8
    Path(pfad).write_bytes(kopf + ev.tobytes())


def bahnen_kreuzung(x_m, y_m, vx=6e-5, vy=5e-5, ids=(1, 2, 3)):
    """Zwei Partikel treffen sich bei (x_m, y_m) zur Fenstermitte, plus eines
    weit abseits als Kontrolle im selben Run."""
    tm = T_ENDE / 2.0
    return [Bahn(ids[0], x_m - vx * tm, y_m - vy * tm, vx,  vy),
            Bahn(ids[1], x_m - vx * tm, y_m + vy * tm, vx, -vy),
            Bahn(ids[2], 900.0, 620.0, 3e-5, 1e-5)]


def bahnen_sauber(dx=0.0, ids=(1, 2, 3)):
    """Drei Partikel, ueber den ganzen Lauf mehr als 250 px auseinander --
    weit jenseits des Beruehrungsabstands der Wolken von ~19 px."""
    return [Bahn(ids[0], 200.0 + dx, 150.0, 6e-5,  5e-5),
            Bahn(ids[1], 620.0 + dx, 150.0, 5e-5,  6e-5),
            Bahn(ids[2], 400.0 + dx, 520.0, 4e-5, -2e-5)]


# Der Seed ist die Run-Nummer: zwei Runs derselben Art unterscheiden sich
# damit in der Realisierung, nicht nur in der Geometrie.
SZENEN = {
    1: ("kreuzung", lambda: bahnen_kreuzung(400.0, 300.0)),
    2: ("kreuzung", lambda: bahnen_kreuzung(700.0, 420.0, vx=4e-5, vy=6e-5)),
    3: ("sauber",   lambda: bahnen_sauber(0.0)),
    4: ("sauber",   lambda: bahnen_sauber(60.0)),
}


def baue_mini_datensatz(data_root, split: str = "train") -> dict:
    """Legt <data_root>/<split>/run{1..4}.dat und _bbox.npy an.

    Rueckgabe: {"kreuzung": [...], "sauber": [...]} -- die Run-Listen, die
    PairSampling als overlap_runs bzw. clean_runs braucht.
    """
    d = Path(data_root) / split
    d.mkdir(parents=True, exist_ok=True)
    aus = {"kreuzung": [], "sauber": []}
    for run_nr, (art, mach_bahnen) in SZENEN.items():
        bahnen = mach_bahnen()
        t, x, y, p = events_array(bahnen, seed=run_nr)
        schreibe_dat(d / f"run{run_nr}.dat", t, x, y, p)
        np.save(d / f"run{run_nr}_bbox.npy", bbox_array(bahnen))
        aus[art].append(run_nr)
    return aus