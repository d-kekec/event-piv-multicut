#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bewertet vorhandene Fit-Dateien aus [7] mit den Kennzahlen von STELLA.

Kennzahlen nach STELLA (Sachs et al., 2026), Abschnitt 4.1, je Run und
ueber alle Runs gepoolt: MedAE in Ort und Geschwindigkeit mit P25 und
P75, MedAE_u relativ zu |U^| (U_NORM_STELLA), t_covered und
N_tracks/N_gt, dazu der Anteil verworfener Bahnen. Definition und
Auslegung stehen in bewerten/evaluate_trajektorien.py (stella_fehler).

Daneben zwei Diagnosen: der Median der Boxbreite und der Laengsfehler
mit Vorzeichen, je Run. Liegt der Ort eines Runs systematisch voraus
oder zurueck, passt die Positionskorrektur aus [7] nicht zu seinen
Partikeln.

Dazu die MedAE in zwei weiteren Lesarten, je Achse gepoolt und im Betrag
von u (stella_fehler). Die Zahl der Arbeit bleibt die Norm.

Liest fit_<split>_run<N>_<stamm>.npz aus dem solves-Verzeichnis und die
Boxen ueber EVENTPIV_DATA; [7] laeuft dafuer nicht neu. Der Stamm folgt
ohne Angabe aus den Vorgaben von Solver, Solver2 und Fit, ein anderer
ueber --stamm.

Aufruf ohne Hydra; EVENTPIV_DATA und EVENTPIV_OUT wie in scripts/_cli.py.
<solves> ist das Verzeichnis, dessen Dateien fit_trajektorien hinter "->"
meldet; ein Run ohne Fit-Datei wird uebersprungen.
    python tools/stella_bewertung.py <solves> val <runs>
    python tools/stella_bewertung.py <solves> val <runs> --json <ziel>.json
    python tools/stella_bewertung.py <solves> val <runs> --stamm <stamm>
"""
import argparse
import json
from pathlib import Path

import numpy as np

from eventpiv.bewerten.evaluate_trajektorien import (stella_fehler,
                                                     stella_kennzahlen)
from eventpiv.kern.data import load_bbox
from eventpiv.kern.runconfig import Fit, Solver, Solver2, slug

SPALTEN = ("t", "x", "y", "ux", "uy", "trajektorie_id")


def zeile(name, k):
    """Eine Ausgabezeile; Quartile in eckigen Klammern."""
    return (f"{name}: MedAE_xy {k['medae_xy']:.3f} px "
            f"[{k['p25_xy']:.3f}, {k['p75_xy']:.3f}]  "
            f"MedAE_u {k['medae_u']:.4f} px/ms "
            f"[{k['p25_u']:.4f}, {k['p75_u']:.4f}] "
            f"= {k['medae_u_rel'] * 100:.2f} % von |U^|  |  "
            f"t_covered {k['t_covered'] * 100:.1f} %  "
            f"N {k['n_tracks']}/{k['n_gt']} = {k['n_tracks_je_gt']:.3f}  "
            f"verworfen {k['verworfen_frac'] * 100:.2f} %  "
            f"zugeordnet {k['zugeordnet_frac'] * 100:.1f} %  |  "
            f"laengs {k['laengs_median']:+.3f} px  |  "
            f"je Achse {k['medae_xy_achse']:.3f} px, "
            f"{k['medae_u_achse_rel'] * 100:.2f} %  "
            f"Betrag {k['medae_u_betrag_rel'] * 100:.2f} %"
            + (f"  Box {k['box_w_median']:.2f} px"
               if "box_w_median" in k else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("solves", type=Path,
                    help="Verzeichnis der Fit-Dateien, aus dem Fit-Log")
    ap.add_argument("split")
    ap.add_argument("runs", type=int, nargs="+")
    ap.add_argument("--stamm",
                    default=f"{Solver().short()}_{Solver2().short()}_"
                            f"{slug(Fit())}")
    ap.add_argument("--json", type=Path, help="Ergebnis zusaetzlich als JSON")
    a = ap.parse_args()

    fehler, je_run = [], {}
    for r in a.runs:
        pfad = a.solves / f"fit_{a.split}_run{r}_{a.stamm}.npz"
        if not pfad.exists():
            print(f"run{r}: {pfad.name} fehlt, uebersprungen", flush=True)
            continue
        with np.load(pfad) as d:
            spalten = tuple(d[s] for s in SPALTEN)
        bb = load_bbox(a.split, r)
        f = stella_fehler(spalten, bb)
        k = stella_kennzahlen(f)
        k["box_w_median"] = float(np.median(bb["w"]))
        je_run[r] = k
        fehler.append(f)
        print(zeile(f"run{r}", k), flush=True)
    if not fehler:
        raise SystemExit(f"keine Fit-Datei mit Stamm {a.stamm}")

    gesamt = stella_kennzahlen(fehler)
    print(zeile(f"{a.split} gepoolt, {len(fehler)} Runs", gesamt), flush=True)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(
            {"solves": str(a.solves), "split": a.split, "stamm": a.stamm,
             "je_run": je_run, "gesamt": gesamt}, indent=1))
        print(f"-> {a.json}", flush=True)


if __name__ == "__main__":
    main()