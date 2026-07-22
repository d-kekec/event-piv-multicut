#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_labels.py -- WEGWERF-Werkzeug fuer die Portierung.

Vergleicht die ALTEN Label-Dateien (aus ~/ba-code/nn-edges/archives/labels_*)
feldweise mit den NEU erzeugten ($EVENTPIV_OUT/labels/<split>/).

Zweck: nachweisen, dass die Portierung das Verhalten nicht veraendert hat.
Nach erfolgreichem Durchlauf entbehrlich -- die dauerhaften Tests sind die
Invarianten in tests/test_labels.py (die brauchen keine Referenzdatei).

Vergleich erfolgt WERTBASIERT (Cast auf int64), nicht dtype-genau: die alten
Dateien koennen mit einer frueheren float64-Version erzeugt worden sein.
Der dtype wird separat gemeldet.

Aufruf (WSL-Terminal, Repo-Wurzel):
  python tools/compare_labels.py --split train --runs 3 40 43
  python tools/compare_labels.py --split train --alle
"""
import argparse
from pathlib import Path

import numpy as np

from eventpiv import config

FELDER = ("t", "x", "y", "p", "track_id", "status")


def vergleiche(alt_pfad: Path, neu_pfad: Path) -> bool:
    """Vergleicht eine Datei. Rueckgabe: True wenn alle Felder gleich sind."""
    alt, neu = np.load(alt_pfad), np.load(neu_pfad)
    ok = True

    fehlend = [f for f in FELDER if f not in alt or f not in neu]
    if fehlend:
        print(f"    !! Felder fehlen: {fehlend}")
        ok = False

    for f in FELDER:
        if f in fehlend:
            continue
        a, b = alt[f], neu[f]

        if a.shape != b.shape:
            print(f"    {f:9s} LAENGE verschieden: {a.shape} vs {b.shape}")
            ok = False
            continue

        # Wertbasiert vergleichen: alte Dateien evtl. float64, neue uint32.
        gleich = np.array_equal(a.astype(np.int64), b.astype(np.int64))
        hinweis = "" if a.dtype == b.dtype else f"  (dtype {a.dtype} -> {b.dtype})"

        if gleich:
            print(f"    {f:9s} OK{hinweis}")
        else:
            idx = np.flatnonzero(a.astype(np.int64) != b.astype(np.int64))
            print(f"    {f:9s} ABWEICHUNG bei {len(idx):,} von {a.size:,} "
                  f"Werten{hinweis}")
            for i in idx[:5]:
                print(f"                 [{i}] alt={a[i]} neu={b[i]}")
            ok = False

    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=config.VALID_SPLITS)
    ap.add_argument("--alt_dir", default=None,
                    help="Default: ~/ba-code/nn-edges/archives/labels_<split>")
    ap.add_argument("--runs", type=int, nargs="+")
    ap.add_argument("--alle", action="store_true",
                    help="alle Runs, zu denen eine ALTE Datei existiert")
    args = ap.parse_args()

    alt_dir = (Path(args.alt_dir).expanduser() if args.alt_dir
               else Path.home() / "ba-code" / "nn-edges" /
                    "archives" / f"labels_{args.split}")
    neu_dir = config.label_dir(args.split, create=False)

    print(f"ALT: {alt_dir}")
    print(f"NEU: {neu_dir}\n")
    if not alt_dir.exists():
        raise SystemExit(f"Alter Ordner fehlt: {alt_dir} (--alt_dir setzen)")

    if args.alle:
        runs = sorted(int(p.name[3:].split("_")[0])
                      for p in alt_dir.glob("run*_eventlabels.npz"))
        print(f"--alle: {len(runs)} alte Label-Dateien gefunden.\n")
    elif args.runs:
        runs = args.runs
    else:
        raise SystemExit("Entweder --runs ODER --alle angeben.")

    identisch, abweichend, fehlend = [], [], []
    for r in runs:
        alt_p = alt_dir / f"run{r}_eventlabels.npz"
        neu_p = neu_dir / f"run{r}_eventlabels.npz"
        print(f"run{r}:")
        if not alt_p.exists() or not neu_p.exists():
            fehlt = "ALT" if not alt_p.exists() else "NEU"
            print(f"    -- {fehlt} fehlt, uebersprungen\n")
            fehlend.append(r)
            continue
        (identisch if vergleiche(alt_p, neu_p) else abweichend).append(r)
        print()

    print("=" * 60)
    print(f"identisch : {len(identisch):3d}  {identisch}")
    print(f"abweichend: {len(abweichend):3d}  {abweichend}")
    print(f"fehlend   : {len(fehlend):3d}  {fehlend}")
    if abweichend:
        print("\n-> Abweichungen oben pruefen. Portierung NICHT verhaltensgleich.")
    elif identisch:
        print("\n-> Portierung verhaltensgleich. Weiter mit Schritt [2] (pairs).")


if __name__ == "__main__":
    main()
