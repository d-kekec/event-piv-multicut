#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer visualize.animate_run -- nur argparse + Aufruf.

Animiert einen ganzen Run (oder ein Teilstueck), Farbe = Polaritaet.
Zweck: visuell finden, WANN Ueberlappungen und Kreuzungen passieren.

Beispiele (WSL-Terminal, Repo-Wurzel):
  python scripts/animate_run.py --list --split train
  python scripts/animate_run.py --split train --run_nr 61
  python scripts/animate_run.py --split val --run_nr 4 --t0 1500000 --dt 500000 \\
      --frame_ms 5
"""
import argparse

from eventpiv import config
from eventpiv.visualize import animate_run


def main():
    ap = argparse.ArgumentParser(description="Ganzen Run animieren")
    ap.add_argument("--split", default="train", choices=config.VALID_SPLITS)
    ap.add_argument("--run_nr", type=int, default=None)
    ap.add_argument("--list", action="store_true",
                    help="nur die verfuegbaren Runs des Splits auflisten")
    ap.add_argument("--t0", type=int, default=None,
                    help="Startzeit µs (Default: Anfang des Runs)")
    ap.add_argument("--dt", type=int, default=None,
                    help="Dauer µs (Default: ganzer Run -- viel RAM!)")
    ap.add_argument("--frame_ms", type=float, default=20.0,
                    help="Akkumulationszeit je Frame")
    ap.add_argument("--speedup", type=float, default=1.0,
                    help=">1 ueberspringt Frames (kuerzeres GIF)")
    ap.add_argument("--downscale", type=int, default=2,
                    help="Bildverkleinerung (2 -> 640x360)")
    ap.add_argument("--point", type=int, default=1,
                    help="Punktgroesse in Pixeln")
    ap.add_argument("--out_dir", default=None,
                    help="Default: $EVENTPIV_OUT/results/animations")
    args = ap.parse_args()

    if args.list:
        for s in config.VALID_SPLITS:
            if args.split and s != args.split:
                continue
            runs = config.available_runs(s)
            print(f"{s}: {len(runs)} Runs -> {runs}")
        return

    if args.run_nr is None:
        ap.error("--run_nr angeben (oder --list zum Auflisten).")

    animate_run(args.split, args.run_nr, t0=args.t0, dt=args.dt,
                frame_ms=args.frame_ms, speedup=args.speedup,
                downscale=args.downscale, point=args.point,
                out_dir=args.out_dir)


if __name__ == "__main__":
    main()
