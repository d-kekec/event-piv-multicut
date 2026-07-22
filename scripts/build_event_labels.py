#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer eventpiv.labels -- nur argparse + Aufruf."""
import argparse

from eventpiv import config
from eventpiv.labels import R_AMBIG_DEFAULT, R_ASSIGN_DEFAULT, build_labels


def main():
    ap = argparse.ArgumentParser(description="Event -> track_id + status")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--runs", type=int, nargs="+", help="explizite Run-Liste")
    g.add_argument("--all_split", action="store_true",
                   help="alle Runs des Splits")
    ap.add_argument("--split", default="train", choices=config.VALID_SPLITS)
    ap.add_argument("--out_dir", default=None,
                    help="Default: $EVENTPIV_OUT/labels/<split>")
    ap.add_argument("--t0", type=int, default=None, help="Startzeit µs")
    ap.add_argument("--dt", type=int, default=None, help="Fensterlaenge µs")
    ap.add_argument("--r_assign", type=float, default=R_ASSIGN_DEFAULT)
    ap.add_argument("--r_ambig", type=float, default=R_AMBIG_DEFAULT)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    runs = config.available_runs(args.split) if args.all_split else args.runs
    if args.all_split:
        print(f"--all_split: {len(runs)} Runs in '{args.split}' gefunden.")

    build_labels(runs, split=args.split, out_dir=args.out_dir,
                 t0=args.t0, dt=args.dt, r_assign=args.r_assign,
                 r_ambig=args.r_ambig, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
