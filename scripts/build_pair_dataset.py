#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer eventpiv.pairs -- nur argparse + Aufruf."""
import argparse

from eventpiv import config, pairs
from eventpiv.graph import NEIGHBORHOOD_MODES

# Run-Listen aus identify_overlap_runs.py (BA_Datensatz_und_Pipeline.md §2.2).
# Hier und nicht im Modul: sie sind Datensatz-Wissen, keine Rechenlogik.
CLEAN_RUNS_TRAIN   = [3, 40, 43, 48, 81]
OVERLAP_RUNS_TRAIN = [2, 11, 19, 30, 32, 35, 44, 59, 61, 76]


def main():
    ap = argparse.ArgumentParser(description="Paar-Datensatz fuer das Kanten-NN")
    ap.add_argument("--split", default="train", choices=config.VALID_SPLITS)
    ap.add_argument("--clean_runs", type=int, nargs="+", default=CLEAN_RUNS_TRAIN)
    ap.add_argument("--overlap_runs", type=int, nargs="+", default=OVERLAP_RUNS_TRAIN)
    ap.add_argument("--alpha", type=int, default=pairs.ALPHA_DEFAULT)
    ap.add_argument("--r_max", type=float, default=pairs.R_MAX_DEFAULT)
    ap.add_argument("--win_dt", type=int, default=pairs.WIN_DT_DEFAULT)
    ap.add_argument("--slab_dt", type=int, default=pairs.SLAB_DT_DEFAULT)
    ap.add_argument("--win_offsets", type=int, nargs="+",
                    default=list(pairs.WIN_OFFSETS_DEFAULT))
    ap.add_argument("--geom", default="ellipsoid", choices=NEIGHBORHOOD_MODES)
    ap.add_argument("--max_yes", type=int, default=pairs.MAX_PER_RUN_DEFAULT)
    ap.add_argument("--max_no", type=int, default=pairs.MAX_PER_RUN_DEFAULT)
    ap.add_argument("--label_dir", default=None)
    ap.add_argument("--out_path", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pairs.build_pair_dataset(
        args.clean_runs, args.overlap_runs, split=args.split,
        alpha=args.alpha, r_max=args.r_max, win_offsets=tuple(args.win_offsets), geom=args.geom,
        win_dt=args.win_dt, slab_dt=args.slab_dt,
        max_yes=args.max_yes, max_no=args.max_no,
        label_dir=args.label_dir, out_path=args.out_path, seed=args.seed)


if __name__ == "__main__":
    main()
