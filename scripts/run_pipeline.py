#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer pipeline.run_pipeline -- nur argparse + Aufruf.

Beispiel (Arbeitspunkt, Labels bereits vorhanden):
  python scripts/run_pipeline.py --alpha 3000 --r_max 5 \\
      --steps dataset train evaluate --val_crossing 4 --val_clean 7 9 12 13 21
"""
import argparse

from eventpiv.model import FEATURE_MODES
from eventpiv.pipeline import DT_DEFAULT, T0_DEFAULT, run_pipeline
from eventpiv.train import IMBALANCE_MODES

CLEAN_RUNS_TRAIN   = [3, 40, 43, 48, 81]
OVERLAP_RUNS_TRAIN = [2, 11, 19, 30, 32, 35, 44, 59, 61, 76]
VAL_CROSSING = [4]
VAL_CLEAN    = [7, 9, 12, 13, 21]

ALLE_SCHRITTE = ("labels", "dataset", "train", "evaluate")


def main():
    ap = argparse.ArgumentParser(description="Komplette NN-Kette")
    ap.add_argument("--alpha", type=int, required=True)
    ap.add_argument("--r_max", type=float, required=True)
    ap.add_argument("--win_dt", type=int, default=None,
                    help="Default: aus alpha/r_max berechnet (pairs.win_dt_for)")
    ap.add_argument("--feature_mode", default="dr", choices=FEATURE_MODES)
    ap.add_argument("--imbalance", default="sample", choices=IMBALANCE_MODES)
    ap.add_argument("--clean_runs", type=int, nargs="+", default=CLEAN_RUNS_TRAIN)
    ap.add_argument("--overlap_runs", type=int, nargs="+",
                    default=OVERLAP_RUNS_TRAIN)
    ap.add_argument("--val_crossing", type=int, nargs="+", default=VAL_CROSSING)
    ap.add_argument("--val_clean", type=int, nargs="+", default=VAL_CLEAN)
    ap.add_argument("--solve_dt", type=int, default=DT_DEFAULT)
    ap.add_argument("--t0", type=int, default=T0_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="PD", choices=("P", "PD", "PD+"),
                    help="RAMA-Solvermodus ('P' = ohne Dual-Phase)")
    ap.add_argument("--steps", nargs="+", default=list(ALLE_SCHRITTE),
                    choices=ALLE_SCHRITTE)
    args = ap.parse_args()

    run_pipeline(alpha=args.alpha, r_max=args.r_max, win_dt=args.win_dt,
                 clean_runs=args.clean_runs, overlap_runs=args.overlap_runs,
                 val_crossing=args.val_crossing, val_clean=args.val_clean,
                 feature_mode=args.feature_mode, imbalance=args.imbalance,
                 solve_dt=args.solve_dt, t0=args.t0, seed=args.seed, mode=args.mode,
                 steps=tuple(args.steps))


if __name__ == "__main__":
    main()
