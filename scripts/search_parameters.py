#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer pipeline.parameter_sweep -- nur argparse + Aufruf.

Run-Listen als Defaults aus identify_overlap_runs.py
(BA_Datensatz_und_Pipeline.md §2.2). Sie gelten fuer die dort genannten
Splits -- bei anderem Split eigene Listen uebergeben.
"""
import argparse

from eventpiv.model import FEATURE_MODES
from eventpiv.pipeline import DT_DEFAULT, parameter_sweep
from eventpiv.train import IMBALANCE_MODES

CLEAN_RUNS_TRAIN   = [3, 40, 43, 48, 81]
OVERLAP_RUNS_TRAIN = [2, 11, 19, 30, 32, 35, 44, 59, 61, 76]
VAL_OVERLAP = [3, 4, 6, 20]         # MAX_OVERLAP im val-Split
VAL_CLEAN   = [7, 9, 12, 13, 21]    # MIN_OVERLAP im val-Split


def main():
    ap = argparse.ArgumentParser(description="(alpha, r_max)-Parameterstudie")
    ap.add_argument("--alphas", type=int, nargs="+", default=[3000, 4000, 5000])
    ap.add_argument("--r_maxs", type=float, nargs="+", default=[5.0, 6.0])
    ap.add_argument("--feature_mode", default="dr", choices=FEATURE_MODES)
    ap.add_argument("--imbalance", default="sample", choices=IMBALANCE_MODES)
    ap.add_argument("--clean_runs", type=int, nargs="+", default=CLEAN_RUNS_TRAIN)
    ap.add_argument("--overlap_runs", type=int, nargs="+",
                    default=OVERLAP_RUNS_TRAIN)
    ap.add_argument("--val_overlap", type=int, nargs="+", default=VAL_OVERLAP)
    ap.add_argument("--val_clean", type=int, nargs="+", default=VAL_CLEAN)
    ap.add_argument("--solve_dt", type=int, default=DT_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="PD", choices=("P", "PD", "PD+"),
                    help="RAMA-Solvermodus ('P' = ohne Dual-Phase)")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    parameter_sweep(args.alphas, args.r_maxs, args.clean_runs,
                    args.overlap_runs, val_overlap=args.val_overlap,
                    val_clean=args.val_clean, feature_mode=args.feature_mode,
                    imbalance=args.imbalance, solve_dt=args.solve_dt,
                    seed=args.seed, mode=args.mode, out_json=args.out_json)


if __name__ == "__main__":
    main()
