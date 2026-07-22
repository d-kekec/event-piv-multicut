#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer eventpiv.solve -- nur argparse + Aufruf."""
import argparse

from eventpiv import config
from eventpiv.solve import solve_and_evaluate


def main():
    ap = argparse.ArgumentParser(description="NN-Kosten -> RAMA -> GT-Abgleich")
    ap.add_argument("--run_nr", type=int, required=True)
    ap.add_argument("--split", default="val", choices=config.VALID_SPLITS)
    ap.add_argument("--model_tag", default="dr_sample")
    ap.add_argument("--alpha", type=int, required=True)
    ap.add_argument("--r_max", type=float, required=True)
    ap.add_argument("--t0", type=int, default=1_700_000)
    ap.add_argument("--dt", type=int, default=5_000)
    ap.add_argument("--use_knn", action="store_true")
    ap.add_argument("--k_max", type=int, default=24)
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--label_dir", default=None)
    ap.add_argument("--no_strict", action="store_true",
                    help="Konsistenzpruefung alpha/r_max nur warnen")
    args = ap.parse_args()

    solve_and_evaluate(args.split, args.run_nr, args.model_tag,
                       alpha=args.alpha, r_max=args.r_max, t0=args.t0,
                       dt=args.dt, use_knn=args.use_knn, k_max=args.k_max,
                       model_dir=args.model_dir, label_dir=args.label_dir,
                       strict=not args.no_strict)


if __name__ == "__main__":
    main()
