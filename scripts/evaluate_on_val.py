#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer pipeline.evaluate_on_val -- nur argparse + Aufruf."""
import argparse

from eventpiv import config
from eventpiv.pipeline import DT_DEFAULT, T0_DEFAULT, evaluate_on_val


def main():
    ap = argparse.ArgumentParser(
        description="Ein Netz ueber mehrere Runs bewerten (CROSSING + CLEAN)")
    ap.add_argument("--model_tag", default="dr_sample")
    ap.add_argument("--alpha", type=int, required=True)
    ap.add_argument("--r_max", type=float, required=True)
    ap.add_argument("--split", default="val", choices=config.VALID_SPLITS)
    ap.add_argument("--crossing", type=int, nargs="+", default=[],
                    help="Runs mit Kreuzungen/Overlap (Schnitt-Test)")
    ap.add_argument("--clean", type=int, nargs="+", default=[],
                    help="saubere Runs (Kontrolle)")
    ap.add_argument("--t0", type=int, default=T0_DEFAULT)
    ap.add_argument("--dt", type=int, default=DT_DEFAULT)
    ap.add_argument("--in_process", action="store_true",
                    help="ohne Subprozess-Isolation (schneller, aber ein "
                         "VRAM-Absturz beendet die ganze Bewertung)")
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--label_dir", default=None)
    ap.add_argument("--no_strict", action="store_true")
    args = ap.parse_args()

    evaluate_on_val(args.model_tag, alpha=args.alpha, r_max=args.r_max,
                    crossing=args.crossing, clean=args.clean,
                    split=args.split, t0=args.t0, dt=args.dt,
                    isolated=not args.in_process, model_dir=args.model_dir,
                    label_dir=args.label_dir, strict=not args.no_strict)


if __name__ == "__main__":
    main()
