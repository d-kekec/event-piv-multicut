#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer eventpiv.overlap -- nur argparse + Aufruf."""
import argparse
import json

from eventpiv import config
from eventpiv.overlap import (BOX_W_DEFAULT, CROSS_THRESH_DEFAULT,
                              analyze_split)


def main():
    ap = argparse.ArgumentParser(
        description="Runs eines Splits nach Partikel-Ueberlappung klassifizieren")
    ap.add_argument("--split", default="val", choices=config.VALID_SPLITS)
    ap.add_argument("--runs", type=int, nargs="+", default=None,
                    help="Default: alle Runs des Splits")
    ap.add_argument("--box_w", type=float, default=BOX_W_DEFAULT)
    ap.add_argument("--cross_thresh", type=float, default=CROSS_THRESH_DEFAULT)
    ap.add_argument("--json_out", default=None,
                    help="Kennzahlen zusaetzlich als JSON ablegen")
    args = ap.parse_args()

    results, cats = analyze_split(args.split, args.runs, box_w=args.box_w,
                                  cross_thresh=args.cross_thresh)
    if args.json_out:
        from pathlib import Path
        Path(args.json_out).write_text(json.dumps(
            {"meta": json.loads(config.meta_json(
                stage="overlap", split=args.split, box_w=args.box_w,
                cross_thresh=args.cross_thresh)),
             "metrics": {str(k): v for k, v in results.items()},
             "categories": cats}, indent=2))
        print(f"-> {args.json_out}")


if __name__ == "__main__":
    main()
