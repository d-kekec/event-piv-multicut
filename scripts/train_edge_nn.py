#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer eventpiv.train -- nur argparse + Aufruf."""
import argparse

from eventpiv.model import FEATURE_MODES
from eventpiv.train import IMBALANCE_MODES, train_edge_nn


def main():
    ap = argparse.ArgumentParser(description="Kantenkosten-NN trainieren")
    ap.add_argument("--feature_mode", default="dr", choices=FEATURE_MODES)
    ap.add_argument("--imbalance", default="sample", choices=IMBALANCE_MODES)
    ap.add_argument("--sample_ratio", type=float, default=3.0)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_path", default=None,
                    help="Default: $EVENTPIV_OUT/pairs/pairs_dataset.npz")
    ap.add_argument("--out_dir", default=None,
                    help="Default: $EVENTPIV_OUT/models")
    args = ap.parse_args()

    train_edge_nn(args.data_path, feature_mode=args.feature_mode,
                  imbalance=args.imbalance, sample_ratio=args.sample_ratio,
                  hidden=args.hidden, epochs=args.epochs, batch=args.batch,
                  lr=args.lr, val_frac=args.val_frac, seed=args.seed,
                  out_dir=args.out_dir)


if __name__ == "__main__":
    main()
