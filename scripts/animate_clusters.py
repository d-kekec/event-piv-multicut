#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-Huelle fuer visualize.animate_clusters -- nur argparse + Aufruf.

Loest ein Zeitfenster und animiert Rohevents, RAMA-Cluster und Ground Truth
nebeneinander. Zweck: Reinheit und Fragmentierung SEHEN statt nur als Zahl
lesen.

Beispiele (WSL-Terminal, Repo-Wurzel):
  # Standard: Trail an, Farbe nach Cluster-ID (zeigt Fragmentierung)
  python scripts/animate_clusters.py --split val --run_nr 7 \\
      --alpha 10000 --r_max 7.06 --model_tag dr_sample

  # Farbe nach GT-Mehrheit: gleiche Farbe = gleiches Partikel in beiden Panels
  python scripts/animate_clusters.py --split val --run_nr 4 \\
      --alpha 10000 --r_max 7.06 --color_by gt

  # Momentaufnahmen statt aufwachsender Spuren
  python scripts/animate_clusters.py --split train --run_nr 32 \\
      --alpha 10000 --r_max 7.06 --no_trail --frame_ms 0.2
"""
import argparse

from eventpiv import config
from eventpiv.visualize import animate_clusters


def main():
    ap = argparse.ArgumentParser(
        description="Geloestes Fenster animieren: Roh | Cluster | GT")
    ap.add_argument("--run_nr", type=int, required=True)
    ap.add_argument("--split", default="val", choices=config.VALID_SPLITS)
    ap.add_argument("--model_tag", default="dr_sample")
    ap.add_argument("--alpha", type=int, required=True)
    ap.add_argument("--r_max", type=float, required=True)
    ap.add_argument("--t0", type=int, default=1_700_000)
    ap.add_argument("--dt", type=int, default=5_000)
    ap.add_argument("--frame_ms", type=float, default=0.25,
                    help="Zeitschritt je Frame (Fenster sind klein!)")
    ap.add_argument("--no_trail", action="store_true",
                    help="nur die Events des aktuellen Frames zeigen, "
                         "statt die Spuren aufwachsen zu lassen")
    ap.add_argument("--color_by", default="cluster", choices=("cluster", "gt"),
                    help="'cluster' zeigt Fragmentierung, 'gt' macht die "
                         "Panels direkt vergleichbar (s. visualize.py)")
    ap.add_argument("--min_size", type=int, default=5,
                    help="Cluster darunter werden grau (Rauschen)")
    ap.add_argument("--downscale", type=int, default=1)
    ap.add_argument("--point", type=int, default=2,
                    help="Punktgroesse -- bei duennen Fenstern hilfreich")
    ap.add_argument("--no_crop", action="store_true",
                    help="ganzes Sensorbild statt Zuschnitt auf die Events")
    ap.add_argument("--no_gt", action="store_true",
                    help="GT-Panel weglassen")
    ap.add_argument("--use_knn", action="store_true")
    ap.add_argument("--k_max", type=int, default=24)
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--label_dir", default=None)
    ap.add_argument("--no_strict", action="store_true",
                    help="Konsistenzpruefung alpha/r_max nur warnen")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    animate_clusters(args.split, args.run_nr, args.model_tag,
                     alpha=args.alpha, r_max=args.r_max, t0=args.t0,
                     dt=args.dt, frame_ms=args.frame_ms,
                     trail=not args.no_trail, downscale=args.downscale,
                     point=args.point, min_size=args.min_size,
                     color_by=args.color_by, show_gt=not args.no_gt,
                     crop=not args.no_crop, use_knn=args.use_knn,
                     k_max=args.k_max, model_dir=args.model_dir,
                     label_dir=args.label_dir, strict=not args.no_strict,
                     out_dir=args.out_dir)


if __name__ == "__main__":
    main()
