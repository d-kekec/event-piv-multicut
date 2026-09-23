#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tracklet_groessen.py -- WEGWERFWERKZEUG, nicht in git.

Groessenverteilung der Tracklets aus Artefakt (A), fuer die Wahl von
kandidaten.min_events und kandidaten.n_events.

Liest (A) und nicht (B): n_on je Tracklet ist ein bincount ueber
tracklet_id, also genau die Groesse, auf die knoten_bauen filtert -- eine
Ableitung statt zweier. Damit entfaellt die Abhaengigkeit auf [5b], das
fuer den aktuellen Arbeitspunkt gar nicht gebaut ist.

PREIS: n_gt fehlt, also die Kreuztabelle klein x Rauschen. Die braucht die
Label-Datei und damit einen vollen Lauf ueber den Eventstrom je Run;
knoten_rausch_frac liefert [6a] ohnehin je Run.

Aufruf (rema, Repo-Wurzel, venv aktiv):
    python tools/throwaway/tracklet_groessen.py
    python tools/throwaway/tracklet_groessen.py --filter 15 --json ~/tmp/groessen.json
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np

from eventpiv.kern import config
from eventpiv.kern.runconfig import (Neighborhood, PairSampling, Selection,
                                     Solver, Training)

QUANTILE = (5, 25, 50, 75, 95)


def dateien(split: str, nb, ps, tr, sv):
    """Alle vorhandenen (A)-Dateien eines Splits, nach Run-Nummer sortiert.

    Gesucht statt geraten: welche Runs gebaut wurden, weiss das
    Verzeichnis, nicht die Konfiguration. glob("*.npz") greift nur eine
    Ebene tief -- features/ und stufe2/ liegen darunter und stoeren nicht.
    """
    d = config.tracklet_file(nb, ps, tr, Selection(split=split, run_nr=0),
                             sv, create=False).parent
    if not d.is_dir():
        raise SystemExit(f"{d} fehlt -- erst scripts/build_tracklets.py laufen "
                         "lassen (gleiche Gruppen).")
    muster = re.compile(rf"^{re.escape(split)}_run(\d+)_"
                        rf"{re.escape(sv.short())}\.npz$")
    treffer = [(int(m.group(1)), p) for p in sorted(d.glob("*.npz"))
               if (m := muster.match(p.name))]
    if not treffer:
        raise SystemExit(f"Keine Dateien in {d}, die auf {split}/"
                         f"{sv.short()} passen.")
    return sorted(treffer), d


def groessen(tracklet_id) -> np.ndarray:
    """Eventzahl je Tracklet aus (A).

    tracklet_id hat einen Eintrag je ON-Event; -1 heisst kein Tracklet.
    Die IDs sind global je Run und LUECKENHAFT -- RAMA nummeriert je
    Fenster ab 0, der Treiber addiert max()+1, und verschenkte Nummern
    kosten nichts. Nullzellen des bincount sind deshalb keine leeren
    Tracklets, sondern nicht vergebene IDs, und fallen heraus. Die Laenge
    des Ergebnisses ist damit n_tracklets_roh aus kandidaten.py.
    """
    tid = np.asarray(tracklet_id)
    zaehl = np.bincount(tid[tid >= 0])
    return zaehl[zaehl > 0].astype(np.int64)


def kurve(on_je_run: dict, on_ges, werte) -> str:
    """Der Filter als Kurve statt als Punkt.

    min_events ist genau dann keine Wahl, wenn die Verteilung zwischen den
    beiden Gipfeln leer ist -- dann liefert jede Schwelle in der Luecke
    dasselbe. Bleibt die Knotenzahl ueber einen ganzen Bereich stehen, ist
    die Luecke da; faellt sie durchgehend, ist sie es nicht und die
    Schwelle wird zu einem Tausch zwischen Knotenzahl und verlorenen
    Events.
    """
    s = (f"  {'Schwelle':>8} {'Knoten':>12} {'Knoten %':>9} "
         f"{'Events weg %':>13}   Knoten % je Run\n")
    for g in werte:
        n = int((on_ges >= g).sum())
        ev = float(on_ges[on_ges < g].sum() / on_ges.sum()) * 100
        r = [float((v >= g).mean()) * 100 for v in on_je_run.values()]
        s += (f"  {g:>8} {n:>12,} {n / len(on_ges) * 100:>7.1f} % "
              f"{ev:>11.2f} %   {min(r):>5.1f} .. {max(r):<5.1f}\n")
    return s


def kleinbild(on_ges, bis: int = 50) -> str:
    """Die Verteilung genau dort, wo der Filter schneidet.

    Der Befund vom 14.08.2026 war eine LUECKE zwischen etwa 5 und 20
    Events -- zwei Gipfel, dazwischen fast nichts. Ein Mittelwert kann sie
    nicht zeigen, ein Quantil auch nicht. Die letzte Zeile sammelt alles
    ab `bis` und geht bewusst NICHT in die Skalierung ein, sonst
    verschwaende der grosse Gipfel den ganzen Bereich davor.
    """
    h = np.bincount(np.minimum(on_ges, bis), minlength=bis + 1)
    hoch = max(int(h[1:bis].max()), 1)
    s = ""
    for k in range(1, bis + 1):
        marke = f">={bis}" if k == bis else str(k)
        s += f"  {marke:>5} {h[k]:>10,}  {'#' * int(48 * h[k] / hoch)}\n"
    return s

def beschreibe(n_on) -> dict:
    """Lagemasse einer Eventzahl-Verteilung.

    Mittel und Standardabweichung stehen dabei, weil der Betreuer sie
    verlangt -- ueber die UNGETEILTE Verteilung sind sie irrefuehrend, weil
    sie zweigipflig ist. Deshalb immer zusammen mit den Quantilen lesen und
    getrennt nach den beiden Gipfeln.
    """
    if len(n_on) == 0:
        return {"n": 0}
    q = np.percentile(n_on, QUANTILE)
    return {"n": int(len(n_on)),
            "summe_events": int(n_on.sum()),
            "mittel": float(n_on.mean()),
            "sd": float(n_on.std(ddof=1)) if len(n_on) > 1 else float("nan"),
            "max": int(n_on.max()),
            **{f"p{p}": float(v) for p, v in zip(QUANTILE, q)}}


def kreuztabelle(n_on, n_gt, grenze: int) -> dict:
    """klein x Rauschen. Beantwortet: entfernt der Filter die Rauschtracklets?

    Die beiden abgeleiteten Zahlen sind das Ergebnis:
      rausch_entfernt  Anteil der Rausch-Tracklets, die der Filter faengt
      klein_ist_rausch Anteil der entfernten Tracklets, die Rauschen sind
    Die erste sagt, ob der Filter seine Aufgabe erfuellt, die zweite, was er
    nebenbei mitnimmt.
    """
    klein = n_on < grenze
    rausch = n_gt == 0
    z = {"klein_rausch": int((klein & rausch).sum()),
         "klein_echt": int((klein & ~rausch).sum()),
         "gross_rausch": int((~klein & rausch).sum()),
         "gross_echt": int((~klein & ~rausch).sum())}
    n = sum(z.values())
    # Eventanteil und mittlere Groesse JE ZELLE. Ohne sie laesst sich der
    # Preis des Filters nur aus Prozentdifferenzen schaetzen, und die
    # Fehlerfortpflanzung frisst dabei das Ergebnis.
    for name, maske in (("klein_rausch", klein & rausch),
                        ("klein_echt", klein & ~rausch),
                        ("gross_rausch", ~klein & rausch),
                        ("gross_echt", ~klein & ~rausch)):
        z[f"{name}_mittel"] = (float(n_on[maske].mean()) if maske.any()
                               else float("nan"))
        z[f"{name}_event_frac"] = float(n_on[maske].sum() / n_on.sum())
    z["rausch_entfernt"] = (z["klein_rausch"] / max(int(rausch.sum()), 1))
    z["klein_ist_rausch"] = (z["klein_rausch"] / max(int(klein.sum()), 1))
    z["rausch_event_frac"] = float(n_on[rausch].sum() / n_on.sum())
    z["n"] = n
    return z


def auswerten(n_on, n_gt, grenze: int, schwellen) -> dict:
    """Alle Kennzahlen EINER Menge von Tracklets (ein Run oder gepoolt)."""
    behalten = n_on[n_on >= grenze]
    out = {
        "alle": beschreibe(n_on),
        "klein": beschreibe(n_on[n_on < grenze]),
        "behalten": beschreibe(behalten),
        "klein_frac": float((n_on < grenze).mean()),
        "klein_event_frac": float(n_on[n_on < grenze].sum() / n_on.sum()),
    }
    # Anteil der BEHALTENEN Tracklets, die eine volle Stichprobe hergeben.
    # Darunter wird gepolstert und die Maske muss greifen.
    for s in schwellen:
        out[f"ge{s}_frac"] = (float((behalten >= s).mean())
                              if len(behalten) else float("nan"))
    # Der Schwanz, fuer den Betreuer und fuer die Wahl von n_events: ein
    # Tracklet mit tausend Events wird von 32 Token zu 3 % abgebildet.
    for s in (500, 1000, 2000):
        out[f"ueber{s}"] = int((n_on >= s).sum())
        out[f"ueber{s}_frac"] = float((n_on >= s).mean())
    if n_gt is not None:
        out["kreuz"] = kreuztabelle(n_on, n_gt, grenze)
    return out


def zeile(name: str, a: dict, grenze: int, schwellen) -> str:
    al, kl, be = a["alle"], a["klein"], a["behalten"]
    s = (f"{name:<10s} {al['n']:>8,}  "
         f"Mittel {al['mittel']:6.1f} SD {al['sd']:6.1f}  "
         f"p5/p50/p95 {al['p5']:5.0f}/{al['p50']:5.0f}/{al['p95']:5.0f}  "
         f"max {al['max']:>6,}\n")
    s += (f"{'  <' + str(grenze):<10s} {kl['n']:>8,}  "
          f"({a['klein_frac']*100:4.1f} % der Knoten, "
          f"{a['klein_event_frac']*100:5.2f} % der Events)  "
          f"Mittel {kl['mittel']:5.1f} SD {kl['sd']:5.1f}\n")
    s += (f"{'  >=' + str(grenze):<10s} {be['n']:>8,}  "
          f"Mittel {be['mittel']:6.1f} SD {be['sd']:6.1f}  "
          f"p5/p50/p95 {be['p5']:5.0f}/{be['p50']:5.0f}/{be['p95']:5.0f}  "
          + "  ".join(f">={s_}: {a[f'ge{s_}_frac']*100:4.1f} %"
                      for s_ in schwellen) + "\n")
    if "kreuz" in a:
        k = a["kreuz"]
        s += (f"{'  Kreuz':<10s} klein&rausch {k['klein_rausch']:>8,}  "
              f"klein&echt {k['klein_echt']:>8,}  "
              f"gross&rausch {k['gross_rausch']:>8,}  "
              f"gross&echt {k['gross_echt']:>8,}\n")
        s += (f"{'':<10s} Filter faengt {k['rausch_entfernt']*100:5.1f} % des "
              f"Rauschens; {k['klein_ist_rausch']*100:5.1f} % des Entfernten "
              f"ist Rauschen; Rauschen traegt "
              f"{k['rausch_event_frac']*100:5.2f} % der Events\n")
        s += (f"{'':<10s} je Zelle  " + "  ".join(
            f"{c}: {k[c + '_event_frac']*100:5.2f} % Events, "
            f"Mittel {k[c + '_mittel']:6.1f}"
            for c in ("klein_rausch", "klein_echt",
                      "gross_rausch", "gross_echt")) + "\n")
    s += (f"{'  Schwanz':<10s} " + "  ".join(
        f">={g}: {a[f'ueber{g}']:,} ({a[f'ueber{g}_frac']*100:.3f} %)"
        for g in (500, 1000, 2000)) + "\n")
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="train")
    ap.add_argument("--filter", type=int, default=15,
                    help="Mindestgroesse in Events")
    ap.add_argument("--schwellen", type=int, nargs="+", default=[32, 64],
                    help="Stichprobengroessen, deren Abdeckung interessiert")
    ap.add_argument("--kurve", type=int, nargs="+",
                    default=[2, 5, 10, 15, 20, 25, 32, 50],
                    help="Schwellen fuer die Filterkurve")
    ap.add_argument("--bis", type=int, default=50,
                    help="obere Grenze des Histogramms")
    ap.add_argument("--json", default=None, help="Ergebnis zusaetzlich als JSON")
    args = ap.parse_args()

    nb, ps, tr, sv = Neighborhood(), PairSampling(), Training(), Solver()
    treffer, d = dateien(args.split, nb, ps, tr, sv)
    print(f"{len(treffer)} Datei(en) in {d}")
    print(f"Filter {args.filter} Events, Schwellen {args.schwellen}\n")

    pro_run, on_je_run = {}, {}
    for run_nr, pfad in treffer:
        with np.load(pfad) as f:
            n_on = groessen(f["tracklet_id"])
        pro_run[run_nr] = auswerten(n_on, None, args.filter, args.schwellen)
        on_je_run[run_nr] = n_on
        print(zeile(f"run{run_nr}", pro_run[run_nr], args.filter,
                    args.schwellen))

    on_ges = np.concatenate(list(on_je_run.values()))
    gesamt = auswerten(on_ges, None, args.filter, args.schwellen)
    print("=" * 78)
    print(zeile("GEPOOLT", gesamt, args.filter, args.schwellen))

    # Spannweite zwischen den Runs. Sie trennt Streuung ZWISCHEN Runs von
    # Streuung INNERHALB eines Runs -- die gepoolte Zahl allein kann beides.
    print("Spannweite ueber die Runs:")
    for key, pfad_ in (("klein_frac", ("klein_frac",)),
                       ("klein_event_frac", ("klein_event_frac",)),
                       *[(f"ge{s}_frac", (f"ge{s}_frac",))
                         for s in args.schwellen]):
        werte = [a[pfad_[0]] for a in pro_run.values()]
        print(f"  {key:<18s} {min(werte)*100:6.2f} % .. {max(werte)*100:6.2f} %")

    print("\nFilterkurve (gepoolt, und Spannweite ueber die Runs):")
    print(kurve(on_je_run, on_ges, args.kurve))
    print(f"Verteilung 1..{args.bis} Events (gepoolt):")
    print(kleinbild(on_ges, args.bis))

    if args.json:
        from pathlib import Path
        p = Path(args.json).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(
            {"filter": args.filter, "schwellen": args.schwellen,
             "split": args.split, "quelle": str(d),
             "pro_run": {str(k): v for k, v in pro_run.items()},
             "gepoolt": gesamt}, indent=2), encoding="utf-8")
        print(f"\n-> {p}")


if __name__ == "__main__":
    main()