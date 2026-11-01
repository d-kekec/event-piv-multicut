#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stufe2_parameter_schaetzer.py -- Belege fuer die Gruppe kandidaten.

Liest das Artefakt (A) aus Stufe 1 und liefert, woraus Stufe 2
ihre Parameter waehlt:

  min_events, n_events   Groessenverteilung der Tracklets, Filterkurve,
                         Abdeckung der Stichprobe; mit --gt dazu die
                         Kreuztabelle klein x Rauschen
  r_max, v_max           nur mit --gt: Zentroidabstand zwischen Knoten
                         DERSELBEN GT-Bahn nach dt, also Tracklets ab
                         --filter Events -- die Taille bei dt < alpha1,
                         die Drift als Steigung der Huelle

WOHER DIE DATEIEN KOMMEN. Ohne --quelle aus den Vorgaben von runconfig,
also dem Arbeitspunkt von STELLA. Die Gruppen eines anderen Datensatzes
(neighborhood=batch1 usw.) erreichen ein Werkzeug nicht -- tools/ kennt
kein Hydra; dort zeigen die Vorgaben ins Leere und die Suche bricht laut
ab. Fuer jeden anderen Datensatz das Verzeichnis mit --quelle angeben.
alpha1, Polaritaet, Split und Run stehen im meta der Dateien und werden
dort gelesen, nicht eingetippt.

OHNE --gt nur (A): n_on je Tracklet ist ein bincount ueber tracklet_id,
genau die Groesse, auf die knoten_bauen filtert. Billig: liest nur (A).

MIT --gt zusaetzlich Eventstrom und Labels, ueber lade_fenster und
knoten_bauen wie in [6a]: dieselben Zentroide, dieselbe Mehrheit, keine
Nachbildung. Braucht auf dichten Runs viel Arbeitsspeicher.

Aufruf aus der Repo-Wurzel, ohne Hydra. EVENTPIV_DATA und EVENTPIV_OUT
muessen zu den Dateien passen (scripts/_cli.py). <tracklets> ist das
Verzeichnis der Dateien, die build_tracklets hinter "->" meldet.
    python tools/schaetzer/stufe2_parameter_schaetzer.py
    S=tools/schaetzer/stufe2_parameter_schaetzer.py
    python $S --quelle <tracklets> --runs 8 18
    python $S --gt --quelle <tracklets> --runs 18 30 66 80 --json <ziel>.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from eventpiv.kern import config
from eventpiv.kern.runconfig import (Neighborhood, PairSampling, Selection,
                                     Solver, Training, Window)
from eventpiv.kern.status import STATUS_NOISE
from eventpiv.kette.fenster import lade_fenster
from eventpiv.kette.knoten import knoten_bauen

QUANTILE = (5, 25, 50, 75, 95)
# Zentroidabstand: Klassenbreite und Obergrenze in dr, die gedruckten
# dt-Klassen und die erste Klasse der Drift-Sekante.
DR_BREITE, DR_MAX = 0.05, 200.0
DRIFT_ZEILEN = (0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50)
SEKANTE_AB = 10


def dateien(split: str, quelle, suffix, runs):
    """Die (A)-Dateien eines Splits, nach Run-Nummer sortiert.

    Gesucht statt geraten: welche Runs gebaut wurden, weiss das
    Verzeichnis, nicht die Konfiguration. glob("*.npz") greift nur eine
    Ebene tief -- stufe2/ liegt darunter und stoert nicht.

    Ohne quelle aus den Vorgaben von runconfig (STELLA), sonst das
    angegebene Verzeichnis. suffix ist der Solve-Teil des Dateinamens,
    Vorgabe Solver().short(); mit solver.cost_offset=0.5 kommt "_o0.5"
    dazu. Ein angefragter Run, der fehlt, bricht ab, statt still zu
    fehlen.
    """
    if quelle:
        d = Path(quelle).expanduser()
    else:
        d = config.tracklet_file(Neighborhood(), PairSampling(), Training(),
                                 Selection(split=split, run_nr=0), Solver(),
                                 create=False).parent
    suffix = suffix or Solver().short()
    if not d.is_dir():
        raise SystemExit(f"{d} fehlt -- ohne --quelle zeigen die Vorgaben "
                         "auf STELLA, fuer andere Daten --quelle angeben; "
                         "sonst erst scripts/build_tracklets.py.")
    muster = re.compile(rf"^{re.escape(split)}_run(\d+)_"
                        rf"{re.escape(suffix)}\.npz$")
    treffer = [(int(m.group(1)), p) for p in sorted(d.glob("*.npz"))
               if (m := muster.match(p.name))]
    if runs:
        fehlend = sorted(set(runs) - {r for r, _ in treffer})
        if fehlend:
            raise SystemExit(f"Runs {fehlend} fehlen in {d} "
                             f"({split}/{suffix}).")
        treffer = [(r, p) for r, p in treffer if r in set(runs)]
    if not treffer:
        raise SystemExit(f"Keine Dateien in {d}, die auf {split}/"
                         f"{suffix} passen.")
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

    Gesucht ist eine LUECKE zwischen zwei Gipfeln, in der fast nichts
    liegt (kurve). Ein Mittelwert kann sie nicht zeigen, ein Quantil auch
    nicht. Die letzte Zeile sammelt alles ab `bis` und geht bewusst NICHT
    in die Skalierung ein, sonst verschwaende der grosse Gipfel den ganzen
    Bereich davor.
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

    Mittel und Standardabweichung stehen dabei, sind ueber die UNGETEILTE
    Verteilung aber irrefuehrend, wenn sie zweigipflig ist. Deshalb immer
    zusammen mit den Quantilen lesen und getrennt nach den beiden Gipfeln.
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
    # Der Schwanz, fuer die Wahl von n_events: ein Tracklet mit tausend
    # Events bilden n_events Tokens nur zu n_events/1000 ab.
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


def lies_meta(pfad) -> dict:
    """meta eines (A)-Artefakts. np.load liest dabei nur diesen Eintrag."""
    with np.load(pfad) as f:
        return json.loads(str(f["meta"]))


def _wurzel(pfad, marke: str):
    """Ausgabewurzel eines Pfads: alles vor dem Verzeichnis `marke`."""
    teile = Path(pfad).expanduser().resolve().parts
    return Path(*teile[:teile.index(marke)]) if marke in teile else None


def gt_messung(pfad, tid, meta, n_fenster: int, min_events: int):
    """Labels dazu: n_on und n_gt je Tracklet, Zentroid-Histogramm.

    Derselbe Weg wie kandidaten_fuer_run: lade_fenster ueber den ganzen
    Run, Laengenvertrag zu (A) pruefen, knoten_bauen -- hier mit
    min_events = 1, damit auch die kleinen Tracklets in der Kreuztabelle
    stehen. Eine eigene Funktion, damit Strom und Knoten mit dem Ruecksprung
    freigegeben werden -- auf dichten Runs sind das viele GB.

    Der Zentroidabstand rechnet nur ueber Knoten ab --filter Events, wie
    [6a] sie mit min_events = --filter bilden wuerde. Kruemel darunter haben
    Zentroide aus wenigen Events, oft auf denselben Pixeln; mitgezaehlt
    beschriebe die Taille die Kruemel statt der Knoten.

    Die Label-Datei muss unter derselben Ausgabewurzel liegen wie (A).
    Sonst passt EVENTPIV_OUT nicht zu den Dateien, und das fiele erst am
    Laengenvertrag auf -- oder gar nicht, haetten zwei Runs gleich viele
    Events.
    """
    split, run_nr = meta["split"], int(meta["run"])
    gt_pfad = Path(config.label_file(split, run_nr))
    wa = _wurzel(pfad, "wp")
    if wa is None or wa != _wurzel(gt_pfad, "labels"):
        raise SystemExit(f"{gt_pfad} liegt nicht unter der Ausgabewurzel von "
                         f"{pfad} -- passt EVENTPIV_OUT zu den Dateien?")
    if not gt_pfad.exists():
        raise SystemExit(f"{gt_pfad} fehlt -- erst die Labels bauen.")
    fe = lade_fenster(Selection(split=split, run_nr=run_nr),
                      Window(t0=None, dt=None),
                      polarity=meta["neighborhood"]["polarity"],
                      gt_path=gt_pfad)
    if len(fe) != len(tid):
        raise SystemExit(f"VERTRAG VERLETZT: {len(fe):,} Events im Strom, "
                         f"{len(tid):,} in {Path(pfad).name}.")
    if not fe.hat_gt:
        raise SystemExit(f"{gt_pfad} lieferte keine Ground Truth.")
    kn = knoten_bauen(fe.t, fe.x, fe.y, tid, min_events=1,
                      track_id=fe.track_id,
                      status_noise=np.asarray(fe.status) == STATUS_NOISE)
    tab = kn.tabelle
    e = (tab["n_gt"] == 1) & (tab["n_on"] >= int(min_events))
    hist = zentroid_histogramm(tab["gt_track_id"][e], tab["t_min"][e],
                               tab["x_c"][e], tab["y_c"][e],
                               meta["neighborhood"]["alpha"], n_fenster)
    return tab["n_on"], tab["n_gt"], hist


def zentroid_histogramm(gt, t, x, y, alpha1, n_fenster: int) -> np.ndarray:
    """Zentroidabstand zwischen Tracklets DERSELBEN GT-Bahn, nach dt.

    Rueckgabe: Zaehlmatrix (n_fenster + 1, n_dr + 1). Zeile k sammelt die
    Paare mit k*alpha1 <= dt < (k+1)*alpha1, Spalte j die mit
    j*DR_BREITE <= dr < (j+1)*DR_BREITE; die letzte Spalte alles ab DR_MAX.

    dt ist die Differenz der t_min, dr der Abstand der Zentroide -- genau
    die beiden Groessen, auf denen die Kandidatensuche arbeitet. Zeile 0
    enthaelt vor allem Bruchstuecke im selben Fenster; ihre Streuung ist
    die Taille, die r_max decken muss. Darueber waechst dr mit der Drift.

    OHNE NACHBARSUCHE und ohne Annahme ueber r_max: sortiert nach (Bahn,
    t_min), dann Versatz L = 1, 2, ... ueber die Zeilen. Innerhalb einer
    Bahn waechst dt mit L; findet ein Versatz kein Paar derselben Bahn
    unter der Grenze mehr, findet es auch kein groesserer. Jedes Paar wird
    genau einmal gezaehlt.
    """
    n_dr = int(round(DR_MAX / DR_BREITE))
    h = np.zeros((n_fenster + 1, n_dr + 1), np.int64)
    if len(gt) < 2:
        return h
    o = np.lexsort((t, gt))
    g, tt = np.asarray(gt)[o], np.asarray(t, np.float64)[o]
    xx, yy = np.asarray(x, np.float64)[o], np.asarray(y, np.float64)[o]
    grenze = (n_fenster + 1) * float(alpha1)
    for L in range(1, len(g)):
        dt = tt[L:] - tt[:-L]
        nimm = (g[L:] == g[:-L]) & (dt < grenze)
        if not nimm.any():
            break
        k = (dt[nimm] // alpha1).astype(np.int64)
        dr = np.hypot(xx[L:][nimm] - xx[:-L][nimm],
                      yy[L:][nimm] - yy[:-L][nimm])
        j = np.minimum((dr / DR_BREITE).astype(np.int64), n_dr)
        h += np.bincount(k * (n_dr + 1) + j,
                         minlength=h.size).reshape(h.shape)
    return h


def drift_auswerten(h, alpha1) -> dict:
    """Lagemasse je dt-Klasse aus der Zaehlmatrix, dazu die Drift.

    Quantile auf die obere Kante ihrer dr-Klasse gerundet: hoechstens
    DR_BREITE zu gross, nie zu klein.

    DIE DRIFT ALS SEKANTE, NICHT AUS EINEM MODELL. Die Huelle aus Streuung
    plus Drift ist weder Kegel (Taille + v*dt) noch Hyperboloid
    (sqrt(Taille^2 + (v*dt)^2)) -- sie liegt dazwischen, und beide Formeln
    verfehlen v, solange die Drift klein gegen die Taille ist.
    Weit draussen nimmt jede glatte Huelle aber die Steigung der Drift an.
    Die Sekante zwischen der ersten belegten Klasse ab SEKANTE_AB und der
    letzten belegten trifft sie, sobald die Drift ueber diese Spanne die
    Taille klar uebersteigt; sonst unterschaetzt sie. Faellt die Sekante
    zu grossem dt hin, sind die Bahnen gekruemmt -- dann ist ein v_max aus
    kurzen Zeiten fuer Stufe 2 zu weit.
    """
    oben = (np.arange(h.shape[1]) + 1) * DR_BREITE
    oben[-1] = np.inf                        # Ueberlauf ab DR_MAX
    out = {}
    for k in range(h.shape[0]):
        n = int(h[k].sum())
        if n == 0:
            out[k] = {"n": 0}
            continue
        kum = np.cumsum(h[k])
        q = {p: float(oben[np.searchsorted(kum, p / 100 * n)])
             for p in (50, 99, 99.9)}
        out[k] = {"n": n, "p50": q[50], "p99": q[99], "p999": q[99.9],
                  "max": float(oben[np.flatnonzero(h[k])[-1]])}
    belegt = [k for k in range(SEKANTE_AB, h.shape[0]) if out[k]["n"]]
    if len(belegt) >= 2 and belegt[-1] - belegt[0] >= SEKANTE_AB:
        a, b = belegt[0], belegt[-1]
        out["steigung"] = {"von": a, "bis": b, **{
            p: (out[b][p] - out[a][p]) / ((b - a) * alpha1) * 1000.0
            for p in ("p99", "p999")}}
    return out


def drift_text(name, d, alpha1, v_max_stufe1, zeilen) -> str:
    """Tabelle je dt-Klasse, dazu Taille und Drift gegen neighborhood.v_max."""
    s = (f"{name}: Zentroidabstand gleicher GT-Bahn (n_gt == 1, ab --filter), "
         f"alpha1 = {alpha1:g} µs\n"
         f"  {'dt [alpha1]':>11} {'Paare':>12} {'p50':>7} {'p99':>7} "
         f"{'p99,9':>7} {'max':>7}   Sekante p99,9 [px/ms]\n")
    vor = None
    for k in zeilen:
        z = d.get(k)
        if z is None or not z["n"]:
            continue
        sek = ""
        if (vor is not None and np.isfinite(z["p999"])
                and np.isfinite(d[vor]["p999"])):
            v = (z["p999"] - d[vor]["p999"]) / ((k - vor) * alpha1) * 1000
            sek = f"{v:8.3f}"
        klasse = f"{k}-{k + 1}"
        s += (f"  {klasse:>11} {z['n']:>12,} {z['p50']:7.2f} {z['p99']:7.2f} "
              f"{z['p999']:7.2f} {z['max']:7.2f}   {sek}\n")
        vor = k
    if d.get(0, {}).get("n"):
        s += (f"  Taille (dt < alpha1): p99 {d[0]['p99']:.2f} px, "
              f"p99,9 {d[0]['p999']:.2f} px\n")
    if "steigung" in d:
        st = d["steigung"]
        s += (f"  Drift, Sekante k = {st['von']}..{st['bis']}: "
              f"p99 {st['p99']:.3f}, p99,9 {st['p999']:.3f} px/ms  --  "
              f"neighborhood.v_max {v_max_stufe1 * 1000:.3f} px/ms "
              f"({v_max_stufe1:g} px/µs)\n")
    return s


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train")
    ap.add_argument("--quelle", default=None,
                    help="Verzeichnis der (A)-Dateien; ohne Angabe aus den "
                         "Vorgaben von runconfig (STELLA)")
    ap.add_argument("--solver", default=None,
                    help="Solve-Teil des Dateinamens, Vorgabe "
                         "Solver().short()")
    ap.add_argument("--runs", type=int, nargs="+", default=None,
                    help="Auswahl; ohne Angabe alle gefundenen")
    ap.add_argument("--gt", action="store_true",
                    help="Labels dazu: Kreuztabelle und Zentroidabstand "
                         "(liest den Eventstrom, viel Arbeitsspeicher)")
    ap.add_argument("--fenster", type=int, default=50,
                    help="Zentroidabstand bis dt < (fenster + 1) * alpha1")
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

    treffer, d = dateien(args.split, args.quelle, args.solver, args.runs)
    metas = {r: lies_meta(p) for r, p in treffer}
    for r, m in metas.items():
        if int(m["run"]) != r or m["split"] != args.split:
            raise SystemExit(f"run{r}: meta sagt {m['split']}/run{m['run']}.")
    nbs = {json.dumps(m["neighborhood"], sort_keys=True)
           for m in metas.values()}
    if len(nbs) > 1:
        raise SystemExit(f"Gemischte Quelle, {len(nbs)} verschiedene "
                         f"neighborhood-Bloecke in {d}.")
    nb = next(iter(metas.values()))["neighborhood"]
    print(f"{len(treffer)} Datei(en) in {d}")
    print(f"alpha1 {nb['alpha']:g} µs, Polaritaet {nb['polarity']}, "
          f"Filter {args.filter} Events, Schwellen {args.schwellen}"
          + (", mit Labels" if args.gt else "") + "\n")

    pro_run, on_je_run, gt_je_run, hist_je_run = {}, {}, {}, {}
    for run_nr, pfad in treffer:
        with np.load(pfad) as f:
            tid = f["tracklet_id"]
        n_on, n_gt = groessen(tid), None
        if args.gt:
            n_on_kn, n_gt, hist_je_run[run_nr] = gt_messung(
                pfad, tid, metas[run_nr], args.fenster, args.filter)
            # Zwei Wege zur selben Zahl: bincount ueber (A) gegen die
            # Knotentabelle, beide nach tracklet_id geordnet.
            if not np.array_equal(n_on_kn, n_on):
                raise SystemExit(f"run{run_nr}: Groessen aus (A) und aus "
                                 "knoten_bauen weichen ab.")
            gt_je_run[run_nr] = n_gt
        del tid
        pro_run[run_nr] = auswerten(n_on, n_gt, args.filter, args.schwellen)
        on_je_run[run_nr] = n_on
        print(zeile(f"run{run_nr}", pro_run[run_nr], args.filter,
                    args.schwellen), flush=True)

    on_ges = np.concatenate(list(on_je_run.values()))
    gt_ges = np.concatenate(list(gt_je_run.values())) if args.gt else None
    gesamt = auswerten(on_ges, gt_ges, args.filter, args.schwellen)
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

    drift = {}
    if args.gt:
        zeilen = [k for k in DRIFT_ZEILEN if k <= args.fenster]
        for run_nr, h in hist_je_run.items():
            drift[str(run_nr)] = drift_auswerten(h, nb["alpha"])
            print(drift_text(f"run{run_nr}", drift[str(run_nr)], nb["alpha"],
                             nb["v_max"], zeilen))
        drift["gepoolt"] = drift_auswerten(sum(hist_je_run.values()),
                                           nb["alpha"])
        print(drift_text("GEPOOLT", drift["gepoolt"], nb["alpha"],
                         nb["v_max"], zeilen))

    if args.json:
        p = Path(args.json).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(
            {"filter": args.filter, "schwellen": args.schwellen,
             "split": args.split, "quelle": str(d), "neighborhood": nb,
             "gt": args.gt,
             "pro_run": {str(k): v for k, v in pro_run.items()},
             "gepoolt": gesamt, "drift": drift}, indent=2),
            encoding="utf-8")
        print(f"\n-> {p}")


if __name__ == "__main__":
    main()