#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
runauswahl.py -- Trainingsruns nach Eventrate und Begegnungsdichte waehlen.

Die Kategorien aus bewerten/overlap.py (MAX_OVERLAP / MIN_OVERLAP /
CROSSING) trennen Runs MIT von Runs OHNE NEIN-Paare. Auf einem dicht
besetzten Datensatz gibt es die zweite Sorte nicht, alle Runs fallen in
dieselbe Klasse, und die Einteilung sagt nichts mehr. Gebraucht wird
stattdessen eine stetige Tabelle.

Gelesen werden nur die Bounding Boxes und die Dateigroessen -- kein Event,
kein Netz, kein Solve.

SCHWELLE fuer "Begegnung", hergeleitet statt gesetzt: zwei Wolken koennen
erst dann eine Kante teilen, wenn ihre Zentren naeher als
2*r_wolke + r_max liegen -- ein Event am aeusseren Rand der einen und
eines am aeusseren Rand der anderen, dazwischen der Nachbarschaftsradius.

SPEICHERSPITZE: der Jaccard legt die Nachbarschaftsmatrix eines Fensters
dicht als float32 an, also (Rate*alpha)^2 * 4 Byte. Sie ist die harte
Obergrenze fuer die Wahl und folgt allein aus der Dateigroesse. Die Rate
zaehlt alle Events der Datei; enthaelt ein Datensatz auch OFF-Events,
ueberschaetzt sie die Last von Stufe 1, die nur ON-Events rechnet.

Aufruf ohne Hydra und ohne EVENTPIV_*. <verzeichnis> enthaelt run<N>.dat
und run<N>_bbox.npy eines Splits. Alle Schluessel sind optional: n Zahl der
Runs, alpha_us Fensterlaenge in µs, r_wolke und r_max in px fuer die
SCHWELLE, budget_gib Speicherspitze in GiB, proben Rasterpunkte je Run,
min_ptk Partikel je Bild als untere Schranke, deckel NEIN-Paare je Run wie
pairs.max_no. Die Vorgaben des Werkzeugs folgen nicht kern.runconfig.
  python tools/runauswahl.py <verzeichnis> [n=30] [alpha_us=500]
         [r_wolke=...] [r_max=...] [budget_gib=8] [proben=100]
         [min_ptk=100] [deckel=1e6]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
KOPF_BYTE = 108        # Prophesee-DAT: fuenf Kommentarzeilen + ev_type/ev_size
BYTE_JE_EVENT = 8

# Die Rate ist ein Mittel ueber den Run; einzelne Fenster liegen darueber.
# Sicherheitsfaktor auf den Speicher, der mit dem Quadrat der Knotenzahl
# je Fenster geht; am Datensatz kalibriert.
SPITZE = 1.5

# Rohe NEIN-Paare je Fenster als Potenzgesetz der Eventrate: Bezugspunkt
# (Rate in MEv/s, NEIN-Paare je Fenster) und Exponent, am Datensatz
# kalibriert, grob. NEIN braucht zwei Events VERSCHIEDENER Partikel in
# einer Nachbarschaft, waechst also ueberproportional mit der Dichte.
NEIN_JE_FENSTER_BEI = (28.2, 7027.0)
NEIN_EXPONENT = 1.85


def _zentren(b: np.ndarray) -> np.ndarray:
    return np.stack([b["x"] + b["w"] / 2.0, b["y"] + b["h"] / 2.0], axis=1)


def kennzahlen(bbox_pfad: Path, dat_pfad: Path, schwelle: float,
               alpha_us: float, proben: int) -> dict:
    """Eine Zeile je Run. Rasterpunkte werden gleichmaessig abgetastet --
    alle zu nehmen kostet ein Vielfaches fuer dieselbe Aussage."""
    b = np.load(bbox_pfad)
    n_ev = (dat_pfad.stat().st_size - KOPF_BYTE) // BYTE_JE_EVENT

    raster = np.unique(b["t"])
    dauer_s = float(raster[-1] - raster[0]) / 1e6
    rate = n_ev / dauer_s / 1e6                     # MEv/s
    knoten = rate * 1e6 * alpha_us / 1e6            # Knoten je Fenster
    gib = knoten ** 2 * 4 / 2 ** 30 * SPITZE
    r0, n0 = NEIN_JE_FENSTER_BEI
    nein_je_f = n0 * (rate / r0) ** NEIN_EXPONENT
    n_fenster = float(raster[-1] - raster[0]) / alpha_us

    # Gruppieren ueber searchsorted statt ueber eine Maske je Rasterpunkt:
    # letzteres ist O(Rasterpunkte * Zeilen) und dominiert sonst alles.
    ordnung = np.argsort(b["t"], kind="stable")
    b_s = b[ordnung]
    grenzen = np.append(np.searchsorted(b_s["t"], raster), b_s.size)
    z = _zentren(b_s)

    auswahl = np.unique(np.linspace(0, raster.size - 1,
                                    min(proben, raster.size)).astype(int))
    paare, n_ptk, min_d = [], [], np.inf
    for i in auswahl:
        p = z[grenzen[i]:grenzen[i + 1]]
        n_ptk.append(p.shape[0])
        if p.shape[0] < 2:
            continue
        baum = cKDTree(p)
        paare.append(len(baum.query_pairs(r=schwelle)) / p.shape[0])
        min_d = min(min_d, float(baum.query(p, k=2)[0][:, 1].min()))

    # |v| als p99.9 der Bahnschritte -- ein Quantil, kein Maximum: die GT
    # enthaelt vereinzelte Spruenge, die kein Partikel wirklich macht.
    t_ms = b["t"].astype(np.float64) / 1000.0
    o = np.lexsort((t_ms, b["track_id"]))
    tid, ts, zs = b["track_id"][o], t_ms[o], _zentren(b)[o]
    dt = ts[1:] - ts[:-1]
    gut = (tid[1:] == tid[:-1]) & (dt > 0)
    v = np.linalg.norm(zs[1:] - zs[:-1], axis=1)[gut] / dt[gut]

    return {"run": int("".join(c for c in bbox_pfad.stem if c.isdigit())),
            "rate": rate, "gib": gib, "knoten": knoten,
            "tracks": int(np.unique(b["track_id"]).size),
            "ptk": float(np.median(n_ptk)),
            "v_max": float(np.percentile(v, 99.9)),
            "nachbarn": float(np.mean(paare)) if paare else float("nan"),
            "min_dist": min_d,
            "nein_je_f": nein_je_f, "n_fenster": n_fenster}


def regel_auswahl(zeilen: list[dict], n_ziel: int, budget_gib: float,
                  min_ptk: float) -> tuple[list[dict], list[dict], list[dict]]:
    """Schichtet ueber die Eventrate, zwischen zwei Schranken.

    OBEN eine Schranke, keine Wahl: was der dichte Jaccard nicht traegt,
    kann nicht gerechnet werden.

    UNTEN ebenfalls eine Schranke. Die Zahl der Begegnungen je Partikel
    folgt bei gleichverteilten Partikeln aus ihrer Zahl allein,
    N*pi*r^2/(2*A). Unterhalb von min_ptk Partikeln je Bild liefert ein
    Run kaum NEIN-Paare -- fuers Training wertlos, als Kontrolle dagegen
    wertvoll: dort DARF nichts verschmelzen.

    Aus dem Rest wird gleichmaessig ueber die Rate gezogen, nicht die
    dichtesten genommen: die Rate bestimmt die lokale
    Nachbarschaftsdichte und damit den Merkmalsraum, den das Netz sieht.
    Traegt es nur dichte Nachbarschaften, ist der Einsatz auf duenneren
    Daten eine Extrapolation.

    Rueckgabe: (gewaehlt, zu_dicht, zu_duenn).
    """
    zu_dicht = [z for z in zeilen if z["gib"] > budget_gib]
    zu_duenn = [z for z in zeilen
                if z["gib"] <= budget_gib and z["ptk"] < min_ptk]
    drin = sorted((z for z in zeilen if z["gib"] <= budget_gib
                   and z["ptk"] >= min_ptk), key=lambda z: z["rate"])
    if len(drin) <= n_ziel:
        return drin, zu_dicht, zu_duenn
    idx = sorted(set(np.linspace(0, len(drin) - 1, n_ziel).round()
                     .astype(int).tolist()))
    return [drin[i] for i in idx], zu_dicht, zu_duenn


def _schritt_bis_deckel(z: dict, deckel: float) -> float:
    """Groesstes fenster_schritt, bei dem dieser Run den NEIN-Deckel noch
    erreicht. Kleiner heisst: mehr Fenster noetig, sonst ist der Run im
    Datensatz untergewichtet."""
    noetig = deckel / max(z["nein_je_f"], 1e-9)
    return z["n_fenster"] / max(noetig, 1e-9)


def _zeile(z: dict, deckel: float) -> str:
    return (f"{z['run']:>4} {z['rate']:>8.1f} {z['gib']:>7.2f} "
            f"{z['knoten']:>8.0f} {z['tracks']:>7} {z['ptk']:>7.0f} "
            f"{z['v_max']:>8.3f} {z['nachbarn']:>9.2f} {z['min_dist']:>9.2f} "
            f"{_schritt_bis_deckel(z, deckel):>9.1f}")


def bericht(ordner: Path, n_ziel: int, alpha_us: float, r_wolke: float,
            r_max: float, budget_gib: float, proben: int,
            min_ptk: float, deckel: float) -> None:
    schwelle = 2 * r_wolke + r_max
    print(f"{ordner}   alpha={alpha_us:.0f} us   Schwelle "
          f"2*{r_wolke}+{r_max} = {schwelle:.1f} px   "
          f"Budget {budget_gib:.1f} GiB")
    print(f"{'run':>4} {'MEv/s':>8} {'GiB':>7} {'Knoten':>8} {'tracks':>7} "
          f"{'Ptk/Bild':>7} {'v_max':>8} {'Nachb/Pt':>9} {'min_d':>9} "
          f"{'schritt':>9}")

    zeilen = []
    for p in sorted(ordner.glob("run*_bbox.npy")):
        dat = p.with_name(p.name.replace("_bbox.npy", ".dat"))
        if not dat.exists():
            print(f"  {p.name}: keine .dat, uebersprungen")
            continue
        z = kennzahlen(p, dat, schwelle, alpha_us, proben)
        zeilen.append(z)
        print(_zeile(z, deckel))

    gewaehlt, zu_dicht, zu_duenn = regel_auswahl(zeilen, n_ziel,
                                                 budget_gib, min_ptk)
    print(f"\n  ueber dem Budget ({len(zu_dicht)}): "
          f"{sorted(z['run'] for z in zu_dicht)}")
    print(f"  zu duenn fuer NEIN-Paare, als KONTROLLE brauchbar "
          f"({len(zu_duenn)}):")
    print("  kontrolle=[" + ",".join(str(z["run"]) for z in
                                     sorted(zu_duenn,
                                            key=lambda z: z["run"])) + "]")
    print(f"\n  GEWAEHLT ({len(gewaehlt)}), geschichtet ueber die Rate:")
    print("  runs=[" + ",".join(str(z["run"]) for z in
                                sorted(gewaehlt, key=lambda z: z["run"])) + "]")
    r = np.array([z["rate"] for z in gewaehlt])
    g = np.array([z["gib"] for z in gewaehlt])
    print(f"  Rate  min={r.min():.1f}  median={np.median(r):.1f}  "
          f"max={r.max():.1f} MEv/s")
    print(f"  Jaccard-Spitze  median={np.median(g):.2f}  max={g.max():.2f} GiB"
          f"  (inkl. Faktor {SPITZE} fuer das dichteste Fenster)")
    s = min(_schritt_bis_deckel(z, deckel) for z in gewaehlt)
    print(f"  pairs.fenster_schritt <= {s:.0f}, damit auch der duennste "
          f"gewaehlte Run den NEIN-Deckel von {deckel:,.0f} erreicht")


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = dict(kv.split("=", 1) for kv in a[1:] if "=" in kv)
    bericht(Path(a[0]).expanduser(),
            int(opt.get("n", 30)), float(opt.get("alpha_us", 500)),
            float(opt.get("r_wolke", 5.1)), float(opt.get("r_max", 5.1)),
            float(opt.get("budget_gib", 8)), int(opt.get("proben", 100)),
            float(opt.get("min_ptk", 100)), float(opt.get("deckel", 1e6)))