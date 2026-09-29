#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wo entstehen die Identitaetswechsel? Zerlegung einer Partition aus [6c].

Je GT-Partikel die IDSW auf dem GT-Raster, gerechnet wie in idsw_metrics,
dazu die Geschwindigkeit aus den Boxmitten, die Eventrate und der Anteil
grauer Events (keiner Trajektorie zugeordnet).

Je Wechsel die Art:
  Bruch       die alte Trajektorie endet, die neue beginnt erst dort --
              zwei Stuecke hintereinander, die eine Verknuepfung mit
              genuegend Reichweite zusammenfuehren koennte
  kein Bruch  die alte laeuft weiter oder die neue gab es schon, geteilt in
              eigenes Stueck    beide Bahnen gehoeren mehrheitlich dem
                                wechselnden Partikel
              fremd beteiligt   mindestens eine gehoert einem anderen
und beim Bruch die Luecke zwischen beiden. Dazu je Wechsel: Rueckkehr in
eine fruehere Bahn und mehrdeutige Events im Rasterpunkt (Ueberlapp).

Anker: die IDSW-Summe je Run muss der Zahl aus [6c] gleichen.

Aufruf mit dem solves-Verzeichnis aus dem Fit-Log und dem Versatz von
Stufe 2 (solver2.cost_offset), mehrere Werte nacheinander:
    python tools/6c_idsw_auswerter.py <solves> 8 13 38 41 66 92 --versatz 0.5 1
"""
import argparse
from pathlib import Path

import numpy as np

from eventpiv.bewerten.evaluate import _einheiten, gt_raster_dt, idsw_metrics
from eventpiv.kern import config
from eventpiv.kern.data import load_bbox
from eventpiv.kern.graph import polarity_mask
from eventpiv.kern.runconfig import Solver, Solver2
from eventpiv.kern.status import STATUS_AMBIG, STATUS_NOISE

SPLIT = "train"
STAMM = "train_run{r}_{stamm}.npz"                # Partition aus [6c]
V_GRENZEN = (0.0, 0.1, 0.2, 0.4, 0.8, np.inf)     # px/ms
RATE_GRENZE = 30.0                                # je ms; Nachweisgrenze bei min_events 15
TOL = 2                                           # Rasterpunkte Spiel beim Bruch


def geschwindigkeit(bb, ids):
    """Median |v| je Partikel aus den Boxmitten, px/ms. NaN ohne Boxen."""
    k = bb["track_id"].astype(np.int64)
    t = bb["t"].astype(np.float64)
    cx = bb["x"].astype(np.float64) + bb["w"].astype(np.float64) / 2
    cy = bb["y"].astype(np.float64) + bb["h"].astype(np.float64) / 2
    o = np.lexsort((t, k))
    k, t, cx, cy = k[o], t[o], cx[o], cy[o]
    selbe = k[1:] == k[:-1]
    v = (np.hypot(np.diff(cx), np.diff(cy)) / np.diff(t) * 1e3)[selbe]
    kv = k[1:][selbe]
    ku, start = np.unique(kv, return_index=True)
    med = np.array([np.median(a) for a in np.split(v, start[1:])])
    aus = np.full(len(ids), np.nan)
    pos = np.clip(np.searchsorted(ku, ids), 0, max(len(ku) - 1, 0))
    ok = (len(ku) > 0) & (ku[pos] == ids)
    aus[ok] = med[pos[ok]]
    return aus


def auswerten(r, solves, stamm):
    with np.load(config.label_file(SPLIT, r)) as g:
        m = polarity_mask(g["p"], "on")
        t = g["t"][m].astype(np.int64)
        gt = g["track_id"][m].astype(np.int64)
        st = g["status"][m]
    with np.load(Path(solves) / STAMM.format(r=r, stamm=stamm)) as d:
        lab = d["trajektorie_id"].astype(np.int64)
    if len(lab) != len(t):
        raise SystemExit(f"run{r}: {len(t):,} ON-Events, {len(lab):,} in der Partition")
    bb = load_bbox(SPLIT, r)
    delta = gt_raster_dt(bb)
    ist_gt = st != STATUS_NOISE

    g_s, lab_s, b_s = _einheiten(lab, gt, t, ist_gt, delta)
    ids, inv = np.unique(g_s, return_inverse=True)
    # IDSW wie in idsw_metrics: Vergleich mit der vorigen VERFOLGTEN Einheit.
    v = np.flatnonzero(lab_s >= 0)
    w = (g_s[v[1:]] == g_s[v[:-1]]) & (lab_s[v[1:]] != lab_s[v[:-1]])
    wer = inv[v[1:]][w]
    idsw = np.bincount(wer, minlength=len(ids))
    alt, neu = lab_s[v[:-1]][w], lab_s[v[1:]][w]
    b_alt, b_neu = b_s[v[:-1]][w], b_s[v[1:]][w]

    # Zeitliche Ausdehnung jeder Trajektorie ueber ALLE ihre Events, im
    # selben Raster wie _einheiten.
    zug = lab >= 0
    b_ev = (t[zug] - t[ist_gt].min()) // int(delta)
    l_u, l_i = np.unique(lab[zug], return_inverse=True)
    erster = np.full(len(l_u), np.iinfo(np.int64).max)
    letzter = np.full(len(l_u), np.iinfo(np.int64).min)
    np.minimum.at(erster, l_i, b_ev)
    np.maximum.at(letzter, l_i, b_ev)
    bruch = ((letzter[np.searchsorted(l_u, alt)] <= b_neu + TOL)
             & (erster[np.searchsorted(l_u, neu)] >= b_alt - TOL))
    luecke = (b_neu - b_alt - 1) * delta / 1e3                      # ms

    # Was liegt in der Luecke? Events des Partikels zwischen den beiden
    # verfolgten Rasterpunkten -- alle grau, sonst waere dort ein Punkt
    # verfolgt. Keine Events heisst: das Partikel schweigt.
    t0 = t[ist_gt].min()
    n_b = int((t[ist_gt].max() - t0) // int(delta)) + 2
    schl = np.sort(np.searchsorted(ids, gt[ist_gt]) * n_b
                   + (t[ist_gt] - t0) // int(delta))
    in_luecke = (np.searchsorted(schl, wer * n_b + b_neu)
                 - np.searchsorted(schl, wer * n_b + b_alt + 1))

    pe = np.searchsorted(ids, gt[ist_gt])
    n_ev = np.bincount(pe, minlength=len(ids))
    grau = np.bincount(pe, weights=lab[ist_gt] < 0, minlength=len(ids))
    einheiten = np.bincount(inv, minlength=len(ids))
    vmed = geschwindigkeit(bb, ids)

    # Wem gehoeren die beiden Bahnen eines Wechsels? Mehrheitspartikel je
    # Bahn ueber ihre Events ausser Rauschen, dieselbe Menge wie in der
    # IDSW-Rechnung. eigen: beide gehoeren dem wechselnden Partikel, es
    # springt zwischen zwei eigenen Stuecken. Sonst ist ein anderes
    # Partikel beteiligt.
    n_i = len(ids)
    m = zug & ist_gt
    k, c = np.unique(lab[m] * n_i + np.searchsorted(ids, gt[m]),
                     return_counts=True)
    k = k[np.lexsort((c, k // n_i))]
    letzt = np.r_[k[1:] // n_i != k[:-1] // n_i, True]
    maj_l, maj_p = k[letzt] // n_i, k[letzt] % n_i
    eigen = ((maj_p[np.searchsorted(maj_l, alt)] == wer)
             & (maj_p[np.searchsorted(maj_l, neu)] == wer))

    # Rueckkehr: das Partikel war schon einmal in der neuen Bahn. Hin und
    # her zaehlt jedes Mal -- Flackern statt Tausch.
    _, erst, r_i = np.unique(inv[v] * (int(lab_s.max()) + 1) + lab_s[v],
                             return_index=True, return_inverse=True)
    pos = np.flatnonzero(w) + 1
    rueck = erst[r_i[pos]] < pos

    # Ueberlapp: mehrdeutige Events des Partikels im Rasterpunkt des
    # Wechsels. Mehrdeutig ist ein Event im Kern einer Ueberlappung, seine
    # track_id ist dort geschaetzt. grund: dasselbe fuer alle verfolgten
    # Rasterpunkte, als Vergleich.
    amb = st == STATUS_AMBIG
    schl_a = np.unique(np.searchsorted(ids, gt[amb]) * n_b
                       + (t[amb] - t0) // int(delta))
    ueberlapp = np.isin(wer * n_b + b_neu, schl_a)
    grund = np.isin(inv[v] * n_b + b_s[v], schl_a)
    idsw_eind = idsw_metrics(np.where(amb, -1, lab), gt, st, t,
                             delta=delta).get("idsw", 0)
    mehrd = (np.bincount(pe, weights=amb[ist_gt], minlength=n_i)
             / np.maximum(n_ev, 1))

    anker = idsw_metrics(lab, gt, st, t, delta=delta).get("idsw", 0)
    print(f"run{r}: {len(ids)} Partikel, IDSW {int(idsw.sum())} "
          f"(idsw_metrics {anker})", flush=True)
    return dict(idsw=idsw, v=vmed, rate=n_ev / (einheiten * delta / 1e3),
                grau=grau / np.maximum(n_ev, 1), bruch=bruch, luecke=luecke,
                in_luecke=in_luecke, v_w=vmed[wer], eigen=eigen, rueck=rueck,
                ueberlapp=ueberlapp, grund=grund, mehrd=mehrd,
                idsw_eind=np.array([idsw_eind]))


def tabelle(titel, a):
    n_w = int(a["idsw"].sum())
    lb = a["luecke"][a["bruch"]]
    print(f"\n{titel}: {len(a['v'])} Partikel, {n_w} IDSW, davon "
          f"{a['bruch'].mean() * 100:.0f} % Bruch; Luecke beim Bruch Median "
          f"{np.median(lb) if len(lb) else 0:.0f} ms, 90 % unter "
          f"{np.percentile(lb, 90) if len(lb) else 0:.0f} ms")
    mit = a["bruch"] & (a["luecke"] > 0)
    ev = a["in_luecke"][mit]
    gl = ev > 0
    print(f"  Brueche mit Luecke: {mit.sum()}, davon still "
          f"{np.mean(ev == 0) * 100 if len(ev) else 0:.0f} %, grau "
          f"{np.mean(gl) * 100 if len(ev) else 0:.0f} %; in grauen Luecken "
          f"im Median {np.median(ev[gl] / a['luecke'][mit][gl]) if gl.any() else 0:.1f}"
          f" Events je ms")
    print(f"  {'|v| px/ms':>11} {'Partikel':>8} {'IDSW':>6} {'Anteil':>7} "
          f"{'je Ptkl':>7} {'Rate':>6} {'grau':>6} {'Bruch':>6} {'Luecke':>7} {'still':>6}"
          f" {'fremd':>6}")
    for lo, hi in zip(V_GRENZEN[:-1], V_GRENZEN[1:]):
        k = (a["v"] >= lo) & (a["v"] < hi)
        kw = (a["v_w"] >= lo) & (a["v_w"] < hi)
        br = a["bruch"][kw]
        lz = a["luecke"][kw][br]
        st = a["in_luecke"][kw][br][lz > 0] == 0
        print(f"  {lo:4.1f}-{hi:<5.1f} {k.sum():8d} {a['idsw'][k].sum():6d} "
              f"{a['idsw'][k].sum() / max(n_w, 1) * 100:6.1f}% "
              f"{a['idsw'][k].mean() if k.any() else 0:7.2f} "
              f"{np.median(a['rate'][k]) if k.any() else 0:6.0f} "
              f"{np.median(a['grau'][k]) * 100 if k.any() else 0:5.1f}% "
              f"{br.mean() * 100 if len(br) else 0:5.0f}% "
              f"{np.median(lz) if len(lz) else 0:5.0f}ms "
              f"{st.mean() * 100 if len(st) else 0:5.0f}% "
              f"{(~br & ~a['eigen'][kw]).mean() * 100 if len(br) else 0:5.0f}%")
    k = a["rate"] < RATE_GRENZE
    print(f"  unter {RATE_GRENZE:.0f} Events/ms: {k.mean() * 100:.1f} % der "
          f"Partikel, {a['idsw'][k].sum() / max(n_w, 1) * 100:.1f} % der IDSW")
    print(f"  {'Art':<16} {'IDSW':>6} {'Anteil':>7} {'Rueckkehr':>10} {'Ueberlapp':>10}")
    for name, sel in (("Bruch", a["bruch"]),
                      ("eigenes Stueck", ~a["bruch"] & a["eigen"]),
                      ("fremd beteiligt", ~a["bruch"] & ~a["eigen"])):
        print(f"  {name:<16} {sel.sum():6d} "
              f"{sel.sum() / max(n_w, 1) * 100:6.1f}% "
              f"{a['rueck'][sel].mean() * 100 if sel.any() else 0:9.0f}% "
              f"{a['ueberlapp'][sel].mean() * 100 if sel.any() else 0:9.0f}%")
    k = a["mehrd"] >= 0.5
    print(f"  Ueberlapp-Grundrate {a['grund'].mean() * 100:.1f} % der verfolgten "
          f"Rasterpunkte; ohne mehrdeutige Events IDSW "
          f"{int(a['idsw_eind'].sum())} statt {n_w}")
    print(f"  ueberwiegend mehrdeutig: {k.sum()} Partikel ({k.mean() * 100:.1f} %) "
          f"mit {a['idsw'][k].sum() / max(n_w, 1) * 100:.1f} % der IDSW")
    k = a["v"] > 2.5 * np.nanmedian(a["v"])
    print(f"  schneller als 2,5 x Median (grob STELLAs Filter): {k.sum()} Partikel "
          f"({k.mean() * 100:.1f} %) mit {a['idsw'][k].sum() / max(n_w, 1) * 100:.1f} % der IDSW")


def partition_stamm(versatz):
    """Solver-Teil des Partitionsnamens aus [6c], Stufe 1 an ihrer Vorgabe."""
    return f"{Solver().short()}_{Solver2(cost_offset=versatz).short()}"


def main(solves, runs, stamm):
    print(f"Partitionen train_run<r>_{stamm}.npz", flush=True)
    alle = [auswerten(r, solves, stamm) for r in runs]
    for r, a in zip(runs, alle):
        tabelle(f"run{r}", a)
    if len(alle) > 1:
        tabelle("gepoolt", {k: np.concatenate([a[k] for a in alle])
                            for k in alle[0]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("solves")
    ap.add_argument("runs", type=int, nargs="+")
    ap.add_argument("--versatz", type=float, nargs="+", required=True,
                    help="solver2.cost_offset der Partitionen aus [6c]")
    a = ap.parse_args()
    for v in a.versatz:
        main(a.solves, a.runs, partition_stamm(v))