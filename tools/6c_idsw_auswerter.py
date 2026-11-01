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

Fuer die fremd beteiligten Wechsel dazu der Mechanismus: In welchem
Stufe-1-Tracklet stecken die Events des Partikels an dem Rasterpunkt, an
dem es in der fremden Bahn liegt? Jedes dieser Events bekommt die Klasse
seines Tracklets, nach dessen Zusammensetzung ohne Rauschen:
  rein        eigener Anteil ab REIN_AB
  gemischt    eigene Mehrheit, aber darunter
  fremd       die Mehrheit gehoert einem anderen Partikel
Der Wechsel zaehlt zur Klasse mit den meisten Events, bei Gleichstand zur
kleineren. Wie das Ergebnis zu lesen ist, steht in LESART; die Auswertung
gibt es am Ende aus.

Gegenprobe, je Run gedruckt: die IDSW-Summe muss der Zahl aus
idsw_metrics gleichen, wie [6c] sie rechnet, die Minderheit in den
Stufe-1-Knoten dem Feld stufe1/merge_n_minderheit derselben [6c]-JSON.
Die zweite prueft, dass (A) richtig gelesen wird.

Aufruf ohne Hydra; EVENTPIV_DATA und EVENTPIV_OUT wie in scripts/_cli.py.
<solves> ist das Verzeichnis der Partitionen aus [6c], dessen Dateien
solve_stufe2 hinter "->" meldet. Gelesen werden nur train-Runs (SPLIT),
dazu ihr (A) mit Stufe 1 an der Vorgabe, Labels und Boxen. --versatz
nennt solver2.cost_offset der Partitionen, mehrere Werte nacheinander:
    python tools/6c_idsw_auswerter.py <solves> 8 13 --versatz 1
    python tools/6c_idsw_auswerter.py <solves> 8 13 --versatz 0.5 1
"""
import argparse
import json
from pathlib import Path

import numpy as np

from eventpiv.bewerten.evaluate import (_einheiten, gt_raster_dt, idsw_metrics,
                                        merge_metrics)
from eventpiv.kern import config
from eventpiv.kern.data import load_bbox
from eventpiv.kern.graph import polarity_mask
from eventpiv.kern.runconfig import Solver, Solver2, Transformer
from eventpiv.kern.status import STATUS_AMBIG, STATUS_NOISE

SPLIT = "train"
STAMM = "train_run{r}_{stamm}.npz"                # Partition aus [6c]
V_GRENZEN = (0.0, 0.1, 0.2, 0.4, 0.8, np.inf)     # px/ms
# 15 Events je 0,5 ms. Fest: folgt weder alpha1 noch min_events.
RATE_GRENZE = 30.0                                # Nachweisgrenze, Events/ms
TOL = 2                                           # Rasterpunkte Spiel beim Bruch
REIN_AB = Transformer().gt_min_anteil             # ab hier gilt ein Knoten als rein
KLASSEN = ("rein", "gemischt", "fremd")           # Stufe-1-Tracklet, Gleichstand: kleinere

LESART = f"""
Lesart Mechanismus. Gezaehlt sind die fremd beteiligten Wechsel, nach dem
Stufe-1-Tracklet, in dem die Events des Partikels am Rasterpunkt in der
fremden Bahn stecken:
  rein      eigener Anteil ab {REIN_AB:g}. Stufe 1 hat richtig getrennt, Stufe 2
            hat ein reines Stueck an die fremde Bahn gehaengt. Ursache: das
            Verbinden in Stufe 2.
  gemischt  eigene Mehrheit unter {REIN_AB:g}. Stufe 1 hat vermischt, Stufe 2
            hat den Knoten zur fremden Bahn gelegt. Beide beteiligt.
  fremd     Mehrheit ein anderes Partikel. Das Partikel steckt als
            Minderheit im Tracklet des anderen, Stufe 2 hat den Knoten nach
            seiner Mehrheit richtig verbunden. Ursache: Stufe 1 -- kein Netz
            in Stufe 2 kann einen Knoten teilen.
Grundrate: dieselbe Einteilung ueber alle verfolgten Rasterpunkte, also
wie oft ein Partikel ueberhaupt ueber ein solches Tracklet laeuft.
Liegen fremd und gemischt an den Wechseln weit ueber ihrer Grundrate,
tragen die Stufe-1-Knoten die Begegnungen, und ein negativer
Stufe-1-Versatz ist der Hebel. Liegt rein vorn, entstehen sie beim
Verbinden, und der Hebel liegt in Stufe 2.
Vorbehalt: mehrdeutige Events zaehlen mit ihrer geschaetzten track_id --
im Ueberlapp ist auch die Zusammensetzung eines Tracklets geschaetzt.
"""


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


def lade_tracklets(r, solves, n_on):
    """tracklet_id aus (A), ein Eintrag je ON-Event wie in der Partition.

    (A) liegt in dem tracklets-Verzeichnis, unter dem
    stufe2/<kd>/<tf>/solves haengt; Stufe 1 an ihrer Vorgabe wie in
    partition_stamm.
    """
    pfad = Path(solves).parents[3] / f"{SPLIT}_run{r}_{Solver().short()}.npz"
    with np.load(pfad) as d:
        tid = d["tracklet_id"].astype(np.int64)
    if len(tid) != n_on:
        raise SystemExit(f"run{r}: {n_on:,} ON-Events, {len(tid):,} in {pfad.name}")
    return tid


def mehrheitsklasse(schl, kl, anfrage):
    """Je Schluessel aus anfrage die Klasse mit den meisten Events.

    schl und kl gehoeren zusammen: Schluessel und Klasse je Event.
    Gleichstand: die kleinere Klasse, also rein vor gemischt vor fremd --
    gegen die Vermutung, die geprueft wird.
    """
    n = np.zeros((len(anfrage), len(KLASSEN)), np.int64)
    for c in range(len(KLASSEN)):
        s = np.sort(schl[kl == c])
        n[:, c] = (np.searchsorted(s, anfrage, "right")
                   - np.searchsorted(s, anfrage, "left"))
    return n.argmax(axis=1)


def auswerten(r, solves, stamm):
    with np.load(config.label_file(SPLIT, r)) as g:
        m = polarity_mask(g["p"], "on")
        t = g["t"][m].astype(np.int64)
        gt = g["track_id"][m].astype(np.int64)
        st = g["status"][m]
    teil = Path(solves) / STAMM.format(r=r, stamm=stamm)
    with np.load(teil) as d:
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

    # Mechanismus: Zusammensetzung jedes Stufe-1-Tracklets ohne Rauschen,
    # Mehrheitspartikel und dessen Anteil. Dann je zugeordnetem Event die
    # Klasse seines Tracklets aus Sicht SEINES Partikels (0 rein,
    # 1 gemischt, 2 fremd).
    tid = lade_tracklets(r, solves, len(t))
    ev = np.flatnonzero(zug & ist_gt)                 # zugeordnet, ohne Rauschen
    if (tid[ev] < 0).any():
        raise SystemExit(f"run{r}: zugeordnete Events ohne Stufe-1-Tracklet")
    mt = (tid >= 0) & ist_gt
    k, c = np.unique(tid[mt] * n_i + np.searchsorted(ids, gt[mt]),
                     return_counts=True)
    o = np.lexsort((c, k // n_i))
    k, c = k[o], c[o]
    letzt = np.r_[k[1:] // n_i != k[:-1] // n_i, True]
    t_u, t_maj = k[letzt] // n_i, k[letzt] % n_i
    t_ant = c[letzt] / np.bincount(np.searchsorted(t_u, k // n_i), weights=c)
    p_ev = np.searchsorted(ids, gt[ev])
    t_i = np.searchsorted(t_u, tid[ev])
    kl = np.where(t_maj[t_i] != p_ev, 2,
                  np.where(t_ant[t_i] < REIN_AB, 1, 0)).astype(np.int8)

    # Je Wechsel der Rasterpunkt, an dem das Partikel in der fremden Bahn
    # liegt: nach dem Wechsel, wenn die neue Bahn fremd ist, sonst davor.
    # Gefragt wird nach (Partikel, Rasterpunkt, Bahn) -- die Bahn ist dort
    # die Mehrheitsbahn des Partikels, hat also Events. grund_mech: dasselbe
    # fuer alle verfolgten Rasterpunkte.
    n_l = int(lab.max()) + 1
    schl_m = (p_ev * n_b + (t[ev] - t0) // int(delta)) * n_l + lab[ev]
    neu_fremd = maj_p[np.searchsorted(maj_l, neu)] != wer
    f_lab = np.where(neu_fremd, neu, alt)
    f_bin = np.where(neu_fremd, b_neu, b_alt)
    kl_q = mehrheitsklasse(schl_m, kl, np.concatenate(
        [(wer * n_b + f_bin) * n_l + f_lab,
         (inv[v] * n_b + b_s[v]) * n_l + lab_s[v]]))
    mech = np.where(eigen, -1, kl_q[:len(wer)])       # -1: keine fremde Bahn
    grund_mech = kl_q[len(wer):]

    # Zweite Gegenprobe: Minderheit in den Stufe-1-Knoten wie in [6c] --
    # Knoten sind Tracklets ab min_events ON-Events, Rauschen eingerechnet.
    with open(teil.with_suffix(".json")) as f:
        j = json.load(f)
    n_on = np.bincount(tid[tid >= 0])
    knoten = (tid >= 0) & (n_on[np.maximum(tid, 0)]
                           >= j["meta"]["kandidaten"]["min_events"])
    m1 = merge_metrics(np.where(knoten, tid, -1), gt, st).get(
        "merge_n_minderheit", 0)

    anker = idsw_metrics(lab, gt, st, t, delta=delta).get("idsw", 0)
    print(f"run{r}: {len(ids)} Partikel, IDSW {int(idsw.sum())} "
          f"(idsw_metrics {anker}); Minderheit in Stufe-1-Knoten {m1} Events "
          f"([6c] {j['lauf']['stufe1']['merge_n_minderheit']})", flush=True)
    return dict(idsw=idsw, v=vmed, rate=n_ev / (einheiten * delta / 1e3),
                grau=grau / np.maximum(n_ev, 1), bruch=bruch, luecke=luecke,
                in_luecke=in_luecke, v_w=vmed[wer], eigen=eigen, rueck=rueck,
                ueberlapp=ueberlapp, grund=grund, mehrd=mehrd,
                idsw_eind=np.array([idsw_eind]), mech=mech,
                grund_mech=grund_mech)


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
    mk, mg = a["mech"][~a["bruch"] & ~a["eigen"]], a["grund_mech"]
    print("  Mechanismus der fremd beteiligten Wechsel, Lesart am Ende:")
    print(f"  {'Stufe-1-Tracklet':<16} {'IDSW':>6} {'Anteil':>7} {'Grundrate':>10}")
    for c, name in enumerate(KLASSEN):
        print(f"  {name:<16} {(mk == c).sum():6d} "
              f"{(mk == c).mean() * 100 if len(mk) else 0:6.1f}% "
              f"{(mg == c).mean() * 100 if len(mg) else 0:9.2f}%")
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
    print(LESART)