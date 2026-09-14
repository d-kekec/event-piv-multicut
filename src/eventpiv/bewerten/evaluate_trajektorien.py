#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_trajektorien.py -- Kennzahlen der BAHN-Ebene (Stufe [7]).

  gt_bahnen        Boxmitten und GT-Geschwindigkeit je Partikel
  bahn_metriken    MedAE einer Bahn gegen ihre GT-Bahn
  trajektorien_metriken   ueber alle Bahnen eines Runs

Dieselbe Familie wie bei STELLA §4.1: MedAE in Ort und Geschwindigkeit,
damit die Zahlen vergleichbar sind. Nicht dieselbe Normierung -- dort ist
|U^| die Amplitude des vorgegebenen Feldes, hier v_max als p99,9 der
Partikelgeschwindigkeit. Die Prozentzahlen sind deshalb NICHT direkt
vergleichbar, die absoluten schon.

DIE GT LIEFERT NUR ORTE. Die Geschwindigkeit muss abgeleitet werden, und
eine zentrale Differenz ueber einen GT-Schritt traegt zu viel
Rasterrauschen: die Boxmitte ist auf 0,1 px aufgeloest, ueber 16,7 ms sind
das rund 0,0012 px/ms gegen ein Signal von 0,05. Gerechnet wird deshalb
eine Ausgleichsgerade ueber 2*gt_fenster+1 Rasterpunkte.

gt_fenster ist ein KOMPROMISS. Zu klein: die Referenz rauscht und
verdeckt, was gemessen werden soll. Zu gross: sie glaettet selbst und
bevorzugt einen zu traegen Filter. Gemessen ueber 1, 3 und 5 wandert der
Arbeitspunkt des Filters NICHT mit -- der Verdacht ist widerlegt, und die
bereinigten Werte bei 3 und 5 stimmen ueberein (Journal 14.09.2026).

MEDIAN UEBER BAHN-MEDIANE, nicht ueber alle Punkte. Bei Bahnlaengen von
21 bis 390 048 Events dominierten sonst wenige lange Bahnen. Das Papier
laesst beides zu; die Wahl steht hier, damit sie nicht je Aufrufer
anders ausfaellt.
"""
from __future__ import annotations

import numpy as np

__all__ = ["gt_bahnen", "bahn_metriken", "trajektorien_metriken"]

GT_FENSTER = 5          # +-Rasterpunkte, also 11 Punkte oder rund 183 ms
MIN_PUNKTE = 3          # weniger tragen keine Ausgleichsgerade
REIN_MIN = 0.99         # ab hier gilt eine Bahn als unverschmolzen


def gt_bahnen(bb, gt_fenster: int = GT_FENSTER) -> dict:
    """Boxmitte und Geschwindigkeit je GT-Partikel. Reine Funktion.

    bb: das strukturierte Array aus data.load_bbox. (x, y) ist die ECKE
    oben links, die Mitte also x + w/2.

    Rueckgabe: {track_id: (t, cx, cy, gu, gv)}, gu/gv in px/µs. Partikel
    mit weniger als 2*gt_fenster+1 Boxen fallen heraus -- ihre
    Geschwindigkeit waere ueberwiegend Rasterrauschen.
    """
    t = bb["t"].astype(np.float64)
    ids = bb["track_id"]
    cx = bb["x"].astype(np.float64) + bb["w"].astype(np.float64) / 2
    cy = bb["y"].astype(np.float64) + bb["h"].astype(np.float64) / 2

    aus = {}
    for k in np.unique(ids):
        m = ids == k
        if int(m.sum()) < 2 * gt_fenster + 1:
            continue
        tk, xk, yk = t[m], cx[m], cy[m]
        ordn = np.argsort(tk)
        tk, xk, yk = tk[ordn], xk[ordn], yk[ordn]
        n = len(tk)
        gu = np.full(n, np.nan)
        gv = np.full(n, np.nan)
        for j in range(n):
            lo, hi = max(0, j - gt_fenster), min(n, j + gt_fenster + 1)
            if hi - lo < MIN_PUNKTE:
                continue
            A = np.vander(tk[lo:hi] - tk[j], 2)
            gu[j] = np.linalg.lstsq(A, xk[lo:hi], rcond=None)[0][0]
            gv[j] = np.linalg.lstsq(A, yk[lo:hi], rcond=None)[0][0]
        aus[int(k)] = (tk, xk, yk, gu, gv)
    return aus


def bahn_metriken(x, y, ux, uy, cx, cy, gu, gv) -> dict | None:
    """MedAE EINER Bahn. Alle Arrays auf denselben GT-Zeitpunkten.

    x, y, ux, uy  Fitausgabe, ux/uy in px/µs
    cx, cy        Boxmitte
    gu, gv        GT-Geschwindigkeit, px/µs, darf NaN enthalten

    Rueckgabe: dict oder None, wenn kein Punkt auswertbar ist.

    du_signiert ist |u_fit| - |u_gt|. Der Betrag der VEKTORdifferenz --
    medae_u -- ist immer positiv und kann einen systematischen Versatz
    nicht von Rauschen unterscheiden.
    """
    d_pos = np.hypot(x - cx, y - cy)
    d_vel = np.hypot(ux - gu, uy - gv)
    v_fit = np.hypot(ux, uy)
    v_gt = np.hypot(gu, gv)
    m = np.isfinite(d_pos) & np.isfinite(d_vel)
    if not m.any():
        return None
    return {"n_punkte": int(m.sum()),
            "medae_xy": float(np.median(d_pos[m])),
            "medae_u": float(np.median(d_vel[m]) * 1e3),
            "du_signiert": float(np.median((v_fit[m] - v_gt[m]) * 1e3)),
            "v_gt": float(np.median(v_gt[m]) * 1e3)}


def trajektorien_metriken(spalten, bb, zuordnung, v_norm,
                          gt_fenster: int = GT_FENSTER) -> dict:
    """Kennzahlen ueber alle Bahnen eines Runs.

    spalten    (t, x, y, ux, uy, trajektorie_id) aus kette.fitte_run
    bb         Boxarray des Runs
    zuordnung  {trajektorie_id: (gt_track_id, reinheit)}
    v_norm     Bezugsgeschwindigkeit in px/ms fuer die Prozentzahlen

    BERICHTET WIRD ZWEIMAL: ueber alle zugeordneten Bahnen und ueber die
    unverschmolzenen. Der Abstand ist selbst eine Aussage -- er misst,
    was ein Clusterfehler den Fit kostet, und nicht der Fit.

    Die Fitausgabe liegt auf fit.dt_aus, die GT auf ihrem eigenen Raster.
    Verglichen wird auf den GT-Zeiten; die Fitwerte werden dorthin
    interpoliert. Das ist zulaessig, weil zwischen zwei GT-Punkten nichts
    passiert, was aus Ort und Geschwindigkeit nicht rekonstruierbar waere
    (gemessen 14.09.2026: unter 1e-7 px).
    """
    t, x, y, ux, uy, tid = spalten
    gt = gt_bahnen(bb, gt_fenster)
    grenzen = np.searchsorted(tid, np.unique(tid))
    grenzen = np.append(grenzen, len(tid))

    zeilen, rein_flags = [], []
    for k in range(len(grenzen) - 1):
        s = slice(grenzen[k], grenzen[k + 1])
        tj = int(tid[s.start])
        if tj not in zuordnung:
            continue
        ptkl, rein = zuordnung[tj]
        if ptkl not in gt:
            continue
        tg, cx, cy, gu, gv = gt[ptkl]
        f = (tg >= t[s][0]) & (tg <= t[s][-1])
        if int(f.sum()) < MIN_PUNKTE:
            continue
        z = bahn_metriken(np.interp(tg[f], t[s], x[s]),
                          np.interp(tg[f], t[s], y[s]),
                          np.interp(tg[f], t[s], ux[s]),
                          np.interp(tg[f], t[s], uy[s]),
                          cx[f], cy[f], gu[f], gv[f])
        if z is None:
            continue
        zeilen.append(z)
        rein_flags.append(rein >= REIN_MIN)

    if not zeilen:
        return {"n_bahnen_bewertet": 0}

    r = np.array(rein_flags, bool)

    def _fasse(idx, suffix=""):
        if not idx.any():
            return {}
        w = {k: np.array([z[k] for z, take in zip(zeilen, idx) if take])
             for k in ("medae_xy", "medae_u", "du_signiert", "v_gt")}
        aus = {f"n_bahnen{suffix}": int(idx.sum()),
               f"medae_xy{suffix}": float(np.median(w["medae_xy"])),
               f"medae_u{suffix}": float(np.median(w["medae_u"])),
               f"du_signiert{suffix}": float(np.median(w["du_signiert"])),
               f"v_gt_median{suffix}": float(np.median(w["v_gt"]))}
        aus[f"medae_u_rel{suffix}"] = (
            aus[f"medae_u{suffix}"] / v_norm if v_norm else float("nan"))
        # Vorzeichentest auf einen systematischen Geschwindigkeitsversatz:
        # bei reinem Rauschen laege der Anteil bei 0,5.
        aus[f"du_neg_frac{suffix}"] = float(
            (w["du_signiert"] < 0).mean())
        return aus

    out = {"n_bahnen_bewertet": len(zeilen)}
    out.update(_fasse(np.ones(len(zeilen), bool)))
    out.update(_fasse(r, "_rein"))
    return out