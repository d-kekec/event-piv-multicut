#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_trajektorien.py -- Kennzahlen der BAHN-Ebene (Stufe [7]).

  gt_bahnen        Boxmitten und GT-Geschwindigkeit je Partikel
  bahn_metriken    MedAE einer Bahn gegen ihre GT-Bahn
  trajektorien_metriken   ueber alle Bahnen eines Runs
  stella_fehler    Fehler je Punkt nach STELLA §4.1, ein Run
  stella_kennzahlen       MedAE mit P25/P75, t_covered, N_tracks/N_gt

ZWEI FASSUNGEN DERSELBEN FAMILIE, MedAE in Ort und Geschwindigkeit.

trajektorien_metriken ordnet jede Bahn ueber die Labels ihrem
Mehrheitspartikel zu und nimmt den Median ueber Bahn-Mediane: bei
Bahnlaengen ueber mehrere Groessenordnungen dominierten sonst wenige
lange Bahnen. Bezug der Prozentzahl ist v_max. Das ist die Diagnose je
Bahn, die [7] mitschreibt.

stella_fehler und stella_kennzahlen rechnen wie STELLA §4.1: Zuordnung je
Zeitpunkt ueber den Abstand zum naechsten Partikel, ohne Labels; Bahnen
mit Ausreissern in der Geschwindigkeit verworfen; Median ueber ALLE
zugeordneten Punkte, dazu P25 und P75; Bezug ist |U^| = 0,6 px/ms, eine
feste Normierung. Nur diese Fassung ist mit den Zahlen des Papiers
vergleichbar.

DIE GT LIEFERT NUR ORTE. Die Geschwindigkeit muss abgeleitet werden, und
eine zentrale Differenz ueber einen GT-Schritt traegt zu viel
Rasterrauschen: die Boxmitte ist auf 0,1 px aufgeloest, ueber 16,7 ms sind
das rund 0,0012 px/ms gegen ein Signal von 0,05. Gerechnet wird deshalb
eine Ausgleichsgerade ueber 2*gt_fenster+1 Rasterpunkte.

gt_fenster ist ein KOMPROMISS. Zu klein: die Referenz rauscht und
verdeckt, was gemessen werden soll. Zu gross: sie glaettet selbst und
bevorzugt einen zu traegen Filter. Die Spanne haengt am GT-Raster und
ist je Datensatz zu pruefen.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["gt_bahnen", "bahn_metriken", "trajektorien_metriken",
           "stella_fehler", "stella_kennzahlen"]

GT_FENSTER = 5          # +-Rasterpunkte (11 Punkte), Spanne je GT-Raster
MIN_PUNKTE = 3          # weniger tragen keine Ausgleichsgerade
REIN_MIN = 0.99         # ab hier gilt eine Bahn als unverschmolzen

# STELLA §4.1 und §3.1
R_ZUORDNUNG = 5.0       # px, Zuordnung je Zeitpunkt
FAKTOR_AUSREISSER = 2.5 # mal Median von |u|: Bahnen darueber fallen weg
U_NORM_STELLA = 0.6     # px/ms, |U^| = 600 px/s


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
    unverschmolzenen. Der Abstand der beiden Mediane bleibt klein, weil
    wenige verschmolzene Bahnen einen Median kaum bewegen; was eine
    Verschmelzung kostet, zeigt die Zahl der Bahnen unter REIN_MIN.

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

def _gt_punkte(bb, gt_fenster: int):
    """Alle GT-Punkte eines Runs, auch kurzer Bahnen, nach Zeit sortiert.

    Rueckgabe (t, cx, cy, gu, gv, k): Boxmitte, Geschwindigkeit in px/µs
    aus gt_bahnen und track_id. Bahnen unter 2*gt_fenster+1 Boxen tragen
    NaN als Geschwindigkeit -- ihr Ort zaehlt fuer Zuordnung und
    Abdeckung, ihre Geschwindigkeit ist nicht bestimmbar.
    """
    t = bb["t"].astype(np.float64)
    cx = bb["x"].astype(np.float64) + bb["w"].astype(np.float64) / 2
    cy = bb["y"].astype(np.float64) + bb["h"].astype(np.float64) / 2
    k = bb["track_id"].astype(np.int64)
    gu = np.full(len(t), np.nan)
    gv = np.full(len(t), np.nan)
    if len(t):
        # Je Bahn zeitlich sortiert, dieselbe Ordnung wie in gt_bahnen.
        ordn = np.lexsort((t, k))
        ks = k[ordn]
        anfang = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
        lage = dict(zip(ks[anfang].tolist(),
                        zip(anfang, np.r_[anfang[1:], len(ks)])))
        for kk, (_, _, _, guk, gvk) in gt_bahnen(bb, gt_fenster).items():
            a, e = lage[kk]
            gu[ordn[a:e]] = guk
            gv[ordn[a:e]] = gvk
    z = np.argsort(t, kind="stable")
    return t[z], cx[z], cy[z], gu[z], gv[z], k[z]


def stella_fehler(spalten, bb, *, r_zuordnung: float = R_ZUORDNUNG,
                  faktor: float = FAKTOR_AUSREISSER,
                  gt_fenster: int = GT_FENSTER) -> dict:
    """Fehler je Punkt nach STELLA §4.1, ein Run. Reine Funktion.

    spalten  (t, x, y, ux, uy, trajektorie_id) aus kette.fitte_run, jede
             Bahn zusammenhaengend, u in px/µs
    bb       Boxarray des Runs

    Die Auslegung, wo das Papier offen ist:

    AUSREISSER: verworfen wird eine Bahn, deren hoechstes |u| das
    faktor-Fache des Medians von |u| ueber alle Punkte des Runs
    uebersteigt. Das Papier nennt Bahnen, die solche Werte ENTHALTEN, also
    das Maximum je Bahn; der Median je Run, weil jede Sequenz ihr eigenes
    Feld traegt.

    ZUORDNUNG je GT-Zeitpunkt: die Bahn wird dorthin interpoliert, wie in
    trajektorien_metriken, und dem naechsten Partikel dieser Zeit
    zugeordnet, wenn es naeher als r_zuordnung liegt. Die Zuordnung darf
    entlang einer Bahn wechseln -- ohne Labels geht es nicht anders. Die
    GT-Zeitpunkte teilen sich alle Partikel, die Boxen stammen aus Bildern.

    ABDECKUNG: ein GT-Punkt gilt als abgedeckt, wenn ihm zu seiner Zeit
    mindestens ein Bahnpunkt zugeordnet ist. Auf einem festen Raster ist
    der Anteil der Punkte zugleich der Anteil der Dauer, t_covered.

    Rueckgabe: Fehler je zugeordnetem Punkt (d_xy in px, d_u in px/ms,
    d_laengs in px) und die Zaehler fuer stella_kennzahlen. d_u fehlt an
    Punkten, deren GT-Bahn keine Geschwindigkeit traegt. d_laengs ist der
    Ortsfehler mit Vorzeichen in GT-Bewegungsrichtung, positiv heisst
    VORAUS: ein konstanter Versatz der Positionskorrektur zeigt sich dort
    und nicht in d_u.
    """
    t, x, y, ux, uy, tid = (np.asarray(a) for a in spalten)
    tg, cxg, cyg, gug, gvg, kg = _gt_punkte(bb, gt_fenster)
    leer = np.empty(0)
    aus = {"d_xy": leer, "d_u": leer, "d_laengs": leer,
           "n_punkte": 0, "n_punkte_zugeordnet": 0,
           "n_gt": int(len(np.unique(kg))), "n_gt_punkte": int(len(tg)),
           "n_gt_gedeckt": 0, "n_bahnen": 0, "n_bahnen_verworfen": 0,
           "u_median": float("nan"), "u_schwelle": float("nan")}
    if len(t) == 0 or len(tg) == 0:
        return aus

    anf = np.flatnonzero(np.r_[True, tid[1:] != tid[:-1]])
    if len(np.unique(tid[anf])) != len(anf):
        raise ValueError("trajektorie_id liegt nicht gruppiert -- jede Bahn "
                         "muss zusammenhaengend stehen, wie aus fitte_run")
    end = np.r_[anf[1:], len(tid)]
    betrag = np.hypot(ux, uy) * 1e3                       # px/ms
    u_median = float(np.median(betrag))
    behalten = np.maximum.reduceat(betrag, anf) <= faktor * u_median
    aus.update(n_bahnen=int(len(anf)),
               n_bahnen_verworfen=int((~behalten).sum()),
               u_median=u_median, u_schwelle=float(faktor * u_median))

    zeiten, za = np.unique(tg, return_index=True)         # tg ist sortiert
    ze = np.r_[za[1:], len(tg)]
    zi, xq, yq, uq, vq = [], [], [], [], []
    for a, e in zip(anf[behalten], end[behalten]):
        ta = t[a:e]
        lo = np.searchsorted(zeiten, ta[0], side="left")
        hi = np.searchsorted(zeiten, ta[-1], side="right")
        if hi <= lo:
            continue
        tq = zeiten[lo:hi]
        zi.append(np.arange(lo, hi))
        xq.append(np.interp(tq, ta, x[a:e]))
        yq.append(np.interp(tq, ta, y[a:e]))
        uq.append(np.interp(tq, ta, ux[a:e]))
        vq.append(np.interp(tq, ta, uy[a:e]))
    if not zi:
        return aus
    zi, xq, yq, uq, vq = (np.concatenate(w) for w in (zi, xq, yq, uq, vq))

    # Ein Baum je GT-Zeitpunkt: er traegt nur die Partikel dieser Zeit,
    # "wer liegt am naechsten" stellt sich nie ueber Zeiten hinweg.
    d = np.full(len(zi), np.inf)
    j = np.zeros(len(zi), np.int64)
    ordn = np.argsort(zi, kind="stable")
    zs = zi[ordn]
    grenzen = np.flatnonzero(np.r_[True, zs[1:] != zs[:-1], True])
    for s, e in zip(grenzen[:-1], grenzen[1:]):
        a, b = za[zs[s]], ze[zs[s]]
        q = ordn[s:e]
        baum = cKDTree(np.column_stack([cxg[a:b], cyg[a:b]]))
        dq, jq = baum.query(np.column_stack([xq[q], yq[q]]))
        d[q] = dq
        j[q] = a + jq

    zu = d < r_zuordnung
    jz = j[zu]
    du = np.hypot(uq[zu] - gug[jz], vq[zu] - gvg[jz]) * 1e3
    g = np.hypot(gug[jz], gvg[jz])
    ok = np.isfinite(g) & (g > 0)
    laengs = ((xq[zu] - cxg[jz]) * gug[jz]
              + (yq[zu] - cyg[jz]) * gvg[jz])[ok] / g[ok]
    gedeckt = np.zeros(len(tg), bool)
    gedeckt[jz] = True
    aus.update(d_xy=d[zu], d_u=du[np.isfinite(du)], d_laengs=laengs,
               n_punkte=int(len(zi)), n_punkte_zugeordnet=int(zu.sum()),
               n_gt_gedeckt=int(gedeckt.sum()))
    return aus


def stella_kennzahlen(fehler, u_norm: float = U_NORM_STELLA) -> dict:
    """Kennzahlen nach STELLA §4.1 aus stella_fehler. Reine Funktion.

    fehler: ein Ergebnis von stella_fehler oder eine Liste davon. Ueber
    mehrere Runs wird GEPOOLT: der Median ueber alle Punkte, nicht der
    Median der Run-Mediane; die Zaehler werden summiert.

    medae_*, p25_*, p75_*  Median und Quartile des absoluten Fehlers, bei
                           STELLA die Fehlerbalken
    medae_u_rel            medae_u / u_norm
    laengs_median          Median des Laengsfehlers mit Vorzeichen, px
    t_covered              abgedeckte GT-Punkte / alle GT-Punkte
    n_tracks_je_gt         behaltene Bahnen / GT-Bahnen. Ueber 1 heisst
                           unterbrochen oder ohne Partner, unter 1 nicht
                           abgedeckt
    verworfen_frac         Anteil der Bahnen, den der Ausreisserfilter nahm
    zugeordnet_frac        Anteil der Bahnpunkte mit Partner
    """
    if isinstance(fehler, dict):
        fehler = [fehler]

    def _cat(k):
        return np.concatenate([np.asarray(f[k], np.float64) for f in fehler]
                              + [np.empty(0)])

    def _sum(k):
        return sum(int(f[k]) for f in fehler)

    def _quot(a, b):
        return float(a) / b if b else float("nan")

    def _quartile(w):
        if len(w) == 0:
            return (float("nan"),) * 3
        return tuple(float(v) for v in np.percentile(w, [25, 50, 75]))

    p25_xy, med_xy, p75_xy = _quartile(_cat("d_xy"))
    p25_u, med_u, p75_u = _quartile(_cat("d_u"))
    laengs = _cat("d_laengs")
    n_tracks = _sum("n_bahnen") - _sum("n_bahnen_verworfen")
    return {"n_punkte_zugeordnet": _sum("n_punkte_zugeordnet"),
            "medae_xy": med_xy, "p25_xy": p25_xy, "p75_xy": p75_xy,
            "medae_u": med_u, "p25_u": p25_u, "p75_u": p75_u,
            "medae_u_rel": _quot(med_u, u_norm),
            "laengs_median": (float(np.median(laengs)) if len(laengs)
                              else float("nan")),
            "t_covered": _quot(_sum("n_gt_gedeckt"), _sum("n_gt_punkte")),
            "n_tracks": n_tracks, "n_gt": _sum("n_gt"),
            "n_tracks_je_gt": _quot(n_tracks, _sum("n_gt")),
            "verworfen_frac": _quot(_sum("n_bahnen_verworfen"),
                                    _sum("n_bahnen")),
            "zugeordnet_frac": _quot(_sum("n_punkte_zugeordnet"),
                                     _sum("n_punkte"))}