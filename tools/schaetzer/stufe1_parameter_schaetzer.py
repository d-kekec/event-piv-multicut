"""
stufe1_parameter_schaetzer.py -- Parameter der Stufe 1 aus den Daten.

Geschaetzt werden: v_max (aus den GT-Boxen), r_assign, R_Ausdehnung und
r_max (aus Events gegen Boxzentren) sowie die beiden Schranken fuer
alpha1 -- Geometrie ueber Drift/r_max, Speicher ueber die Eventrate.

NICHT geschaetzt: r_ambig und box_w (je eine weitere Messung noetig) und
alles, was nicht aus den Daten allein folgt -- cost_offset, feature_mode,
die Trainingsparameter.

Misst den Abstand jedes Events zum naechsten Partikelzentrum und stellt ihm
ein NULLMODELL gegenueber: Zufallspunkte im selben Bild gegen dieselben
Zentren. Deren Abstandsverteilung ist exakt das, was reines Rauschen
erzeugt -- samt des Abfalls bei grossem r, den ein angenommenes Plateau
falsch beschreibt. Der Ueberschuss der Events darueber IST das Signal.

Damit kommt der Schaetzer ohne Annahme ueber Partikeldichte, Bildgroesse
oder Wolkengroesse aus und gilt ueber Datensaetze hinweg.

Die Boxmitten werden zwischen den GT-Rasterpunkten linear interpoliert.
Bei 1-ms-Raster und ~1,5 px/ms kostet blosses Aufrunden bis zu 0,75 px --
mehr als die halbe Wolke.

Aufruf:  python tools/schaetzer/stufe1_parameter_schaetzer.py <split> <run_nr> [t0_ms] [dt_ms]
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.spatial import cKDTree

from eventpiv.kern import data

UNTER_US = 100        # Unterschritt der Interpolation
BIN_PX = 0.1          # Binbreite des Abstandshistogramms
REF_JE_SLICE = 2000   # Zufallspunkte je Unterschritt (Nullmodell)
LAUF_BINS = 10        # so viele Bins muessen unter dem Rauschen bleiben
SIGNAL_Q = 99.9       # Quantil des Ueberschusses fuer R_Ausdehnung
N_FENSTER = 3         # Stichprobenfenster je Run, gleichmaessig verteilt
FENSTER_MS = 10.0     # Laenge eines Stichprobenfensters
KANTEN = np.arange(0.0, 400.0 + BIN_PX, BIN_PX)   # gemeinsames Histogrammgitter
WOLKE_PX = 8.0        # halbe Kantenlaenge des Bildes im Bewegungsrahmen
SEKTOREN = 8          # Richtungssektoren fuer die Konventionsprobe


# ============================================================
# Messung
# ============================================================

def _zentren(b: np.ndarray) -> np.ndarray:
    return np.stack([b["x"] + b["w"] / 2.0, b["y"] + b["h"] / 2.0], axis=1)


def messe_abstaende(split: str, run_nr: int, t0_us: int, dt_us: int) -> dict:
    """Abstand jedes Events zum naechsten interpolierten Boxzentrum.

    Liefert d (px), p (Polaritaet), w (Boxbreite der naechsten Bahn) und
    die Zahl der Events ohne gueltiges Rasterintervall.
    """
    ev = data.load_events(split, run_nr, t0=t0_us, dt=dt_us)
    b = data.load_bbox(split, run_nr)

    raster = np.unique(b["t"])
    schritt = int(np.median(np.diff(raster)))

    ordnung = np.argsort(b["t"], kind="stable")
    b_s = b[ordnung]
    grenzen = np.searchsorted(b_s["t"], raster)
    grenzen = np.append(grenzen, b_s.size)
    z_s, tid_s, w_s = _zentren(b_s), b_s["track_id"], b_s["w"]

    d_alle, p_alle, w_alle = [], [], []
    vek_alle, dl_alle, dq_alle = [], [], []
    v_alle, phi_alle, ruhig_alle = [], [], []
    ohne_intervall = 0

    rng = np.random.default_rng(0)
    ecke_lo = np.array([float(ev["x"].min()), float(ev["y"].min())])
    ecke_hi = np.array([float(ev["x"].max()), float(ev["y"].max())])
    dref_alle = []

    i0 = int(np.searchsorted(raster, ev["t"].min(), side="right") - 1)
    i1 = int(np.searchsorted(raster, ev["t"].max(), side="right"))
    for i in range(max(i0, 0), min(i1, raster.size - 1)):
        r0, r1 = int(raster[i]), int(raster[i + 1])
        im_intervall = (ev["t"] >= r0) & (ev["t"] < r1)
        if not im_intervall.any():
            continue

        a0, a1 = slice(grenzen[i], grenzen[i + 1]), slice(grenzen[i + 1], grenzen[i + 2])
        gemeinsam, k0, k1 = np.intersect1d(tid_s[a0], tid_s[a1],
                                           return_indices=True)
        if gemeinsam.size == 0:
            ohne_intervall += int(im_intervall.sum())
            continue
        z0, z1 = z_s[a0][k0], z_s[a1][k1]
        w_gem = w_s[a0][k0]

        # Bewegungsrichtung je Bahn, aus demselben Rasterintervall wie die
        # Interpolation. Bahnen, die sich im Intervall gar nicht bewegen,
        # haben keine definierte Richtung und werden markiert statt
        # stillschweigend auf (1,0) gesetzt.
        richt = z1 - z0
        laenge = np.linalg.norm(richt, axis=1)
        ein = richt / np.maximum(laenge, 1e-12)[:, None]
        v_bahn = laenge / (r1 - r0) * 1000.0        # px/ms
        phi_bahn = np.arctan2(ein[:, 1], ein[:, 0])
        ruhig_bahn = laenge <= 1e-12

        ev_i = ev[im_intervall]
        for u0 in range(r0, r1, UNTER_US):
            m = (ev_i["t"] >= u0) & (ev_i["t"] < min(u0 + UNTER_US, r1))
            if not m.any():
                continue
            anteil = (u0 + UNTER_US / 2.0 - r0) / (r1 - r0)
            z = z0 + anteil * (z1 - z0)
            punkte = np.stack([ev_i["x"][m].astype(np.float64),
                               ev_i["y"][m].astype(np.float64)], axis=1)
            baum = cKDTree(z)
            d, j = baum.query(punkte, k=1)
            zufall = rng.uniform(ecke_lo, ecke_hi, size=(REF_JE_SLICE, 2))
            dref_alle.append(baum.query(zufall, k=1)[0])
            d_alle.append(d)
            p_alle.append(ev_i["p"][m])
            w_alle.append(w_gem[j])

            # Verschiebung Event -> Zentrum, einmal in BILD-, einmal in
            # BEWEGUNGSkoordinaten. Die Konventionsfrage (Ecke oder Mitte)
            # haengt an der ersten: ein Fehler dort steht in allen
            # Richtungen gleich. Der Vorwaertsversatz haengt an der
            # zweiten: er dreht sich mit dem Partikel.
            vek = punkte - z[j]
            vek_alle.append(vek)
            dl_alle.append(vek[:, 0] * ein[j, 0] + vek[:, 1] * ein[j, 1])
            dq_alle.append(-vek[:, 0] * ein[j, 1] + vek[:, 1] * ein[j, 0])
            v_alle.append(v_bahn[j])
            phi_alle.append(phi_bahn[j])
            ruhig_alle.append(ruhig_bahn[j])

    return {
        "d": np.concatenate(d_alle) if d_alle else np.empty(0),
        "d_ref": np.concatenate(dref_alle) if dref_alle else np.empty(0),
        "p": np.concatenate(p_alle) if p_alle else np.empty(0, dtype=np.int8),
        "w": np.concatenate(w_alle) if w_alle else np.empty(0),
        "vek": (np.concatenate(vek_alle) if vek_alle
                else np.empty((0, 2))),
        "dl": np.concatenate(dl_alle) if dl_alle else np.empty(0),
        "dq": np.concatenate(dq_alle) if dq_alle else np.empty(0),
        "v_lok": np.concatenate(v_alle) if v_alle else np.empty(0),
        "phi": np.concatenate(phi_alle) if phi_alle else np.empty(0),
        "ruhig": (np.concatenate(ruhig_alle) if ruhig_alle
                  else np.empty(0, bool)),
        "n_events": int(ev.size),
        "ohne_intervall": ohne_intervall,
        "raster_us": schritt,
    }


def messe_v_max(split: str, run_nr: int) -> np.ndarray:
    """|v| je Bahnschritt aus den GT-Boxen, in px/ms.

    Braucht keine Events. Schritte zwischen verschiedenen track_id oder
    mit dt = 0 fallen heraus.
    """
    b = data.load_bbox(split, run_nr)
    z = _zentren(b)
    t_ms = b["t"].astype(np.float64) / 1000.0
    ordnung = np.lexsort((t_ms, b["track_id"]))
    tid, t_s, z_s = b["track_id"][ordnung], t_ms[ordnung], z[ordnung]
    dt = t_s[1:] - t_s[:-1]
    weg = np.linalg.norm(z_s[1:] - z_s[:-1], axis=1)
    gueltig = (tid[1:] == tid[:-1]) & (dt > 0)
    return weg[gueltig] / dt[gueltig]


# ============================================================
# Regeln
# ============================================================

def regel_v_max(v: np.ndarray) -> tuple[float, dict]:
    """v_max = p99.9 der Bahngeschwindigkeiten.

    REGEL: v_max ist ein Quantil, kein Maximum -- so schon in
    Datensatz.md definiert. Das haelt vereinzelte Spruenge der GT
    draussen, ohne sie einzeln aussortieren zu muessen: auf STELLA liegen
    0,0002 % der Schritte zehn- bis dreissigfach ueber dem Rumpf, und
    p99.9 bleibt davon unberuehrt.

    UEBER MEHRERE RUNS gilt das MAXIMUM der Run-Quantile. v_max ist ein
    Wert fuer alle Runs -- er steckt im Hash der Neighborhood-Gruppe --,
    und ein zu kleiner Wert schneidet auf den schnellen Runs echte Paare
    desselben Partikels geometrisch weg.
    """
    return float(np.percentile(v, 99.9)), {
        "p50": float(np.percentile(v, 50)),
        "p99": float(np.percentile(v, 99)),
        "max": float(v.max()),
        "n": int(v.size),
    }

def histogramme(d: np.ndarray, d_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Abstandshistogramme auf dem gemeinsamen Gitter KANTEN.

    Auf festem Gitter, damit sich die Histogramme mehrerer Runs
    AUFADDIEREN lassen. Die Regel laeuft dann einmal ueber die Summe,
    statt je Run zu schaetzen und anschliessend zu mitteln -- ein
    Maximum ueber viele Einzelschaetzungen greift sonst den schlechtesten
    Fehlschlag ab.
    """
    return (np.histogram(d, bins=KANTEN)[0],
            np.histogram(d_ref, bins=KANTEN)[0])


def regel_r_assign(n_ev: np.ndarray, n_ref: np.ndarray) -> tuple[float, dict]:
    """r_assign = wo das Signal unter das Rauschen faellt.

    REGEL: Die Zufallspunkte liefern die Abstandsverteilung des reinen
    Rauschens. Ihr Anteil folgt aus dem FERNBEREICH -- oberhalb des p90
    der Referenzverteilung liegt sicher kein Signal, dort muessen beide
    Histogramme bis auf einen Faktor uebereinstimmen. Dieser Faktor
    skaliert die Referenz zur Rauscherwartung je Bin; was darueber liegt,
    ist Signal.

    r_assign ist der erste Radius jenseits des Signalgipfels, ab dem der
    Ueberschuss ueber LAUF_BINS Bins hinweg KLEIN bleibt. Klein heisst
    unter der Rauscherwartung ODER unter einem Promille des Gipfels: im
    Fernbereich traegt ein Bin ein bis zwei Events bei einer Erwartung um
    0,5, und eine Poisson-Schwankung darf die Grenze nicht nach aussen
    reissen.
    """
    mitte = KANTEN[:-1]
    kum_ref = np.cumsum(n_ref)
    if kum_ref[-1] <= 0 or n_ev.sum() <= 0:
        return float("nan"), {"kanten": KANTEN, "n_ev": n_ev,
                              "rausch": np.zeros_like(mitte),
                              "ueberschuss": np.zeros_like(mitte),
                              "r_fern": float("nan"), "skal": float("nan"),
                              "n_signal": 0.0}

    i_fern = int(np.searchsorted(kum_ref / kum_ref[-1], 0.90))
    r_fern = float(mitte[min(i_fern, mitte.size - 1)])
    fern = mitte >= r_fern
    skal = n_ev[fern].sum() / max(1, n_ref[fern].sum())
    rausch = skal * n_ref
    ueberschuss = np.maximum(n_ev - rausch, 0.0)

    i_gipfel = int(np.argmax(ueberschuss))
    klein = 1e-3 * float(ueberschuss.max())
    unter = (ueberschuss <= rausch) | (ueberschuss <= klein)
    grenze = float("nan")
    for i in range(i_gipfel + 1, unter.size - LAUF_BINS):
        if unter[i:i + LAUF_BINS].all():
            grenze = float(KANTEN[i])
            break

    return grenze, {
        "kanten": KANTEN, "n_ev": n_ev, "rausch": rausch,
        "ueberschuss": ueberschuss, "r_fern": r_fern, "skal": skal,
        "n_signal": float(ueberschuss.sum()),
    }


def regel_r_max(kanten: np.ndarray, ueberschuss: np.ndarray, grenze: float,
                v_max_px_ms: float, alpha_ms: float) -> tuple[float, float]:
    """r_max = R_Ausdehnung + Drift je Fenster.

    REGEL aus Kantenkosten_NN.md §2: R_Ausdehnung ist das p99.9 der
    Signalstreuung, die Drift v_max*alpha kommt dazu. Gerechnet wird auf
    dem RAUSCHBEREINIGTEN Ueberschuss innerhalb r_assign -- ein Quantil
    ueber alle Events traegt sonst den Rauschschwanz mit.

    r_max deckt bewusst nicht jedes Paar eines Partikels ab: Multicut ist
    transitiv, es genuegt, den Bogen zusammenhaengend zu halten.
    """
    innen = kanten[:-1] < grenze
    kum = np.cumsum(ueberschuss[innen])
    if kum[-1] <= 0:
        return float("nan"), float("nan")
    i = int(np.searchsorted(kum / kum[-1], SIGNAL_Q / 100.0))
    r_ausdehnung = float(kanten[min(i + 1, kanten.size - 1)])
    return r_ausdehnung, r_ausdehnung + v_max_px_ms * alpha_ms


def regel_geometrie(r_ausdehnung: float, v_max_px_ms: float,
                    alpha_ms: float) -> dict:
    """r_max je Geometrie, mit umschliessender Kugel als Kostenmass.

    REGEL: Beim ZYLINDER ist der Radius konstant, er muss die Drift des
    Fensters also enthalten -- r_max = R + v_max*alpha. Beim HYPERBOLOID
    waechst der Radius selbst mit Δt, dort ist r_max die TAILLE und die
    Drift kommt aus der Form -- r_max = R. Beides zu addieren zaehlt die
    Drift doppelt.

    Die umschliessende Kugel (Code_Guide §3.1) misst, was der KDTree
    materialisiert: sqrt(2)*r_max beim Zylinder,
    sqrt(2*r_max^2 + (v_max*alpha)^2) beim Hyperboloid. Das Suchvolumen
    geht mit der dritten Potenz.
    """
    drift = v_max_px_ms * alpha_ms
    r_zyl = r_ausdehnung + drift
    q_zyl = np.sqrt(2.0) * r_zyl
    r_hyp = r_ausdehnung
    q_hyp = float(np.sqrt(2.0 * r_hyp ** 2 + drift ** 2))
    return {"drift": drift, "r_zyl": r_zyl, "q_zyl": float(q_zyl),
            "r_hyp": r_hyp, "q_hyp": q_hyp,
            "vol_rel": (q_hyp / q_zyl) ** 3 if q_zyl > 0 else float("nan"),
            "drift_rel": drift / r_zyl if r_zyl > 0 else float("nan")}


# ============================================================
# Bericht
# ============================================================

def _zeige(name: str, v: np.ndarray, einheit: str = "px") -> None:
    if v.size == 0:
        print(f"  {name:26s} keine Werte")
        return
    p = np.percentile(v, (50, 90, 99, 99.9))
    print(f"  {name:26s} n={v.size:>9d}  p50={p[0]:.3f}  p90={p[1]:.3f}  "
          f"p99={p[2]:.3f}  p99.9={p[3]:.3f} {einheit}")


def schaetze_run(split: str, run_nr: int) -> dict:
    """Alle Stufe-1-Parameter EINES Runs, aus N_FENSTER Stichprobenfenstern.

    Die Fenster liegen gleichmaessig ueber den Run verteilt -- ein
    einzelnes traegt laut Journal 06.08. bis zu 32 % Rauschen, wo andere
    0,02 % zeigen. Die Abstaende aller Fenster gehen GEMEINSAM in die
    Regeln, nicht als Mittel von Einzelwerten.
    """
    b = data.load_bbox(split, run_nr)
    raster = np.unique(b["t"])
    t_ende = int(raster[-1])
    dt_us = int(FENSTER_MS * 1000)

    d_l, p_l, w_l, dref_l, n_ev = [], [], [], [], 0
    for k in range(N_FENSTER):
        t0 = int((k + 0.5) / N_FENSTER * (t_ende - dt_us))
        m = messe_abstaende(split, run_nr, t0, dt_us)
        d_l.append(m["d"]); p_l.append(m["p"])
        w_l.append(m["w"]); dref_l.append(m["d_ref"])
        n_ev += m["n_events"]

    d = np.concatenate(d_l); p = np.concatenate(p_l)
    w = np.concatenate(w_l); d_ref = np.concatenate(dref_l)

    n_ev_h, n_ref_h = histogramme(d, d_ref)
    v_max, v_info = regel_v_max(messe_v_max(split, run_nr))
    grenze, info = regel_r_assign(n_ev_h, n_ref_h)
    r_aus, _ = regel_r_max(info["kanten"], info["ueberschuss"], grenze,
                           v_max, 1.0)

    on_anteil = float((p == 1).sum()) / max(1, p.size)
    rate_ges = n_ev / (N_FENSTER * FENSTER_MS * 1000.0)   # MEvents/s
    return {
        "run": run_nr, "v_max": v_max, "v_p50": v_info["p50"],
        "r_assign": grenze, "R": r_aus,
        "rausch": 1.0 - info["n_signal"] / max(1, d.size),
        "on": on_anteil, "rate_on": rate_ges * on_anteil,
        "w_p50": float(np.median(w)), "n": int(d.size),
        "n_ev_h": n_ev_h, "n_ref_h": n_ref_h,
    }


def bericht(split: str, runs: list[int]) -> None:
    print(f"\nSplit {split}, {len(runs)} Runs, je {N_FENSTER} Fenster "
          f"zu {FENSTER_MS} ms")
    print(f"{'run':>5} {'ON%':>6} {'ON-Rate':>9} {'v_max':>8} {'v_p50':>8} "
          f"{'r_assign':>9} {'R':>6} {'w_p50':>6} {'Rausch%':>8}")
    zeilen = []
    for nr in runs:
        z = schaetze_run(split, nr)
        zeilen.append(z)
        print(f"{z['run']:>5} {z['on'] * 100:>6.1f} {z['rate_on']:>9.2f} "
              f"{z['v_max']:>8.4f} {z['v_p50']:>8.4f} {z['r_assign']:>9.2f} "
              f"{z['R']:>6.2f} {z['w_p50']:>6.2f} {z['rausch'] * 100:>8.2f}")

    def spalte(k): return np.array([z[k] for z in zeilen], dtype=float)

    v_max = float(np.nanmax(spalte("v_max")))
    rate_max = float(np.nanmax(spalte("rate_on")))

    # Gepoolt statt gemittelt: ein Histogramm ueber alle Runs, die Regel
    # laeuft einmal darueber. Robust gegen einzelne Fehlschaetzungen.
    n_ev_s = np.sum([z["n_ev_h"] for z in zeilen], axis=0)
    n_ref_s = np.sum([z["n_ref_h"] for z in zeilen], axis=0)
    r_ass, info_s = regel_r_assign(n_ev_s, n_ref_s)
    R, _ = regel_r_max(info_s["kanten"], info_s["ueberschuss"], r_ass,
                       v_max, 1.0)
    print(f"\n  Gepoolt ueber alle Runs:  r_assign={r_ass:.2f} px   "
          f"R_Ausdehnung={R:.2f} px   "
          f"Rausch={100 * (1 - info_s['n_signal'] / max(1, n_ev_s.sum())):.2f} %")

    print("\n  Zusammenfassung -- ein Wert gilt fuer ALLE Runs, also das")
    print("  Maximum: ein zu kleines v_max oder r_max schneidet auf den")
    print("  schnellsten Runs echte Paare geometrisch weg.")
    for k, name in (("v_max", "v_max [px/ms]"), ("r_assign", "r_assign [px]"),
                    ("R", "R_Ausdehnung [px]"), ("rate_on", "ON-Rate [MEv/s]")):
        s = spalte(k)
        print(f"    {name:20s} min={np.nanmin(s):8.4f}  "
              f"median={np.nanmedian(s):8.4f}  max={np.nanmax(s):8.4f}")

    print(f"\n  Kandidaten fuer alpha1  (R={R:.2f}, v_max={v_max:.4f} px/ms, "
          f"max. ON-Rate {rate_max:.1f} MEv/s)")
    print(f"  {'alpha':>7} {'Drift':>7} {'Zyl r':>7} {'Hyp r':>7} "
          f"{'Vol Hyp':>8} {'Knoten':>9} {'Jaccard':>9}")
    for alpha in (0.25, 0.5, 1.0, 2.0, 5.0, 10.0):
        g = regel_geometrie(R, v_max, alpha)
        n_knoten = rate_max * 1e6 * alpha / 1000.0
        gb = n_knoten ** 2 * 4 / 1024 ** 3
        print(f"  {alpha:>7.2f} {g['drift']:>7.2f} {g['r_zyl']:>7.2f} "
              f"{g['r_hyp']:>7.2f} {g['vol_rel'] * 100:>7.0f}% "
              f"{n_knoten:>9.0f} {gb:>8.2f}G")


def messe_wolke(split: str, run_nr: int) -> dict:
    """Wolkenform eines Runs aus N_FENSTER Stichprobenfenstern."""
    b = data.load_bbox(split, run_nr)
    t_ende = int(np.unique(b["t"])[-1])
    dt_us = int(FENSTER_MS * 1000)
    teile = [messe_abstaende(split, run_nr,
                             int((k + 0.5) / N_FENSTER * (t_ende - dt_us)),
                             dt_us)
             for k in range(N_FENSTER)]
    aus = {k: np.concatenate([t[k] for t in teile])
           for k in ("d", "d_ref", "p", "w", "dl", "dq", "v_lok", "phi",
                     "ruhig")}
    aus["vek"] = np.concatenate([t["vek"] for t in teile])
    return aus


def regel_versatz(dl, dq, v_lok, n_gruppen: int = 5) -> list[dict]:
    """Vorwaertsversatz und Querstreuung, nach Geschwindigkeit geschichtet.

    REGEL-KANDIDAT, noch keine Regel. STELLA korrigiert um beta*sigma_quer
    entgegen der Bewegungsrichtung, OHNE Geschwindigkeitsterm; unser
    bisheriges c0 + c1*|u| hat keinen Groessenterm. Diese Tabelle
    entscheidet zwischen beiden: bleibt beta = Versatz/sigma_quer ueber die
    Quintile konstant, traegt STELLAs Form und beta ist direkt ablesbar.
    Waechst es, fehlt ihr der Geschwindigkeitsterm.
    """
    grenzen = np.percentile(v_lok, np.linspace(0, 100, n_gruppen + 1))
    zeilen = []
    for lo, hi in zip(grenzen[:-1], grenzen[1:]):
        m = (v_lok >= lo) & (v_lok <= hi)
        if int(m.sum()) < 100:
            continue
        s_q = float(dq[m].std())
        ver = float(dl[m].mean())
        zeilen.append({"v": float(np.median(v_lok[m])), "n": int(m.sum()),
                       "versatz": ver, "sigma_quer": s_q,
                       "sigma_laengs": float(dl[m].std()),
                       "beta": ver / s_q if s_q > 0 else float("nan")})
    return zeilen


def _bild(dl, dq, spalten: int = 41, zeilen: int = 21) -> None:
    """Grobes Bild der Wolke im Bewegungsrahmen. Rechts = Bewegungsrichtung."""
    h, _, _ = np.histogram2d(dq, dl, bins=[zeilen, spalten],
                             range=[[-WOLKE_PX, WOLKE_PX],
                                    [-WOLKE_PX, WOLKE_PX]])
    rampe = " .:-=+*#%@"
    m = h.max()
    if m <= 0:
        print("    (leer)")
        return
    for r in range(zeilen - 1, -1, -1):
        print("    |" + "".join(
            rampe[min(len(rampe) - 1,
                      int(h[r, c] / m * (len(rampe) - 1) + 0.5))]
            for c in range(spalten)) + "|")


def bericht_wolke(split: str, runs: list[int]) -> None:
    print(f"\nWolkenform, Split {split}, Runs {runs}, je {N_FENSTER} "
          f"Fenster zu {FENSTER_MS} ms")
    teile = [messe_wolke(split, nr) for nr in runs]
    a = {k: np.concatenate([t[k] for t in teile])
         for k in ("d", "d_ref", "p", "w", "dl", "dq", "v_lok", "phi",
                   "ruhig")}
    vek = np.concatenate([t["vek"] for t in teile])

    r_ass, _ = regel_r_assign(*histogramme(a["d"], a["d_ref"]))
    sig = (a["d"] < r_ass) & (~a["ruhig"])
    print(f"  r_assign (gepoolt) {r_ass:.2f} px -- {int(sig.sum()):,} von "
          f"{a['d'].size:,} Events im Signal, {int(a['ruhig'].sum()):,} "
          f"ohne Bewegungsrichtung")
    dlx, dqx, vkx = a["dl"][sig], a["dq"][sig], vek[sig]

    print("\n  KONVENTIONSPROBE -- mittlere Verschiebung Event minus Zentrum.")
    print("  Ein falsch gelesenes Boxzentrum steht in ALLEN Sektoren gleich")
    print("  (Bild dx/dy konstant); ein Vorwaertsversatz dreht sich mit")
    print("  (laengs konstant positiv, Bild dx/dy wechselnd).")
    print(f"  {'Sektor':>8} {'n':>10} {'Bild dx':>9} {'Bild dy':>9} "
          f"{'laengs':>9} {'quer':>9}")
    kanten = np.linspace(-np.pi, np.pi, SEKTOREN + 1)
    idx = np.clip(np.digitize(a["phi"][sig], kanten) - 1, 0, SEKTOREN - 1)
    for s in range(SEKTOREN):
        m = idx == s
        if int(m.sum()) < 100:
            continue
        grad = np.degrees((kanten[s] + kanten[s + 1]) / 2.0)
        print(f"  {grad:>7.0f}° {int(m.sum()):>10,} "
              f"{vkx[m, 0].mean():>9.3f} {vkx[m, 1].mean():>9.3f} "
              f"{dlx[m].mean():>9.3f} {dqx[m].mean():>9.3f}")

    print(f"\n  Wolke im Bewegungsrahmen (rechts = Bewegungsrichtung, "
          f"+/-{WOLKE_PX:.0f} px):")
    _bild(dlx, dqx)
    print(f"    laengs: Mittel {dlx.mean():+.3f}  Median "
          f"{np.median(dlx):+.3f}  sigma {dlx.std():.3f} px")
    print(f"    quer  : Mittel {dqx.mean():+.3f}  Median "
          f"{np.median(dqx):+.3f}  sigma {dqx.std():.3f} px")

    print("\n  Versatz gegen Geschwindigkeit (Quintile):")
    print(f"  {'|u| px/ms':>10} {'n':>11} {'Versatz':>9} {'sig quer':>9} "
          f"{'sig laengs':>11} {'beta':>8}")
    for z in regel_versatz(dlx, dqx, a["v_lok"][sig]):
        print(f"  {z['v']:>10.3f} {z['n']:>11,} {z['versatz']:>9.3f} "
              f"{z['sigma_quer']:>9.3f} {z['sigma_laengs']:>11.3f} "
              f"{z['beta']:>8.3f}")

    print("\n  Nach Polaritaet -- auf einem Datensatz mit beiden Sicheln")
    print("  heben sie sich im Mittel auf, und der Versatz kaeme null")
    print("  heraus, obwohl es ihn je Polaritaet gibt.")
    print(f"  {'p':>3} {'n':>11} {'laengs':>9} {'sig laengs':>11} "
          f"{'sig quer':>9} {'beta':>8}")
    ps = a["p"][sig]
    for pv in np.unique(ps):
        m = ps == pv
        if int(m.sum()) < 100:
            continue
        s_q = float(dqx[m].std())
        print(f"  {int(pv):>3} {int(m.sum()):>11,} {dlx[m].mean():>9.3f} "
              f"{dlx[m].std():>11.3f} {s_q:>9.3f} "
              f"{dlx[m].mean() / s_q if s_q > 0 else float('nan'):>8.3f}")

    print("\n  Nach Boxbreite:")
    ws = a["w"][sig]
    werte = np.unique(ws)
    if werte.size <= 6:
        paare = [(float(v), float(v)) for v in werte]
    else:
        q = np.percentile(ws, (0, 33.3, 66.7, 100))
        paare = [(float(q[i]), float(q[i + 1])) for i in range(3)]
    for lo, hi in paare:
        m = (ws >= lo) & (ws <= hi)
        if int(m.sum()) < 100:
            continue
        r = np.hypot(dlx[m], dqx[m])
        print(f"    w in [{lo:.2f},{hi:.2f}]  n={int(m.sum()):>10,}  "
              f"Versatz {dlx[m].mean():+.3f}  sig quer {dqx[m].std():.3f}  "
              f"R(p99.9) {np.percentile(r, 99.9):.3f} px")


if __name__ == "__main__":
    from eventpiv.kern import config

    argv = [a for a in sys.argv[1:] if a != "wolke"]
    split = argv[0]
    runs = [int(a) for a in argv[1:]] or config.available_runs(split)
    (bericht_wolke if "wolke" in sys.argv else bericht)(split, runs)