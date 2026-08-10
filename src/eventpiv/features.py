#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
features.py -- Artefakt (B): ein Tracklet, eine Zeile.

ABGELEITETER SCHRITT, kein Anbau an Stufe 1. Die Tabelle entsteht aus dem
Tracklet-Artefakt plus dem neu geladenen Run; solve_run bleibt unberuehrt.
Grund: eine Aenderung an einem Merkmal kostet so zehn Sekunden je Run statt
eines vollstaendigen Neubaus, und der Anker bleibt gueltig.

NICHT die Transformer-Eingabe -- der bekommt Rohevents als Token. Diese
Tabelle traegt den Kandidatenfilter und die MLP-Ablation.

DER VERTRAG zum Tracklet-Artefakt: `tracklet_id` hat einen Eintrag je
ON-Event des Runs, in der Reihenfolge des polaritaetsgefilterten
Label-Stroms. Wer diese Datei baut, MUSS die Laengen vergleichen -- ein
Versatz um einen Eintrag ist still und verfaelscht jedes Merkmal.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def fenster_index(t, fenster_t0, alpha):
    """Fensternummer je Event. -1 vor dem ersten Fenster.

    Der Treiber schreitet in ANSTOSSENDEN Fenstern der Laenge alpha ab,
    fenster_t0 ist damit ein aequidistantes Raster -- searchsorted genuegt,
    eine Suche je Event ist nicht noetig.
    """
    t = np.asarray(t, dtype=np.float64)
    if len(fenster_t0) == 0:
        return np.full(len(t), -1, np.int64)
    k = np.searchsorted(fenster_t0, t, side="right") - 1
    # Nach dem letzten Fenster: nur gueltig, solange t < t0 + alpha.
    zu_spaet = (k == len(fenster_t0) - 1) & (t >= fenster_t0[-1] + alpha)
    k[zu_spaet] = -1
    return k.astype(np.int64)


def off_zuordnung(t_on, x_on, y_on, tracklet_id, t_off, x_off, y_off, *,
                  s, k=8):
    """OFF-Events auf Tracklets abbilden. EIN Fenster. Reine Funktion.

    OFF-Events sind keine Knoten im Graphen und bekommen deshalb kein
    Cluster-Label. Stufe 2 braucht aber beide Zentroide: D = ON-Zentroid
    minus OFF-Zentroid liefert Richtung und Betrag.

    REGEL: Mehrheit unter den k naechsten ON-Events in der Metrik
    dx^2 + dy^2 + (s*dt)^2, ohne raeumliche Grenze. Eine Grenze zensiert die
    weit entfernten OFF-Events und zieht den Zentroid zur ON-Seite.

    s ist NICHT graph.time_scale. Dort folgt s aus alpha und r_max und
    grenzt einen KOERPER ab; hier bildet es eine RANGFOLGE, und erst dabei
    ist die Gewichtung eine Entscheidung. s = 0,1 heisst 1 px ~ 10 µs --
    faktisch ein Zeitfenster, in dem der Ort entscheidet. Messreihe und
    Begruendung: Journal 09.08.2026.

    GT-frei, und das ist die Bedingung: fuer echte Messdaten gibt es keine
    Boxen. Mit GT waere die Frage ohnehin nicht loesbar -- sie sagt, zu
    welchem PARTIKEL ein OFF-Event gehoert, nicht zu welchem seiner
    Fragmente.

    Rueckgabe: (zuordnung, konsens)
      zuordnung  tracklet_id je OFF-Event, -1 ohne ON-Events im Fenster
      konsens    Anteil der k Nachbarn, die fuer den Sieger stimmten
    """

    n_off = len(t_off)
    if n_off == 0 or len(t_on) == 0:
        return (np.full(n_off, -1, np.int64), np.zeros(n_off, np.float64))

    baum = cKDTree(np.column_stack([np.asarray(x_on, np.float64),
                                    np.asarray(y_on, np.float64),
                                    np.asarray(t_on, np.float64) * s]))
    kk = min(k, len(t_on))
    _, idx = baum.query(np.column_stack([np.asarray(x_off, np.float64),
                                         np.asarray(y_off, np.float64),
                                         np.asarray(t_off, np.float64) * s]),
                        k=kk, workers=-1)
    idx = idx.reshape(n_off, kk)

    # Mehrheit ohne Tabelle ueber alle Tracklets: bei k Nachbarn genuegt
    # der paarweise Vergleich innerhalb der Zeile.
    tid = np.asarray(tracklet_id)
    stimmen = tid[idx]                                   # (n_off, kk)
    zaehl = (stimmen[:, :, None] == stimmen[:, None, :]).sum(axis=2)
    # query liefert nach Abstand sortiert, argmax nimmt die kleinste
    # Position -- bei Gleichstand gewinnt also der naechste.
    best = zaehl.argmax(axis=1)
    zeile = np.arange(n_off)
    return (stimmen[zeile, best].astype(np.int64),
            zaehl[zeile, best] / kk)


def _mehrheit(kompakt, werte_n, schluessel):
    """Haeufigster Schluessel je Gruppe, plus sein Anteil. Ohne Schleife.

    Dieselbe Kreuztabellen-Mechanik wie evaluate._kreuztabelle, hier aber
    je TRACKLET statt je Cluster: Paare (Gruppe, Schluessel) zaehlen, dann
    je Gruppe das Maximum.

    Rueckgabe: (sieger, anteil, n_verschieden). Bei leerer Gruppe
    (-1, 0.0, 0) -- ein Tracklet aus lauter Rauschen hat kein GT-Partikel.
    """
    s_u, s_idx = np.unique(schluessel, return_inverse=True)
    key = kompakt.astype(np.int64) * len(s_u) + s_idx
    kombi, anzahl = np.unique(key, return_counts=True)
    k_grp, k_s = kombi // len(s_u), kombi % len(s_u)

    best = np.zeros(werte_n, np.int64)
    np.maximum.at(best, k_grp, anzahl)
    gesamt = np.bincount(k_grp, weights=anzahl,
                         minlength=werte_n).astype(np.int64)
    n_versch = np.bincount(k_grp, minlength=werte_n)

    sieger = np.full(werte_n, -1, np.int64)
    # Bei Gleichstand gewinnt der groessere Schluessel -- willkuerlich, aber
    # deterministisch, und der Fall ist bei 150+ Events selten.
    treffer = anzahl == best[k_grp]
    sieger[k_grp[treffer]] = s_u[k_s[treffer]]

    anteil = np.divide(best, gesamt, out=np.zeros(werte_n, np.float64),
                       where=gesamt > 0)
    return sieger, anteil, n_versch


def tracklet_features(t_on, x_on, y_on, tracklet_id,
                      t_off, x_off, y_off, off_tracklet, off_konsens,
                      *, fenster_idx, track_id=None, status_noise=None):
    """Eine Zeile je Tracklet. EIN Fenster. Reine Funktion.

    Erwartet ON-Arrays des Fensters mit ihrer tracklet_id sowie die
    OFF-Arrays mit dem Ergebnis von off_zuordnung. Events mit
    tracklet_id < 0 (kantenlose Knoten) fallen heraus -- sie gehoeren zu
    keinem Tracklet.

    track_id und status_noise sind optional und nur beim Training
    vorhanden: status_noise ist eine boolesche Maske ueber die ON-Events
    ("ist Rauschen"). Ohne sie fehlen die gt_-Felder, alles andere bleibt.

    n_events aus dem Entwurf entfaellt: unter polarity="on" ist es
    identisch mit n_on, und n_off steht daneben.

    Rueckgabe: dict von Arrays gleicher Laenge, in aufsteigender
    tracklet_id.
    """
    tid = np.asarray(tracklet_id)
    gilt = tid >= 0
    werte, kompakt = np.unique(tid[gilt], return_inverse=True)
    m = len(werte)
    if m == 0:
        return {}

    x = np.asarray(x_on, np.float64)[gilt]
    y = np.asarray(y_on, np.float64)[gilt]
    t = np.asarray(t_on, np.float64)[gilt]

    n_on = np.bincount(kompakt, minlength=m).astype(np.int64)
    mx = np.bincount(kompakt, weights=x, minlength=m) / n_on
    my = np.bincount(kompakt, weights=y, minlength=m) / n_on

    # Zweite Momente fuer die Kovarianz. Der Verschiebungssatz reicht hier:
    # die Koordinaten liegen unter 1280, ein katastrophaler Ausloeschungs-
    # fehler ist bei float64 nicht zu erwarten.
    cxx = np.bincount(kompakt, weights=x * x, minlength=m) / n_on - mx * mx
    cyy = np.bincount(kompakt, weights=y * y, minlength=m) / n_on - my * my
    cxy = np.bincount(kompakt, weights=x * y, minlength=m) / n_on - mx * my
    spur, det = cxx + cyy, cxx * cyy - cxy * cxy
    wurzel = np.sqrt(np.maximum(spur * spur / 4.0 - det, 0.0))
    # Unter drei Punkten ist die Kovarianz entartet: lam2 wird exakt null
    # und lam1/lam2 explodiert. NAN wie bei x_off -- nicht messbar ist
    # nicht dasselbe wie null.
    entartet = n_on < 3
    lam1 = np.where(entartet, np.nan, spur / 2.0 + wurzel)
    lam2 = np.where(entartet, np.nan, spur / 2.0 - wurzel)

    t_min = np.full(m, np.inf); np.minimum.at(t_min, kompakt, t)
    t_max = np.full(m, -np.inf); np.maximum.at(t_max, kompakt, t)
    x_min = np.full(m, np.inf); np.minimum.at(x_min, kompakt, x)
    x_max = np.full(m, -np.inf); np.maximum.at(x_max, kompakt, x)
    y_min = np.full(m, np.inf); np.minimum.at(y_min, kompakt, y)
    y_max = np.full(m, -np.inf); np.maximum.at(y_max, kompakt, y)

    out = {
        "tracklet_id": werte.astype(np.int64),
        "fenster_idx": np.full(m, int(fenster_idx), np.int32),
        "n_on": n_on,
        "t_min": t_min, "t_max": t_max,
        "x_on": mx, "y_on": my,
        "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
        # lam1 >= lam2. Als Verschmelzungsindikator NICHT bestaetigt
        # (TODO §C), bleibt beschreibend.
        "lam1": lam1,
        "lam2": lam2,
    }

    # --- OFF-Seite. Zuordnungen auf fremde oder ungueltige IDs fallen weg.
    ot = np.asarray(off_tracklet)
    pos = np.searchsorted(werte, ot)
    ok = (pos < m) & (werte[np.clip(pos, 0, m - 1)] == ot)
    p, n_off = pos[ok], np.zeros(m, np.int64)
    n_off += np.bincount(p, minlength=m)
    sicher = n_off > 0
    out["n_off"] = n_off
    for name, quelle in (("x_off", x_off), ("y_off", y_off)):
        summe = np.bincount(p, weights=np.asarray(quelle, np.float64)[ok],
                            minlength=m)
        out[name] = np.divide(summe, n_off, out=np.full(m, np.nan),
                              where=sicher)
    # Mittlerer Konsens der zugeordneten OFF-Events: Unsicherheitsmarker
    # fuer D. Zusammen mit n_off/n_on sagt er, ob die Richtung traegt.
    out["off_konsens"] = np.divide(
        np.bincount(p, weights=np.asarray(off_konsens, np.float64)[ok],
                    minlength=m),
        n_off, out=np.full(m, np.nan), where=sicher)

    # --- GT, nur beim Training
    if track_id is not None and status_noise is not None:
        gt = np.asarray(track_id)[gilt]
        echt = ~np.asarray(status_noise)[gilt]
        sieger, anteil, n_versch = _mehrheit(kompakt[echt], m, gt[echt])
        out["gt_track_id"] = sieger
        out["gt_anteil"] = anteil
        out["n_gt"] = n_versch.astype(np.int32)
    return out