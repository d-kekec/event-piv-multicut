#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trajektorien.py -- BAUSTEIN [7]: aus der Partition die BAHNEN.

  lade_on_strom      (t, x, y) des ON-Stroms eines Runs, ohne Ground Truth
  gruppiere          Events nach Trajektorie, CSR-artig
  bahn               (t, x, y) EINER Trajektorie
  kalman_bahn        EINE Bahn filtern, wahlweise mit RTS-Glaetter
  versatz_korrektur  den Vorwaertsversatz der Eventwolke aufheben
  fitte_run          alle Bahnen eines Runs

Was [6c] als Event -> Trajektorie ablegt, wird hier zur Eingabe des Fits:
je Trajektorie drei gleich lange Arrays in aufsteigender Zeit.

DER VERTRAG zu [6c]: trajektorie_id hat EINEN Eintrag je ON-Event des
Runs, in der Reihenfolge des polaritaetsgefilterten Stroms. gruppiere
verlangt die Stromlaenge und bricht bei Abweichung ab -- ein Versatz um
einen Eintrag ordnete jedes Event der falschen Bahn zu und faellt an
keiner Zahl auf.

OHNE GROUND TRUTH: gelesen wird die .dat, nicht die Label-Datei. Echte
Messdaten haben keine Boxen, und der Fit braucht sie nicht.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eventpiv.kern.data import load_events
from eventpiv.kern.graph import polarity_mask


@dataclass
class Trajektorien:
    """Die Trajektorien eines Runs, plus der Weg zu ihren Events.

    trajektorie_id  (m,) ID je Zeile, aufsteigend -- derselbe Wert wie im
                    Artefakt von [6c], nicht ein zweiter Name dafuer
    reihenfolge     Indizes in den ON-Strom, nach Trajektorie gruppiert
    grenzen         m+1 Offsets in reihenfolge

    Dieselbe Ueberlegung wie bei Knoten: die drei gehoeren zusammen und
    sind einzeln wertlos. Keine Tabelle wie dort -- es gibt genau eine
    Spalte, und ein dict verspraeche mehr.

    Die Bruecke zurueck zur Partition. Nach kompakt() sind die IDs
    lueckenlos, dann ist sie die Identitaet. Aber nur dann, und das ist
    keine Zusicherung des Solvers.
    """
    trajektorie_id: np.ndarray
    reihenfolge: np.ndarray
    grenzen: np.ndarray

    def __len__(self) -> int:
        return len(self.grenzen) - 1


def _leer() -> Trajektorien:
    return Trajektorien(np.empty(0, np.int64), np.empty(0, np.int64),
                        np.zeros(1, np.int64))


def lade_on_strom(split: str, run_nr: int, polarity: str = "on"):
    """(t, x, y) des Runs als float64, nach Polaritaet gefiltert.

    float64, weil der Fit damit rechnet: die Zeitstempel gehen bis 8e6 µs
    und liegen damit unter der float32-Grenze von 2^24, aber der Abstand
    ist kleiner als eine Groessenordnung. Preis: rund 90 MB je Array bei
    11 Mio. ON-Events.
    """
    ev = load_events(split, run_nr)
    m = polarity_mask(ev["p"], polarity)
    return (ev["t"][m].astype(np.float64),
            ev["x"][m].astype(np.float64),
            ev["y"][m].astype(np.float64))


def gruppiere(trajektorie_id, n_strom: int) -> Trajektorien:
    """Events nach Trajektorie gruppieren. Reine Funktion.

    trajektorie_id  ein Eintrag je ON-Event, -1 fuer nicht zugeordnete
    n_strom         Laenge des Eventstroms, gegen den geprueft wird

    Events mit -1 fallen heraus: sie gehoeren zu keiner Bahn.

    kind="stable" haelt die ZEITORDNUNG innerhalb einer Trajektorie: der
    ON-Strom ist zeitlich sortiert, und der Kalman-Filter bildet sein dt
    aus aufeinanderfolgenden Messungen. Ohne die Stabilitaet muesste der
    Aufrufer je Bahn nachsortieren.
    """
    tid = np.asarray(trajektorie_id)
    if len(tid) != int(n_strom):
        raise ValueError(
            f"VERTRAG VERLETZT: {int(n_strom):,} Events im Strom, "
            f"{len(tid):,} Trajektorien-IDs. Andere Polaritaet, anderer "
            "Run oder ein anderes Artefakt.")
    gilt = tid >= 0
    if not gilt.any():
        return _leer()

    idx_gilt = np.flatnonzero(gilt)
    werte, grp = np.unique(tid[gilt], return_inverse=True)
    m = len(werte)
    ordn = np.argsort(grp, kind="stable")
    return Trajektorien(
        werte.astype(np.int64),
        idx_gilt[ordn].astype(np.int64),
        np.concatenate([[0], np.cumsum(np.bincount(grp, minlength=m))]
                       ).astype(np.int64))


def bahn(tr: Trajektorien, i: int, t, x, y):
    """(t, x, y) der i-ten Trajektorie, zeitlich aufsteigend.

    Sicht auf den Strom, keine Kopie der Rohdaten -- die Indizierung legt
    drei kleine Arrays an, nicht drei grosse.
    """
    idx = tr.reihenfolge[tr.grenzen[i]:tr.grenzen[i + 1]]
    return np.asarray(t)[idx], np.asarray(x)[idx], np.asarray(y)[idx]


# Zustand je Achse: (Position px, Geschwindigkeit px/s, Beschleunigung
# px/s^2). INTERN SEKUNDEN, weil die Kovarianz sonst ueber achtzehn
# Groessenordnungen spannt: in µs traegt die Position 1e-4 und die
# Beschleunigung 1e-22, und das liegt an der float64-Grenze. Die Zeit
# kommt in µs herein und geht in µs hinaus; umgerechnet wird an genau
# zwei Stellen, hier markiert mit S_JE_US.
S_JE_US = 1e-6
MIN_EVENTS_FIT = 10


def _start_p(n_init, t_spanne_s, r, a_unbekannt):
    """Startkovarianz aus der Statistik der Ausgleichsgeraden.

    sigma_p^2 = R/n, sigma_v^2 = 12 R/(n T^2) -- die Unsicherheit einer
    Regression, nicht ein geratenes Vielfaches. Faktor 4 als Reserve.
    Die Beschleunigung schaetzt die Gerade nicht: dort steht die Streuung
    a_unbekannt, in px/s^2.

    UNTERSCHAETZEN IST DER TEURE FEHLER: ein zu sicherer Startzustand
    wird nur langsam korrigiert, und der Bahnanfang bleibt sichtbar
    falsch. Zu gross kostet nur die ersten Schritte.
    """
    n = max(int(n_init), 2)
    tt = max(float(t_spanne_s), 1e-9)
    return (4.0 * r / n,
            4.0 * 12.0 * r / (n * tt * tt),
            a_unbekannt * a_unbekannt)


def _startzustand(t, x, y, init_us):
    """Position und Geschwindigkeit aus einer Ausgleichsgeraden.

    Ueber die ersten init_us des Bahnanfangs, Beschleunigung null. Ohne
    sie muesste der Filter die Geschwindigkeit aus dem Stand einregeln,
    und der Anfang jeder Bahn waere systematisch falsch.
    """
    m = t <= t[0] + init_us
    if int(m.sum()) < 3:
        m = np.zeros(len(t), bool)
        m[:3] = True
    tc = (t[m] - t[0]) * S_JE_US
    A = np.vander(tc, 2)
    kx = np.linalg.lstsq(A, x[m], rcond=None)[0]
    ky = np.linalg.lstsq(A, y[m], rcond=None)[0]
    return ((kx[1], kx[0], 0.0), (ky[1], ky[0], 0.0),
            int(m.sum()), float(tc[-1] - tc[0]))


def _rts_rueckwaerts(t, zp, zv, za, wp, wv, wa, pf, qc):
    """RTS-Glaetter, an Ort und Stelle. zp..za ist x, wp..wa ist y.

    Der Rueckwaertslauf benutzt KEINE Messungen -- jede geht genau einmal
    ein, im Vorwaertslauf. Er propagiert die bereits geglaettete
    Schaetzung zurueck und mischt sie mit dem, was der Vorwaertsfilter an
    dieser Stelle wusste:

        C_k   = P_k|k A^T (P_k+1|k)^-1
        x_k|N = x_k|k + C_k (x_k+1|N - A x_k|k)

    C ist das Verhaeltnis "wie sicher war ich hier" zu "wie sicher war
    meine Vorhersage nach dort". War der Filter hier unsicher und die
    Vorhersage gut, zieht die Zukunft stark.

    ALLE C GEMEINSAM, GESTAPELT: anders als im Vorwaertslauf haengt C_k
    nur an P_k und h_k, nicht am vorigen C. Beide liegen fertig vor, also
    rechnet NumPy die Inversen in einem Zug. Sequentiell bleibt nur das
    Mischen. Preis: rund 400 Byte je Event waehrend der Rechnung.

    Bei h = 0 ist A die Einheit und C damit ebenfalls -- die geglaettete
    Schaetzung wird unveraendert uebernommen, was zur selben Zeit richtig
    ist. Kein Sonderfall noetig.

    Die GEGLAETTETE Kovarianz wird nicht gefuehrt: sie ginge nur in
    Fehlerbalken ein, und die sind verworfen (Journal 13.09.).
    """
    n = len(t)
    h = (t[1:] - t[:-1]) * S_JE_US
    g = 0.5 * h * h
    m = n - 1

    A = np.zeros((m, 3, 3))
    A[:, 0, 0] = A[:, 1, 1] = A[:, 2, 2] = 1.0
    A[:, 0, 1] = A[:, 1, 2] = h
    A[:, 0, 2] = g

    P = np.empty((m, 3, 3))
    P[:, 0, 0] = pf[:-1, 0]
    P[:, 0, 1] = P[:, 1, 0] = pf[:-1, 1]
    P[:, 0, 2] = P[:, 2, 0] = pf[:-1, 2]
    P[:, 1, 1] = pf[:-1, 3]
    P[:, 1, 2] = P[:, 2, 1] = pf[:-1, 4]
    P[:, 2, 2] = pf[:-1, 5]

    h2 = h * h
    h3 = h2 * h
    Q = np.zeros((m, 3, 3))
    Q[:, 0, 0] = qc * h3 * h2 / 20.0
    Q[:, 0, 1] = Q[:, 1, 0] = qc * h2 * h2 / 8.0
    Q[:, 0, 2] = Q[:, 2, 0] = qc * h3 / 6.0
    Q[:, 1, 1] = qc * h3 / 3.0
    Q[:, 1, 2] = Q[:, 2, 1] = qc * h2 / 2.0
    Q[:, 2, 2] = qc * h

    M = P @ A.transpose(0, 2, 1)          # P A^T
    C = M @ np.linalg.inv(A @ M + Q)      # P A^T (A P A^T + Q)^-1
    del P, M, Q

    Z = np.stack([zp, zv, za], axis=1)
    W = np.stack([wp, wv, wa], axis=1)
    zvor = np.einsum("kij,kj->ki", A, Z[:-1])     # A x_k|k
    wvor = np.einsum("kij,kj->ki", A, W[:-1])
    del A

    # Flache Sichten: ein Skalarzugriff auf c00[i] ist deutlich billiger
    # als C[i, 0, 0], und ein C[i] @ d je Schritt waere ein NumPy-Aufruf
    # mit eigenem Ueberhang.
    c00, c01, c02 = C[:, 0, 0], C[:, 0, 1], C[:, 0, 2]
    c10, c11, c12 = C[:, 1, 0], C[:, 1, 1], C[:, 1, 2]
    c20, c21, c22 = C[:, 2, 0], C[:, 2, 1], C[:, 2, 2]

    for i in range(m - 1, -1, -1):
        d0 = zp[i + 1] - zvor[i, 0]
        d1 = zv[i + 1] - zvor[i, 1]
        d2 = za[i + 1] - zvor[i, 2]
        zp[i] += c00[i] * d0 + c01[i] * d1 + c02[i] * d2
        zv[i] += c10[i] * d0 + c11[i] * d1 + c12[i] * d2
        za[i] += c20[i] * d0 + c21[i] * d1 + c22[i] * d2

        d0 = wp[i + 1] - wvor[i, 0]
        d1 = wv[i + 1] - wvor[i, 1]
        d2 = wa[i + 1] - wvor[i, 2]
        wp[i] += c00[i] * d0 + c01[i] * d1 + c02[i] * d2
        wv[i] += c10[i] * d0 + c11[i] * d1 + c12[i] * d2
        wa[i] += c20[i] * d0 + c21[i] * d1 + c22[i] * d2


def kalman_bahn(t, x, y, *, r_mess, q_c, dt_aus=None, t_aus=None,
                init_us=100_000.0, glaetten=False, a_unbekannt=100.0):
    """EINE Bahn filtern und auf ein Zeitgitter legen. Reine Funktion.

    t, x, y   Events einer Bahn, t in µs aufsteigend
    r_mess    Varianz eines Einzelevents um die Bahn, px^2
    q_c       Intensitaet des Prozessrauschens, px^2/s^5
    dt_aus    Gitterweite der Ausgabe in µs, oder None
    t_aus     Ausgabezeitpunkte in µs, oder None -- genau eines von
              beiden. Explizite Zeiten vermeiden das Interpolieren beim
              Vergleich mit einem fremden Raster.
    init_us   Fenster fuer den Startzustand
    glaetten  RTS-Rueckwaertslauf anschliessen
    a_unbekannt  Streuung der Startbeschleunigung, px/s^2

    Rueckgabe: (t_aus, x_aus, y_aus, ux_aus, uy_aus)
      t_aus in µs, Ort in px, Geschwindigkeit in px/µs.

    EREIGNISGETRIEBEN: dt kommt aus aufeinanderfolgenden Events, nicht aus
    einem Raster. Unregelmaessige Zeitstempel sind der Normalfall.

    KONSTANTE BESCHLEUNIGUNG, je Achse entkoppelt. Die Kruemmung im
    Wirbelfeld erscheint in beiden 1D-Filtern als Beschleunigung -- ein
    Modell mit konstanter Geschwindigkeit liefe ihr systematisch hinterher.

    P IST FUER BEIDE ACHSEN DASSELBE: es haengt an dt, q_c und r_mess,
    nicht an den Messwerten. Einmal gerechnet, zweimal benutzt -- das gilt
    im Vorwaertslauf wie im Glaetter.

    DAS PROZESSRAUSCHEN HAENGT AN DER ZEIT, NICHT AN DER ZAHL DER
    MESSUNGEN. Ein fester Zuschlag je Schritt waere bei gleichmaessiger
    Abtastung dasselbe, hier nicht: bei 36 500 Events je Sekunde
    akkumulierte er das 36 500-fache je Sekunde, und der Filter hielte
    seine eigene Beschleunigung fuer unbestimmt. Gerechnet wird deshalb
    die Q-Matrix des Modells mit weissem Ruck.

    OHNE glaetten IST DER FILTER KAUSAL: an jeder Stelle kennt er nur die
    Vergangenheit, laeuft jeder Richtungsaenderung hinterher und hat am
    Bahnanfang gar nichts. Der Glaetter behebt beides, kostet einen
    zweiten Durchgang und rund 96 Byte je Event.

    Bei gleichem Zeitstempel wird NICHT vorhergesagt, nur korrigiert --
    ohne Zeit vergeht keine Modellunsicherheit. Rund 2,8 % der Events
    teilen sich einen Stempel.
    """
    t = np.asarray(t, np.float64)
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    n = len(t)
    if n < MIN_EVENTS_FIT:
        raise ValueError(
            f"{n} Events, mindestens {MIN_EVENTS_FIT} noetig. Kurze Bahnen "
            "vor dem Aufruf aussortieren -- ein Fit darauf ist eine Zahl "
            "ohne Aussage.")
    if (dt_aus is None) == (t_aus is None):
        raise ValueError("genau eines von dt_aus und t_aus setzen")
    if t_aus is None:
        raster = np.arange(np.ceil(t[0] / dt_aus) * dt_aus, t[-1], dt_aus)
    else:
        raster = np.asarray(t_aus, np.float64)
        if len(raster) > 1 and (np.diff(raster) <= 0).any():
            raise ValueError("t_aus muss aufsteigend sein")
    qc = float(q_c)
    r = float(r_mess)

    (xp, xv, xa), (yp, yv, ya), n_init, t_init = _startzustand(
        t, x, y, init_us)
    p00, p11, p22 = _start_p(n_init, t_init, r, a_unbekannt)
    p01 = p02 = p12 = 0.0

    # Zustand und Kovarianz je Event. Der Glaetter braucht sie; ohne ihn
    # kostet es 96 Byte je Event und macht die Ausgabe vektorisierbar.
    zp = np.empty(n); zv = np.empty(n); za = np.empty(n)
    wp = np.empty(n); wv = np.empty(n); wa = np.empty(n)
    pf = np.empty((n, 6))
    t_prev = t[0]

    for i in range(n):
        h = (t[i] - t_prev) * S_JE_US
        if h > 0.0:
            g = 0.5 * h * h
            # Vorhersage des Zustands: Newton mit konstanter Beschleunigung.
            xp += xv * h + xa * g;  xv += xa * h
            yp += yv * h + ya * g;  yv += ya * h
            # Vorhersage der Unsicherheit: P <- A P A^T + Q, mit
            #   A = [[1, h, h^2/2], [0, 1, h], [0, 0, 1]]
            # P ist symmetrisch, also genuegen sechs statt neun Eintraege.
            # aij ist die Zwischenmatrix A P. Ausgeschrieben statt
            # A @ P @ A.T, weil die Schleife je Bahn hunderttausendfach
            # laeuft und ein 3x3-Matrixprodukt in NumPy dort ein
            # Vielfaches der zwoelf Additionen kostet.
            a00 = p00 + h * p01 + g * p02
            a01 = p01 + h * p11 + g * p12
            a02 = p02 + h * p12 + g * p22
            a11 = p11 + h * p12
            a12 = p12 + h * p22
            # Q des Modells mit weissem Ruck der Intensitaet q_c. Die
            # Reihenfolge zaehlt: p11 liest p12 und p22 im alten Stand.
            h2 = h * h
            h3 = h2 * h
            p00 = a00 + h * a01 + g * a02 + qc * h3 * h2 / 20.0
            p01 = a01 + h * a02 + qc * h2 * h2 / 8.0
            p02 = a02 + qc * h3 / 6.0
            p11 = a11 + h * (p12 + h * p22) + qc * h3 / 3.0
            p12 = a12 + qc * h2 / 2.0
            p22 = p22 + qc * h
            t_prev = t[i]

        # Korrektur. C = [1,0,0] und R skalar, also ist S = C P C^T + R
        # eine Zahl und K = P C^T / S ein Spaltenvektor. Keine
        # Matrixinversion.
        #
        # K hat drei Eintraege, obwohl nur die POSITION gemessen wird:
        # ueber die Kopplungen p01 und p02 korrigiert eine Positions-
        # messung auch Geschwindigkeit und Beschleunigung. Genau deshalb
        # braucht es keine numerische Ableitung.
        s = p00 + r
        k0, k1, k2 = p00 / s, p01 / s, p02 / s
        dx, dy = x[i] - xp, y[i] - yp
        xp += k0 * dx; xv += k1 * dx; xa += k2 * dx
        yp += k0 * dy; yv += k1 * dy; ya += k2 * dy
        # P <- (I - K C) P, ebenfalls ausgeschrieben. bij haelt die alte
        # erste Zeile fest, die weiter unten noch gebraucht wird.
        b00, b01, b02 = p00, p01, p02
        p00 -= k0 * b00; p01 -= k0 * b01; p02 -= k0 * b02
        p11 -= k1 * b01; p12 -= k1 * b02
        p22 -= k2 * b02

        zp[i] = xp; zv[i] = xv; za[i] = xa
        wp[i] = yp; wv[i] = yv; wa[i] = ya
        pf[i, 0] = p00; pf[i, 1] = p01; pf[i, 2] = p02
        pf[i, 3] = p11; pf[i, 4] = p12; pf[i, 5] = p22

    if glaetten:
        _rts_rueckwaerts(t, zp, zv, za, wp, wv, wa, pf, qc)

    # Ausgabe: vom letzten Event VOR dem Gitterpunkt frei propagiert.
    # Vektorisiert -- die Zustaende liegen fertig vor.
    j = np.clip(np.searchsorted(t, raster, side="right") - 1, 0, n - 1)
    h = (raster - t[j]) * S_JE_US
    g = 0.5 * h * h
    return (raster,
            zp[j] + zv[j] * h + za[j] * g,
            wp[j] + wv[j] * h + wa[j] * g,
            (zv[j] + za[j] * h) * S_JE_US,
            (wv[j] + wa[j] * h) * S_JE_US)


def versatz_korrektur(x, y, u, v, *, c0, c1):
    """Den Zentroid gegen die Bewegungsrichtung zurueckschieben. Rein.

    Die ON-Events liegen an der VORDERKANTE des Partikelbilds, der
    Schwerpunkt also in Bewegungsrichtung versetzt. Der Versatz waechst
    mit der Geschwindigkeit: 1,60 px bei 0,028 px/ms gegen 3,03 px bei
    0,076 px/ms.

    Betrag c0 + c1*|u|, nicht beta*sigma_quer wie bei STELLA: die
    Querstreuung ist ueber unseren Geschwindigkeitsbereich konstant
    (3,03 bis 3,17 px) und kann den wachsenden Versatz nicht abbilden.

    u, v in px/µs; c1 in px je (px/ms), daher der Faktor 1e3. c0 = c1 = 0
    schaltet die Korrektur ab.
    """
    b = np.hypot(u, v)
    e = np.maximum(b, 1e-12)
    schub = float(c0) + float(c1) * b * 1e3
    return x - schub * u / e, y - schub * v / e


def fitte_run(tr: Trajektorien, t, x, y, *, r_mess, q_c, c0, c1,
              dt_aus, min_dauer_us, init_us=100_000.0, glaetten=False,
              a_unbekannt=100.0):
    """Alle Bahnen eines Runs filtern und korrigieren.

    Rueckgabe: sechs gleich lange Arrays (t, x, y, ux, uy,
    trajektorie_id), alle Bahnen aneinandergehaengt. Die Stuetzpunkte
    einer Bahn liegen zusammen und sind innerhalb aufsteigend in t.

    Bahnen unter min_dauer_us fallen heraus: der Filter braucht init_us
    allein zum Einschwingen, und ein Fit ueber wenig mehr ist eine Zahl
    ohne Aussage. Bahnen unter MIN_EVENTS_FIT ebenfalls -- sonst wuerfe
    kalman_bahn mitten in der Schleife.
    """
    tf, xf, yf, uf, vf, idf = [], [], [], [], [], []
    for i in range(len(tr)):
        tb, xb, yb = bahn(tr, i, t, x, y)
        if len(tb) < MIN_EVENTS_FIT or tb[-1] - tb[0] < min_dauer_us:
            continue
        ta, xa, ya, ua, va = kalman_bahn(
            tb, xb, yb, r_mess=r_mess, q_c=q_c, dt_aus=dt_aus,
            init_us=init_us, glaetten=glaetten, a_unbekannt=a_unbekannt)
        if len(ta) == 0:
            continue
        xk, yk = versatz_korrektur(xa, ya, ua, va, c0=c0, c1=c1)
        tf.append(ta);  xf.append(xk);  yf.append(yk)
        uf.append(ua);  vf.append(va)
        idf.append(np.full(len(ta), tr.trajektorie_id[i], np.int64))
    if not tf:
        leer = np.empty(0)
        return (leer,) * 5 + (np.empty(0, np.int64),)
    return (np.concatenate(tf), np.concatenate(xf), np.concatenate(yf),
            np.concatenate(uf), np.concatenate(vf), np.concatenate(idf))


def schreibe_fit(out, cfg, run_nr, spalten):
    """Die gefitteten Bahnen als npz. EIN Eintrag je Stuetzpunkt.

    out: das Ziel. Der AUFRUFER loest den Pfad auf -- er tut es ohnehin
    fuer die Ueberspringen-Pruefung, und dieses Modul soll schreiben und
    nicht wissen, wo etwas liegt.

    spalten: (t, x, y, ux, uy, trajektorie_id) aus
    fitte_run. Eine Spalte je Punkt und kein CSR: wer die Datei oeffnet,
    sieht acht gleich lange Spalten und weiss, was er hat. Nach Bahnen
    aufteilen geht mit searchsorted, die IDs liegen gruppiert.
    """
    from pathlib import Path
    from dataclasses import replace
    from eventpiv.kern import config
    from eventpiv.kern.runconfig import identity_meta

    t, x, y, ux, uy, tid = spalten
    sel = replace(cfg.selection, run_nr=run_nr)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        t=np.asarray(t, np.float64),
        x=np.asarray(x, np.float32), y=np.asarray(y, np.float32),
        ux=np.asarray(ux, np.float32), uy=np.asarray(uy, np.float32),
        trajektorie_id=np.asarray(tid, np.int32),
        meta=config.meta_json(
            stage="fit", n_punkte=int(len(t)),
            n_bahnen=int(len(np.unique(tid))),
            **identity_meta(neighborhood=cfg.neighborhood, pairs=cfg.pairs,
                            training=cfg.training, solver=cfg.solver,
                            kandidaten=cfg.kandidaten,
                            transformer=cfg.transformer,
                            solver2=cfg.solver2, fit=cfg.fit,
                            selection=sel)))
    return out