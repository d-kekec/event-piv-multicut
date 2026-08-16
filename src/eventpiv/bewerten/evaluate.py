#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py -- Cluster-Qualitaet gegen die GT-track_id.

Zwei Kennzahlen, die IMMER ZUSAMMEN gelesen werden muessen:

  REINHEIT je Cluster: groesster GT-Anteil in diesem Cluster.
      1.0 = der Cluster enthaelt nur ein Partikel.
  FRAGMENTIERUNG je GT-Partikel: in wie viele Cluster zerfaellt es?
      1.0 = ideal, das Partikel ist genau ein Cluster.

Warum zusammen: Reinheit allein ist trivial zu maximieren -- lauter
Singletons sind perfekt rein. Genau das lieferte das dr_weight-Netz:
Reinheit 0.999 bei Fragmentierung 5.38.

ZWEI MASKEN, und der Unterschied ist der Kern der Bewertung.

Reinheit, Fragmentierung und die Clustergroessen laufen ueber SAUBERE
Events. Ein mehrdeutiges Event traegt zwar eine track_id, aber eine
unsichere.

Die MERGE-Kennzahlen (Praefix `merge_`) laufen ueber alles ausser Rauschen.
Sie muessen es: mehrdeutig ist ein Event im Kern einer Ueberlappung, und
dort verschmilzt der Solver. `merge_event_frac` ist deshalb NICHT
`1 - purity_event_mean` -- verschiedene Teilmengen.

Ein verschmolzenes Tracklet ist in Stufe 2 EIN Knoten und laesst sich nicht
halbieren; ein zerschnittenes kann Stufe 2 verbinden. Die Fehler sind nicht
gleich teuer.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from eventpiv.kern.graph import edge_deltas
from eventpiv.kern.status import STATUS_CLEAN, STATUS_NOISE

# px. Darunter beruehren sich die Eventwolken zweier Partikel (Streuung bis
# ~6 px je Wolke, dazu r_max). Erst hier KANN der Solver verschmelzen.
NAH_PX = 19.0

def _kreuztabelle(labels, track_id, maske):
    """Cluster x GT-Kreuztabelle fuer eine Teilmenge der Events.

    Die eine Rechnung, aus der beide Kennzahlensaetze fallen -- einmal auf
    den sauberen Events, einmal auf allem ausser Rauschen.

    Kantenlose Knoten (label < 0) fallen heraus: sie gehoeren zu keinem
    Cluster.

    Rueckgabe: None, wenn nichts bewertbar ist. Sonst ein dict mit
      groessen         Eventzahl je Cluster
      groesste         groesste GT-Gruppe je Cluster
      n_gt_je_cluster  Zahl verschiedener GT-Partikel je Cluster
      frag             Zahl verschiedener Cluster je GT-Partikel
      gt_groessen      Eventzahl je GT-Partikel
      gt_groesster     groesstes Fragment je GT-Partikel
      zell_anzahl      Eventzahl je (Cluster, GT)-Zelle
      zell_lab         Clusterindex je Zelle (in lab_u)
      zell_gt          GT-Index je Zelle (in gt_u)
      lab_u, gt_u, gt_idx, gueltig   fuer die abgeleiteten Felder

    Die drei zell_-Felder sind die ungefaltete Kreuztabelle. Wer sie
    braucht, misst ueber ZUORDNUNGEN statt ueber Cluster oder Partikel --
    die Assoziationsmasse der HOTA-Familie tun das.
    """
    labels = np.asarray(labels)
    lab = labels[maske]
    gt = np.asarray(track_id)[maske]

    gueltig = lab >= 0
    lab, gt = lab[gueltig], gt[gueltig]
    if len(lab) == 0:
        return None

    lab_u, lab_idx = np.unique(lab, return_inverse=True)
    gt_u, gt_idx = np.unique(gt, return_inverse=True)

    # (Cluster, GT)-Kombinationen zaehlen
    key = lab_idx.astype(np.int64) * len(gt_u) + gt_idx
    kombi, anzahl = np.unique(key, return_counts=True)
    k_lab = kombi // len(gt_u)
    k_gt = kombi % len(gt_u)

    groesste = np.zeros(len(lab_u), dtype=np.int64)
    np.maximum.at(groesste, k_lab, anzahl)
    # Groesstes FRAGMENT je GT-Partikel -- das Gegenstueck zu groesste.
    # Erst diese Zahl unterscheidet "in drei gleiche Teile zerlegt" von
    # "ein grosses Stueck plus Kruemel".
    gt_groesster = np.zeros(len(gt_u), dtype=np.int64)
    np.maximum.at(gt_groesster, k_gt, anzahl)

    return {
        "gueltig": gueltig,
        "groessen": np.bincount(lab_idx, minlength=len(lab_u)),
        "groesste": groesste,
        "n_gt_je_cluster": np.bincount(k_lab, minlength=len(lab_u)),
        "frag": np.bincount(k_gt, minlength=len(gt_u)),
        "gt_groessen": np.bincount(gt_idx, minlength=len(gt_u)),
        "gt_groesster": gt_groesster,
        "zell_anzahl": anzahl, "zell_lab": k_lab, "zell_gt": k_gt,
        "lab_u": lab_u, "gt_u": gt_u, "gt_idx": gt_idx,
    }


def merge_metrics(labels, track_id, status, extra_maske=None) -> dict:
    """Wie oft verschmilzt der Solver zwei Partikel? Reine Funktion.

    Ueber alles AUSSER Rauschen, mehrdeutige Events eingeschlossen -- das
    ist der Unterschied zu cluster_metrics.

    DREI ZAHLEN AUS EINER RECHNUNG, und ihr ABSTAND ist die Diagnose:
      merge_frac_strikt  Cluster mit >= 2 GT-Partikeln. Ein einziges
                         fremdes Event genuegt, also pessimistisch.
      merge_frac_10      Cluster mit Minderheit >= 10 % -- "echt
                         verschmolzen".
      merge_event_frac   Anteil der EVENTS, die in ihrem Cluster nicht zur
                         Mehrheit gehoeren.
    Weit auseinander heisst Ausreisser, gleichauf heisst echte Verschmelzung.

    extra_maske : Einschraenkung auf eine Teilmenge, etwa die Partikel mit
        nahem Nachbarn. Unbedenklich fuer NAH/FERN: verschmolzene Partikel
        sind einander per Definition nah, liegen also in derselben Teilmenge.

    Rueckgabe: flaches dict mit `merge_`-Praefix. Leer, wenn nichts
    bewertbar ist.
    """
    maske = np.asarray(status) != STATUS_NOISE
    if extra_maske is not None:
        maske = maske & np.asarray(extra_maske)
    kt = _kreuztabelle(labels, track_id, maske)
    if kt is None:
        return {}
    groessen, groesste = kt["groessen"], kt["groesste"]
    minderheit = 1.0 - groesste / groessen
    return {
        "merge_n_events": int(groessen.sum()),
        "merge_n_clusters": int(len(kt["lab_u"])),
        "merge_n_gt": int(len(kt["gt_u"])),
        "merge_frac_strikt": float((kt["n_gt_je_cluster"] >= 2).mean()),
        "merge_frac_10": float((minderheit >= 0.10).mean()),
        "merge_event_frac": float(1.0 - groesste.sum() / groessen.sum()),
        # Absolut, nicht nur als Anteil: nur so lassen sich Fenster ohne
        # Gewichtungsfehler zu einem Run aufsummieren.
        "merge_n_minderheit": int(groessen.sum() - groesste.sum()),
        "merge_n_gt_max": int(kt["n_gt_je_cluster"].max()),
    }


def cluster_metrics(labels, track_id, status, t=None) -> dict:
    """Vergleicht eine Partition gegen die GT. Reine Funktion.

    labels    : Cluster-ID je Event (RAMA-Ausgabe; < 0 = keinem Cluster
                zugeordnet, entsteht bei Knoten ohne Kante)
    track_id  : GT je Event
    status    : status.STATUS_* je Event
    t         : optional Zeitstempel in µs. Nur fuer gt_visible_ms_median --
                ohne sie fehlt genau dieses Feld, alles andere bleibt.

    Rueckgabe: flaches dict. Leeres dict, wenn nichts zu bewerten ist.

    ZWEI REINHEITEN, und die Unterscheidung ist keine Spitzfindigkeit:
      purity_cluster_mean  Mittel UEBER CLUSTER -- ein Cluster aus 3 Events
                           zaehlt so viel wie einer aus 5000.
      purity_event_mean    Mittel UEBER EVENTS -- jedes Event zaehlt gleich.
    Bei vielen kleinen Fragmenten liegt die erste hoeher. Fuer die finale
    Zahl ist die eventgewichtete die ehrlichere.

    EINSCHRAENKUNG: nur SAUBERE Events. Auch die gt_-Felder gelten damit nur
    fuer diese Teilmenge -- gt_visible_ms_median unterschaetzt die
    Sichtbarkeit, wenn ein Partikel am Rand mehrdeutige Events hat.
    """
    clean = np.asarray(status) == STATUS_CLEAN
    kt = _kreuztabelle(labels, track_id, clean)
    if kt is None:
        return {}

    tt = None
    if t is not None:
        tt = np.asarray(t)[clean][kt["gueltig"]]

    lab_u, gt_u, gt_idx, gueltig = kt["lab_u"], kt["gt_u"], kt["gt_idx"], kt["gueltig"]
    groessen, groesste = kt["groessen"], kt["groesste"]
    reinheit = groesste / groessen
    frag, gt_groessen = kt["frag"], kt["gt_groessen"]

    out = {
        "n_events": int(groessen.sum()),
        "n_clusters": int(len(lab_u)),
        "n_gt": int(len(gt_u)),
        "purity_cluster_mean": float(reinheit.mean()),
        "purity_event_mean": float(groesste.sum() / groessen.sum()),
        "purity_lt90": int((reinheit < 0.9).sum()),
        "purity_lt90_frac": float((reinheit < 0.9).mean()),
        "cluster_size_min": int(groessen.min()),
        "cluster_size_median": float(np.median(groessen)),
        "cluster_size_max": int(groessen.max()),
        "n_singletons": int((groessen == 1).sum()),
        "frag_mean": float(frag.mean()),
        "frag_max": int(frag.max()),
        # 1.0 = ungeteilt. frag_mean allein unterscheidet "drei gleiche
        # Teile" nicht von "ein grosses plus Kruemel" -- fuer die
        # Zentroidgenauigkeit ist das der Unterschied.
        "frag_groesster_anteil_median": float(
            np.median(kt["gt_groesster"] / gt_groessen)),
        "gt_events_per_particle_median": float(np.median(gt_groessen)),
    }
    if tt is not None:
        # Sichtbarkeitsspanne je GT-Partikel. Vektorisiert, weil eine
        # Schleife ueber die Partikel bei grossen Fenstern der Engpass waere.
        tmax = np.full(len(gt_u), -np.inf)
        tmin = np.full(len(gt_u), np.inf)
        np.maximum.at(tmax, gt_idx, tt)
        np.minimum.at(tmin, gt_idx, tt)
        out["gt_visible_ms_median"] = float(np.median(tmax - tmin) / 1000.0)
    return out


def _voll(werte_u, werte):
    """Eventzahl je Eintrag von werte_u. Fremde Werte fallen heraus.

    werte_u kommt aus np.unique und ist damit sortiert. searchsorted
    liefert auch fuer fremde Werte eine Position -- ohne den
    Trefferabgleich zaehlte ein Partikel, das gar keinen Cluster
    erreicht hat, auf seinen Nachbarn.
    """
    if len(werte_u) == 0:
        return np.zeros(0, np.int64)
    pos = np.clip(np.searchsorted(werte_u, werte), 0, len(werte_u) - 1)
    treffer = werte_u[pos] == werte
    return np.bincount(pos[treffer], minlength=len(werte_u))


def _zellen(labels, track_id, status):
    """Zellen der Kreuztabelle plus die VOLLEN Trajektorienlaengen.

    Gemeinsame Vorarbeit der Tracking-Masse. _kreuztabelle ueber Millionen
    Events zweimal zu bilden waere reine Verschwendung.

    g_voll zaehlt alle Events des Partikels EINSCHLIESSLICH der
    unzugeordneten, p_voll alle Events des Clusters EINSCHLIESSLICH der
    Rauschevents -- nicht die Zellsummen.

    Rueckgabe: None, wenn nichts bewertbar ist. Sonst
    (kt, anz, kl, kg, g_voll, p_voll, n_tp, n_fn, n_fp).
    """
    labels = np.asarray(labels)
    track_id = np.asarray(track_id)
    ist_gt = np.asarray(status) != STATUS_NOISE
    ist_pr = labels >= 0

    kt = _kreuztabelle(labels, track_id, ist_gt)
    if kt is None:
        return None
    anz = kt["zell_anzahl"].astype(np.int64)
    return (kt, anz, kt["zell_lab"], kt["zell_gt"],
            _voll(kt["gt_u"], track_id[ist_gt]),
            _voll(kt["lab_u"], labels[ist_pr]),
            int(anz.sum()),
            int((ist_gt & ~ist_pr).sum()),
            int((~ist_gt & ist_pr).sum()))


def tracking_metrics(labels, track_id, status) -> dict:
    """Die HOTA-Familie auf EVENTEBENE. Reine Funktion.

    Nach Luiten, Osep, Dendorfer, Torr, Geiger, Leal-Taixe, Leibe:
    "HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking",
    IJCV 2021 (arXiv:2009.07736), Gleichungen 18 bis 27.

    EINE DETEKTION IST EIN ON-EVENT, und dasselbe Event ist zugleich
    gtDet und prDet. Daraus folgt dreierlei, und alles drei vereinfacht:

      - Die Aehnlichkeit S ist binaer (dasselbe Event oder nicht). Die
        Integration ueber Lokalisierungsschwellen alpha entfaellt, es
        gibt genau EIN HOTA und kein LocA.
      - Das bijektive Matching je Zeitschritt ist die Identitaet. Kein
        ungarisches Verfahren, keine Kostenmatrix.
      - Es gibt keine Zeitachse. TPA, FNA und FPA sind ueber Zellzahlen
        der Kreuztabelle berechenbar; das Paper nennt diese Abkuerzung
        selbst.

    DIE DREI MENGEN, und die Abgrenzung zur Paar-Ebene ist der Kern:

      event_tp  Event mit GT-Partikel UND in einem Cluster
      event_fn  Event mit GT-Partikel, aber label < 0 -- gefiltertes
                Tracklet (min_events) oder kantenloser Knoten
      event_fp  RAUSCHevent, das in einen Cluster eingewandert ist

    Ein MEHRDEUTIGES Event ist KEIN false positive. Es traegt eine
    track_id (das naechste Zentrum) und ist damit ein gueltiges gtDet;
    landet es im falschen Cluster, ist das ein ASSOZIATIONSfehler.
    Detektion beantwortet "kam das Event in irgendeinen Cluster",
    Assoziation "in den richtigen". Das Ueberlappungsproblem sitzt
    vollstaendig auf der zweiten Seite -- det_a liegt bei uns nahe 1,
    das Signal traegt ass_a. Das ist eine Eigenschaft der Aufgabe und
    gehoert beim Berichten gesagt.

    Deshalb dieselbe Maske wie merge_metrics (alles ausser Rauschen)
    und NICHT die von cluster_metrics: mehrdeutig ist ein Event genau
    im Kern einer Ueberlappung, und dort entscheidet sich die
    Assoziation. Preis: die GT dieser Events ist selbst unsicher.

    FN UND FP GEHEN IN DIE TRAJEKTORIENLAENGEN EIN. |g| zaehlt alle
    Events des Partikels einschliesslich seiner unclusterten, |p| alle
    Events des Clusters einschliesslich seiner Rauschevents -- nicht die
    Zellsummen. Das Paper begruendet es in Abschnitt 7 ("Including vs
    Excluding Detection Errors in the Association Score"): ohne diese
    Terme ist die Metrik NICHT MONOTON, das Hinzufuegen einer richtigen
    Detektion koennte den Wert senken.

    ass_re faellt, wenn ein Partikel in mehrere Cluster zerfaellt.
    ass_pr faellt, wenn ein Cluster sich ueber mehrere Partikel
    erstreckt. Das sind die etablierten Gegenstuecke zu unserer
    Fragmentierung und Verschmelzung -- rangkorreliert, aber NICHT
    dieselben Zahlen (gemessen: purity_event_mean gegen ass_pr
    Spearman 1,00 bei Pearson 0,998; frag_mean gegen ass_re nur 0,86).
    Beide Saetze bleiben deshalb nebeneinander stehen.

    Rueckgabe: flaches dict. Leeres dict, wenn nichts bewertbar ist --
    dieselbe Politik wie cluster_metrics und merge_metrics.
    """
    z = _zellen(labels, track_id, status)
    if z is None:
        return {}
    kt, anz, kl, kg, g_voll, p_voll, n_tp, n_fn, n_fp = z

    # A(c) je ZELLE statt je TP: alle TPs einer Zelle haben dieselbe
    # gtID und dieselbe prID, also denselben Wert. Die Gewichtung mit
    # anz holt die Mittelung ueber Detektionen zurueck.
    a = anz / (g_voll[kg] + p_voll[kl] - anz)

    det_a = n_tp / (n_tp + n_fn + n_fp)
    ass_a = float((anz * a).sum() / n_tp)
    # Verteilung statt Mittel: ass_a allein unterscheidet "ueberall
    # leicht unsauber" nicht von "zwei Ueberlappungen kaputt".
    jp = _je_partikel(kt, anz, kl, kg, g_voll, p_voll)
    aap = jp["ass_a_ptkl"]
    return {
        "ass_a_min": float(aap.min()),
        "ass_a_p10": float(np.percentile(aap, 10)),
        "ass_a_lt90": int((aap < 0.90).sum()),
        "ass_a_lt90_frac": float((aap < 0.90).mean()),
        "ass_pr_min": float(jp["ass_pr_ptkl"].min()),
        "ass_pr_lt90": int((jp["ass_pr_ptkl"] < 0.90).sum()),
        "n_cluster_eff_max": float(jp["n_cluster_eff"].max()),
        "event_tp": n_tp, "event_fn": n_fn, "event_fp": n_fp,
        "det_a": float(det_a),
        "det_re": float(n_tp / (n_tp + n_fn)),
        "det_pr": float(n_tp / (n_tp + n_fp)),
        "ass_a": ass_a,
        "ass_re": float((anz * (anz / g_voll[kg])).sum() / n_tp),
        "ass_pr": float((anz * (anz / p_voll[kl])).sum() / n_tp),
        "hota": float(np.sqrt(det_a * ass_a)),
    }


def _je_partikel(kt, anz, kl, kg, g_voll, p_voll) -> dict:
    """Die Assoziationsmasse je GT-Partikel, aus fertigen Zellen."""
    n_g = len(kt["gt_u"])
    a = anz / (g_voll[kg] + p_voll[kl] - anz)
    n_ptkl = np.bincount(kg, weights=anz, minlength=n_g)

    def mittel(werte):
        # Gewicht ist die Eventzahl der Zelle -- dieselbe Mittelung ueber
        # Detektionen wie im Laufwert, nur je Partikel abgebrochen.
        return np.bincount(kg, weights=anz * werte, minlength=n_g) / n_ptkl

    return {
        "gt_track_id": kt["gt_u"],
        "n_ptkl": n_ptkl.astype(np.int64),
        "ass_re_ptkl": mittel(anz / g_voll[kg]),
        "ass_pr_ptkl": mittel(anz / p_voll[kl]),
        "ass_a_ptkl": mittel(a),
        # EFFEKTIVE Clusterzahl: Kehrwert des Simpson-Index ueber die
        # Verteilung der Events eines Partikels auf seine Cluster. Bei k
        # gleich grossen Stuecken genau k; ein grosses Stueck plus Kruemel
        # bleibt knapp ueber 1. Schwellenfrei und monoton in der
        # Zersplitterung -- eine 10-%-Schwelle zaehlte dreizehn Stuecke
        # zu 8 % als NULL und waere damit gegenlaeufig zum Zweck.
        "n_cluster_eff": n_ptkl ** 2 / np.bincount(
            kg, weights=(anz.astype(np.float64)) ** 2, minlength=n_g),
    }


def tracking_metrics_je_partikel(labels, track_id, status) -> dict:
    """Die Assoziationsmasse AUFGESCHLUESSELT je GT-Partikel.

    ERFINDET KEINE KENNZAHL. ass_a ist bereits das mit der Eventzahl
    gewichtete Mittel ueber die GT-Partikel; diese Funktion unterlaesst
    nur das Zusammenfassen:

        ass_a == sum_p (n_ptkl[p] / n_tp) * ass_a_ptkl[p]

    Der Grund: ein Laufmittel ueber Millionen Events kann "ueberall
    leicht unsauber" nicht von "fast alles perfekt, zwei Ueberlappungen
    kaputt" trennen -- beide erzeugen dieselbe Zahl. Die Summanden
    trennen sie.

    VIER SIGNATUREN, und nur eine davon ist der gesuchte Fall:

      sauber rekonstruiert  ass_re ~1   ass_pr ~1   n_cluster_10 = 1
      Kruemel abgesplittert ass_re ab   ass_pr ~1   n_cluster_10 = 1
      IDENTITAETSWECHSEL    ass_re ab   ass_pr ab   n_cluster_10 = 2
      fremdes Stueck geerbt ass_re ~1   ass_pr ab   n_cluster_10 = 1

    Nur der Identitaetswechsel senkt BEIDE Masse gleichzeitig und hebt
    die Zahl der substanziellen Cluster ueber eins -- und er tritt
    PAARWEISE auf, an zwei Partikeln zugleich. Das macht ihn pruefbar
    statt bloss plausibel.

    Rueckgabe: dict von Arrays, alle parallel zu gt_track_id (aufsteigend).
    Leeres dict, wenn nichts bewertbar ist. Partikel ohne ein einziges
    zugeordnetes Event kommen NICHT vor -- sie haben kein TP und tragen zu
    keiner Assoziation bei; sie stecken vollstaendig in event_fn.
    """
    z = _zellen(labels, track_id, status)
    if z is None:
        return {}
    return _je_partikel(*z[:6])


def idf1_metrics(labels, track_id, status) -> dict:
    """IDF1 und seine Bestandteile. Reine Funktion.

    Nach Ristani, Solera, Zou, Cucchiara, Tomasi: "Performance Measures
    and a Data Set for Multi-Target, Multi-Camera Tracking", ECCV
    Workshops 2016.

    ANDERE BAUART ALS DIE HOTA-FAMILIE, und der Unterschied ist der
    Kern der Kritik an IDF1: gematcht wird EINE Zuordnung GT-Trajektorie
    zu Cluster ueber den ganzen Run, bijektiv. Jeder Cluster ausserhalb
    dieser Zuordnung zaehlt vollstaendig als IDFP -- auch wenn er
    perfekt rein ist.

    DIE VEREINFACHUNG. Matcht man g mit p, kostet das |g| - c an IDFN
    und |p| - c an IDFP, also zusammen

        IDFN + IDFP = summe|g| + summe|p| - 2 * IDTP

    Die beiden Summen haengen nicht vom Matching ab. IDF1 zu maximieren
    heisst damit, die Gesamtueberlappung zu maximieren -- ein
    rechteckiges Zuordnungsproblem auf der Kreuztabelle, kein Aufbau
    von Kostenmatrizen mit Dummy-Knoten.

    ALLE GT WERDEN ZWANGSWEISE GEMATCHT. Zulaessig, weil ein Match wegen
    -2c nie mehr kostet als eine Nichtzuordnung; gemessen an 400
    Zufallsfaellen gegen vollstaendige Aufzaehlung, Abweichung null.

    LAUFZEIT: die Ueberlappungsmatrix ist (GT x Cluster). Auf der
    stufe1-Basislinie sind das rund 85 x 130.000, also 90 MB in float64
    und 0,4 s -- gemessen 16.08. float32 waere zu knapp: Zellzahlen
    reichen bis in die Hunderttausende, float32 ist nur bis 2^24 exakt.

    Rueckgabe: flaches dict. Leeres dict, wenn nichts bewertbar ist.
    """
    z = _zellen(labels, track_id, status)
    if z is None:
        return {}
    kt, anz, kl, kg, _, _, n_tp, n_fn, n_fp = z

    ueberlappung = np.zeros((len(kt["gt_u"]), len(kt["lab_u"])), np.float64)
    ueberlappung[kg, kl] = anz
    zeile, spalte = linear_sum_assignment(ueberlappung, maximize=True)
    idtp = int(ueberlappung[zeile, spalte].sum())

    n_gtdet, n_prdet = n_tp + n_fn, n_tp + n_fp
    return {
        "idtp": idtp,
        "idfn": int(n_gtdet - idtp),
        "idfp": int(n_prdet - idtp),
        "idr": float(idtp / n_gtdet),
        "idp": float(idtp / n_prdet),
        "idf1": float(2 * idtp / (n_gtdet + n_prdet)),
    }


def _einheiten(labels, track_id, t, ist_gt, delta):
    """Zeitgeordnete Einheiten je GT-Partikel. Reine Funktion.

    delta = None  eine Einheit ist ein EVENT (die woertliche Definition)
    delta = Zahl  eine Einheit ist (Partikel, Zeitbin) mit dem
                  MEHRHEITSCLUSTER seiner zugeordneten Events
                  Gleichstand: die kleinste Cluster-ID gewinnt.

    Rueckgabe: (g, lab), aufsteigend nach (Partikel, Zeit). lab < 0 heisst
    "in dieser Einheit nicht verfolgt".
    """
    idx = np.flatnonzero(ist_gt)
    g = np.asarray(track_id)[idx]
    zeit = np.asarray(t)[idx].astype(np.int64)
    lab = np.asarray(labels)[idx].astype(np.int64)
    if delta is None:
        return (lambda o: (g[o], lab[o]))(np.lexsort((zeit, g)))

    b = (zeit - zeit.min()) // int(delta)
    g_u, g_i = np.unique(g, return_inverse=True)
    n_b = int(b.max()) + 1
    # schl_u ist aufsteigend, die Einheiten sind damit ohne zweites
    # Sortieren nach (Partikel, Bin) geordnet.
    schl_u, schl_i = np.unique(g_i * n_b + b, return_inverse=True)

    lab_e = np.full(len(schl_u), -1, np.int64)
    m = lab >= 0
    if m.any():
        l_u, l_i = np.unique(lab[m], return_inverse=True)
        k, c = np.unique(schl_i[m] * len(l_u) + l_i, return_counts=True)
        # Aufsteigend nach Anzahl schreiben: der letzte Schreibvorgang je
        # Einheit ist die Mehrheit. BEI GLEICHSTAND GEWINNT DIE KLEINSTE
        # CLUSTER-ID -- willkuerlich, aber festgelegt. Ohne die zweite
        # Sortierstufe entscheidet die Reihenfolge in `k`, und die ist
        # keine Definition; sie wich von der naiven Referenz in 1694 von
        # 300 Zufallsfaellen ab (gemessen 16.08.).
        o = np.lexsort((-(k % len(l_u)), c))
        lab_e[(k // len(l_u))[o]] = l_u[(k % len(l_u))[o]]
    return g_u[schl_u // n_b], lab_e


def idsw_metrics(labels, track_id, status, t, delta=None) -> dict:
    """Identitaetswechsel und Fragmentierungen. Reine Funktion.

    IDSW nach Bernardin & Stiefelhagen, "Evaluating Multiple Object
    Tracking Performance: The CLEAR MOT Metrics", EURASIP J. Image Video
    Process. 2008; formal in Luiten et al. 2021, Gl. 44.

    Frag nach Li, Huang, Nevatia, "Learning to Associate:
    HybridBoosted Multi-Target Tracker for Crowded Scene", CVPR 2009,
    zurueckgehend auf Wu & Nevatia, CVPR 2006. NICHT zu verwechseln mit
    FragA aus dem HOTA-Papier (Gl. 33/34) und nicht mit unserem
    lhn_frag_mean -- drei verschiedene Groessen unter demselben Kurznamen:

      lhn_frag_mean  Cluster je GT-Partikel, ZEITLOS (cluster_metrics)
      lhn_frag       Zahl der UNTERBRECHUNGEN einer GT-Trajektorie (hier)
      lhn_frag_a     Anteil der Assoziation im zusammenhaengenden Stueck
                 (HOTA-Papier, Baustein von FA-HOTA)

    IDSW: ein verfolgtes Ereignis zaehlt, wenn seine Cluster-ID von der
    des ZULETZT verfolgten Ereignisses desselben Partikels abweicht.
    EINE LUECKE SETZT NICHT ZURUECK -- verschwindet ein Partikel und
    kommt mit anderer ID wieder, ist das ein Wechsel. So steht es in
    Gl. 44 (`prev` ist das TP mit groesstem t, nicht das unmittelbar
    vorangehende Ereignis).

    Frag: Zahl der Uebergaenge "nicht verfolgt -> verfolgt" NACH dem
    ersten verfolgten Ereignis. Gleichwertig: Zahl der zusammenhaengenden
    verfolgten Abschnitte minus eins, je Partikel. Ein Abschnitt ohne
    GT-Events ist keine Luecke -- das Partikel existiert dort nicht.

    Die beiden sind ORTHOGONAL: eine Unterbrechung ohne ID-Wechsel ist
    Frag ohne IDSW, ein Wechsel ohne Unterbrechung umgekehrt.

    delta in µs rastert die Zeit; None rechnet auf Eventebene. Warum
    ueberhaupt eine Rasterung, obwohl die Definition sie nicht verlangt:
    liegen zwei Tracklets desselben Partikels innerhalb eines Fensters
    zeitlich VERSCHRAENKT, springt die ID am Eventstrom mehrfach hin und
    her, obwohl die Trajektorie nur einmal geteilt ist. Dazu kommen
    Zeitgleichstaende -- rund vier Events je µs, und ihre Reihenfolge
    ist willkuerlich. delta fasst beides zusammen. OB das ins Gewicht
    faellt, ist eine Messung und keine Annahme; deshalb ist die
    Voreinstellung None, die nichts hinzuerfindet.

    delta beschleunigt NICHT. Die Rasterung braucht denselben Durchgang
    und zusaetzlich die Mehrheitsbestimmung.

    Rueckgabe: flaches dict, `delta_us` = 0 heisst Eventebene. Leeres
    dict, wenn nichts bewertbar ist.
    """
    ist_gt = np.asarray(status) != STATUS_NOISE
    if not ist_gt.any():
        return {}
    g_s, lab_s = _einheiten(labels, track_id, t, ist_gt, delta)
    if len(g_s) == 0:
        return {}

    verfolgt = lab_s >= 0
    gleich = np.zeros(len(g_s), bool)
    gleich[1:] = g_s[1:] == g_s[:-1]
    grp = np.zeros(len(g_s), np.int64)
    grp[1:] = np.cumsum(~gleich[1:])
    n_grp = int(grp[-1]) + 1

    # --- IDSW: Vergleich mit der VORHERIGEN VERFOLGTEN Einheit, nicht mit
    # der unmittelbar vorangehenden. Das Ausduennen auf `v` ueberspringt
    # die Luecken, ohne sie zurueckzusetzen.
    wechsel = np.zeros(len(g_s), bool)
    v = np.flatnonzero(verfolgt)
    if len(v) > 1:
        wechsel[v[1:]] = ((g_s[v[1:]] == g_s[v[:-1]])
                          & (lab_s[v[1:]] != lab_s[v[:-1]]))
    idsw_je = np.bincount(grp[wechsel], minlength=n_grp)

    # --- Frag: Abschnitte minus eins. Ein Abschnitt beginnt bei einer
    # verfolgten Einheit, deren Vorgaenger fehlt, zu einem anderen
    # Partikel gehoert oder nicht verfolgt war.
    beginn = verfolgt.copy()
    beginn[1:] &= ~(gleich[1:] & verfolgt[:-1])
    abschnitte = np.bincount(grp[beginn], minlength=n_grp)
    frag_je = np.maximum(abschnitte - 1, 0)

    return {
        "delta_us": float(delta) if delta else 0.0,
        "n_einheiten": int(len(g_s)),
        "idsw": int(idsw_je.sum()),
        "idsw_max": int(idsw_je.max()),
        "idsw_ptkl_betroffen": int((idsw_je > 0).sum()),
        "lhn_frag": int(frag_je.sum()),
        "lhn_frag_max": int(frag_je.max()),
        "lhn_frag_ptkl_betroffen": int((frag_je > 0).sum()),
    }


def align_gt_window(gt_t, event_t):
    """Findet den Index-Ausschnitt der Label-Datei, der zum Eventfenster passt.

    Warum nicht per Zeitmaske: viele Events teilen sich einen Zeitstempel
    (rund vier je µs), und dann liefert (t >= t0) & (t <= t1) am Fensterrand
    mehr Eintraege als das geladene Fenster hat. Da die Label-Datei denselben
    Eventstrom in derselben Reihenfolge enthaelt, ist der Ausschnitt
    ZUSAMMENHAENGEND: Startindex suchen, n Eintraege nehmen, Zeitstempel
    gegenpruefen.

    Ein GEFILTERTER Eventstrom (ON-only) faellt hier auf -- als Teilfolge
    laufen die Positionen sofort auseinander. Die Meldung zeigt dann
    allerdings auf die falsche Ursache. Richtig: erst auf dem ungefilterten
    Strom ausrichten, dann dieselbe Maske auf beide Seiten.

    Rueckgabe: slice. ValueError, wenn die Zeitstempel nicht passen.
    """
    gt_t = np.asarray(gt_t)
    event_t = np.asarray(event_t)
    n = len(event_t)
    if n == 0:
        return slice(0, 0)

    start = int(np.searchsorted(gt_t, event_t[0], side="left"))
    sl = slice(start, start + n)
    ausschnitt = gt_t[sl]
    if len(ausschnitt) != n or not np.array_equal(
            ausschnitt.astype(np.int64), event_t.astype(np.int64)):
        raise ValueError(
            f"GT passt nicht zum Eventfenster (Start {start}, {n} Events). "
            "Wurde die Label-Datei ueber den GANZEN Run erzeugt?")
    return sl


def edge_gt_stats(i, j, t, x, y, track_id, status, pos=None) -> dict:
    """Statistik ueber die KANTEN eines Fensters, gegen die GT.

    Gegenstueck zu cluster_metrics: dort wird die AUSGABE des Solvers
    bewertet, hier seine EINGABE. edge_ja_frac ist die Klassenverteilung,
    die das Netz im Einsatz vorfindet -- ohne sie liesse sich nicht trennen,
    ob eine Abweichung von der Holdout-Rate an der Verteilung liegt oder an
    den Merkmalen.

    NUR Kanten mit ZWEI sauberen Endpunkten sind bewertbar, dieselbe
    Bedingung wie beim Sammeln. edge_bewertbar_frac sagt, auf welchem
    Anteil der Kanten die Aussage ruht.

    dt_mean und dr_mean laufen dagegen ueber ALLE Kanten -- das ist, was das
    Netz im Einsatz sieht.

    Legt vier Arrays der Kantenzahl in float64 an, deshalb NACH dem
    RAMA-Aufruf.
    """
    i, j = np.asarray(i), np.asarray(j)
    if len(i) == 0:
        return {}
    dt, dx, dy = edge_deltas(t, x, y, i, j)
    out = {"dt_mean": float(dt.mean()),
           "dr_mean": float(np.sqrt(dx * dx + dy * dy).mean())}

    clean = np.asarray(status) == STATUS_CLEAN
    beide = clean[i] & clean[j]
    n_bew = int(beide.sum())
    out["edge_n_bewertbar"] = n_bew
    out["edge_bewertbar_frac"] = float(n_bew / len(i))
    if n_bew:
        tid = np.asarray(track_id)
        ja = tid[i[beide]] == tid[j[beide]]
        out["edge_ja_frac"] = float(ja.mean())
        if pos is not None:
            # VORHERSAGE ist das Vorzeichen der Kosten, nicht der Schnitt des
            # Solvers: Multicut darf von der Kantenpraeferenz abweichen, wenn
            # die Bilanz es verlangt. Hier wird das NETZ bewertet, nicht RAMA.
            # Schwelle bei 0, weil die Kosten Logits sind.
            pos = np.asarray(pos)[beide]
            out["edge_tp"] = int((ja & pos).sum())
            out["edge_fn"] = int((ja & ~pos).sum())
            out["edge_fp"] = int((~ja & pos).sum())
            out["edge_tn"] = int((~ja & ~pos).sum())
    return out


def min_abstand_je_partikel(bbox, box_w=11.0):
    """Kleinster Abstand JE PARTIKEL zu irgendeinem anderen, je Rasterpunkt.

    Nicht das globale Minimum ueber alle Paare: bei rund 50 Partikeln liegt
    IRGENDEINES praktisch immer naeher als NAH_PX -- gemessen fielen auf
    run76 alle 799 Fenster auf dieselbe Seite. Die Frage ist ohnehin eine je
    Partikel.

    Rueckgabe: (raster_t, ids, d). `ids` sind die track_ids in aufsteigender
    Ordnung, `d` hat die Form (Rasterpunkte, Partikel). inf steht dort, wo
    ein Partikel zu diesem Zeitpunkt nicht sichtbar ist oder allein im Bild
    -- beides heisst "kann sich mit niemandem begegnen".
    """
    raster_t = np.unique(bbox["t"])
    ids = np.unique(bbox["track_id"]).astype(np.int64)
    pos = {t: k for k, t in enumerate(ids)}
    d = np.full((len(raster_t), len(ids)), np.inf)

    for k, tt in enumerate(raster_t):
        m = bbox["t"] == tt
        cx = bbox["x"][m].astype(np.float64) + box_w / 2.0
        cy = bbox["y"][m].astype(np.float64) + box_w / 2.0
        tid = bbox["track_id"][m].astype(np.int64)
        if len(cx) < 2:
            continue
        dd = np.hypot(cx[:, None] - cx[None, :], cy[:, None] - cy[None, :])
        np.fill_diagonal(dd, np.inf)
        d[k, [pos[int(a)] for a in tid]] = dd.min(axis=1)
    return raster_t, ids, d


def abstand_im_fenster(raster_t, ids, d, t_lo, t_hi):
    """Kleinster Nachbarabstand je Partikel im Zeitraum [t_lo, t_hi).

    UMSCHLIESSEND, nicht filternd: das Raster hat 16,67 ms Abstand, ein
    Tracklet-Fenster 10 ms -- rund vier von zehn Fenstern enthalten keinen
    Rasterpunkt. Eine Filterung meldete dort "keine Begegnung", wo nur nicht
    gemessen wurde.

    Genommen wird vom letzten Rasterpunkt bei oder vor t_lo bis zum ersten
    bei oder nach t_hi. Das unterschaetzt den Abstand, und das ist die
    sichere Richtung: ein Partikel faelschlich als NAH zu zaehlen kostet
    Trennschaerfe, es zu verfehlen verdeckt einen Merge.

    Rueckgabe: Array der Laenge len(ids), parallel zu ids.
    """
    if len(raster_t) == 0:
        return np.full(len(ids), np.inf)
    lo = max(int(np.searchsorted(raster_t, t_lo, side="right")) - 1, 0)
    hi = min(int(np.searchsorted(raster_t, t_hi, side="left")) + 1,
             len(raster_t))
    if hi <= lo:
        return np.full(len(ids), np.inf)
    return d[lo:hi].min(axis=0)


def nah_maske(track_id, ids, d_fenster, schwelle=NAH_PX):
    """Boolesche Maske je EVENT: hat sein Partikel einen nahen Nachbarn?

    Events ohne Partikel (Rauschen, track_id = -1) sind nie nah -- sie
    gehoeren zu keinem.
    """
    track_id = np.asarray(track_id)
    k = np.clip(np.searchsorted(ids, track_id), 0, max(len(ids) - 1, 0))
    treffer = (len(ids) > 0) & (ids[k] == track_id)
    return treffer & (d_fenster[k] < schwelle)