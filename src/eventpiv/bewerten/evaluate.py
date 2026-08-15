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
    labels = np.asarray(labels)
    track_id = np.asarray(track_id)
    ist_gt = np.asarray(status) != STATUS_NOISE
    ist_pr = labels >= 0

    kt = _kreuztabelle(labels, track_id, ist_gt)
    if kt is None:
        return {}

    anz = kt["zell_anzahl"].astype(np.int64)
    kl, kg = kt["zell_lab"], kt["zell_gt"]

    n_tp = int(anz.sum())
    n_fn = int((ist_gt & ~ist_pr).sum())
    n_fp = int((~ist_gt & ist_pr).sum())

    g_voll = _voll(kt["gt_u"], track_id[ist_gt])
    p_voll = _voll(kt["lab_u"], labels[ist_pr])

    # A(c) je ZELLE statt je TP: alle TPs einer Zelle haben dieselbe
    # gtID und dieselbe prID, also denselben Wert. Die Gewichtung mit
    # anz holt die Mittelung ueber Detektionen zurueck.
    a = anz / (g_voll[kg] + p_voll[kl] - anz)

    det_a = n_tp / (n_tp + n_fn + n_fp)
    ass_a = float((anz * a).sum() / n_tp)
    return {
        "event_tp": n_tp, "event_fn": n_fn, "event_fp": n_fp,
        "det_a": float(det_a),
        "det_re": float(n_tp / (n_tp + n_fn)),
        "det_pr": float(n_tp / (n_tp + n_fp)),
        "ass_a": ass_a,
        "ass_re": float((anz * (anz / g_voll[kg])).sum() / n_tp),
        "ass_pr": float((anz * (anz / p_voll[kl])).sum() / n_tp),
        "hota": float(np.sqrt(det_a * ass_a)),
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