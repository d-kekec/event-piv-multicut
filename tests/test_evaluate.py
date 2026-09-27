#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Invariantentests fuer evaluate.

Kernstueck: cluster_metrics_naiv als offensichtlich korrekte Referenz gegen
die vektorisierte Fassung -- dasselbe Muster wie der Brute-Force-Test in
test_graph.py. Zusaetzlich Faelle mit von Hand ausgerechnetem Ergebnis.
"""
import numpy as np
import pytest

from eventpiv.bewerten.evaluate import (align_gt_window, cluster_metrics,
                               edge_gt_stats, merge_metrics, fa_hota_metrics)
from eventpiv.kern.status import STATUS_AMBIG, STATUS_CLEAN, STATUS_NOISE


def cluster_metrics_naiv(labels, track_id, status, t=None):
    """Referenz: direkte Schleifen, keine Tricks.

    Deckt bewusst ALLE Felder ab, nicht nur die urspruenglichen: sonst
    laufen neue Kennzahlen ungeprueft mit, und genau dort sitzen die Fehler
    (purity_event_mean nimmt hier den anderen Rechenweg -- Summe ueber
    Reinheit x Groesse statt Summe der Mehrheitsanteile).
    """
    clean = np.asarray(status) == STATUS_CLEAN
    lab = np.asarray(labels)[clean]
    gt = np.asarray(track_id)[clean]
    tt = np.asarray(t)[clean] if t is not None else None
    ok = lab >= 0
    lab, gt = lab[ok], gt[ok]
    if tt is not None:
        tt = tt[ok]
    if len(lab) == 0:
        return {}

    reinheit, groessen = [], []
    for c in np.unique(lab):
        _, cnt = np.unique(gt[lab == c], return_counts=True)
        reinheit.append(cnt.max() / cnt.sum())
        groessen.append(cnt.sum())
    frag = np.array([len(np.unique(lab[gt == p])) for p in np.unique(gt)])
    gt_gr = [int((gt == p).sum()) for p in np.unique(gt)]
    reinheit, groessen = np.array(reinheit), np.array(groessen)

    out = {"n_events": int(len(lab)),
           "n_clusters": int(len(np.unique(lab))),
           "n_gt": int(len(np.unique(gt))),
           "purity_cluster_mean": float(reinheit.mean()),
           "purity_event_mean": float((reinheit * groessen).sum()
                                      / groessen.sum()),
           "purity_lt90": int((reinheit < 0.9).sum()),
           "purity_lt90_frac": float((reinheit < 0.9).mean()),
           "cluster_size_min": int(groessen.min()),
           "cluster_size_median": float(np.median(groessen)),
           "cluster_size_max": int(groessen.max()),
           "n_singletons": int((groessen == 1).sum()),
           "frag_mean": float(frag.mean()),
           "frag_max": int(frag.max()),
           "frag_groesster_anteil_median": float(np.median(
               [max((lab[gt == p] == c).sum() for c in np.unique(lab[gt == p]))
                / (gt == p).sum() for p in np.unique(gt)])),
           "gt_events_per_particle_median": float(np.median(gt_gr))}
    if tt is not None:
        sp = [tt[gt == p].max() - tt[gt == p].min() for p in np.unique(gt)]
        out["gt_visible_ms_median"] = float(np.median(sp) / 1000.0)
    return out


def test_perfekte_partition():
    """Zwei Partikel, zwei Cluster, exakt deckungsgleich."""
    lab = np.array([0, 0, 0, 1, 1, 1])
    gt = np.array([7, 7, 7, 9, 9, 9])
    st = np.full(6, STATUS_CLEAN, np.uint8)
    m = cluster_metrics(lab, gt, st)
    assert m["purity_cluster_mean"] == 1.0
    assert m["purity_event_mean"] == 1.0
    assert m["frag_mean"] == 1.0
    assert m["n_clusters"] == m["n_gt"] == 2
    assert m["n_singletons"] == 0


def test_fragmentierung_von_hand():
    """Ein Partikel in DREI Cluster zerlegt -> Frag 3.0, Reinheit bleibt 1.0.

    Zeigt, warum beide Zahlen zusammengehoeren: perfekte Reinheit trotz
    voelliger Zerstueckelung. BEIDE Reinheiten sind hier 1.0 -- die
    Gewichtung hilft gegen Verschmelzung, nicht gegen Zerteilung.
    """
    lab = np.array([0, 0, 1, 1, 2, 2])
    gt = np.full(6, 5)
    st = np.full(6, STATUS_CLEAN, np.uint8)
    m = cluster_metrics(lab, gt, st)
    assert m["purity_cluster_mean"] == 1.0
    assert m["purity_event_mean"] == 1.0
    assert m["frag_mean"] == 3.0 and m["frag_max"] == 3


def test_reinheit_von_hand():
    """Ein Cluster mit 3x Partikel 1 und 1x Partikel 2 -> Reinheit 0.75."""
    lab = np.zeros(4, np.int64)
    gt = np.array([1, 1, 1, 2])
    st = np.full(4, STATUS_CLEAN, np.uint8)
    m = cluster_metrics(lab, gt, st)
    assert m["purity_cluster_mean"] == pytest.approx(0.75)
    assert m["purity_event_mean"] == pytest.approx(0.75)   # ein Cluster
    assert m["purity_lt90"] == 1
    assert m["purity_lt90_frac"] == pytest.approx(1.0)
    assert m["frag_mean"] == 1.0        # beide GT jeweils in EINEM Cluster


def test_nur_saubere_events_zaehlen():
    """Mehrdeutige und Rausch-Events duerfen die Metrik nicht beeinflussen."""
    lab = np.array([0, 0, 0, 0])
    gt = np.array([1, 1, 99, 99])
    st = np.array([STATUS_CLEAN, STATUS_CLEAN, STATUS_AMBIG, STATUS_NOISE],
                  dtype=np.uint8)
    m = cluster_metrics(lab, gt, st)
    assert m["n_events"] == 2 and m["n_gt"] == 1
    assert m["purity_cluster_mean"] == 1.0


def test_negative_label_werden_ignoriert():
    """RAMA vergibt -1 fuer Knoten ohne Kante -- die gehoeren keinem Cluster."""
    lab = np.array([-1, 0, 0])
    gt = np.array([1, 1, 1])
    st = np.full(3, STATUS_CLEAN, np.uint8)
    assert cluster_metrics(lab, gt, st)["n_events"] == 2


@pytest.mark.parametrize("seed", range(8))
def test_gegen_naive_referenz(seed):
    """Zufaellige Partitionen: vektorisiert == naiv, in ALLEN Feldern."""
    rng = np.random.default_rng(seed)
    n = 400
    lab = rng.integers(-1, 25, n)
    gt = rng.integers(1, 12, n)
    st = rng.choice([STATUS_CLEAN, STATUS_AMBIG, STATUS_NOISE], n,
                    p=[0.9, 0.05, 0.05]).astype(np.uint8)
    t = np.sort(rng.integers(0, 20_000, n)).astype(np.int64)
    a = cluster_metrics(lab, gt, st, t=t)
    b = cluster_metrics_naiv(lab, gt, st, t=t)
    assert set(a) == set(b), "Feld ungeprueft: " + str(set(a) ^ set(b))
    for k in b:
        assert a[k] == pytest.approx(b[k]), f"Abweichung in {k}"


def test_leere_bewertung():
    st = np.full(3, STATUS_NOISE, np.uint8)
    assert cluster_metrics(np.zeros(3), np.ones(3), st) == {}


def test_align_findet_das_fenster():
    gt_t = np.arange(0, 1000, 10)
    ev_t = np.arange(300, 400, 10)
    sl = align_gt_window(gt_t, ev_t)
    np.testing.assert_array_equal(gt_t[sl], ev_t)


def test_align_mit_doppelten_zeitstempeln():
    """Genau der Fall, an dem die alte Zeitmasken-Loesung scheiterte."""
    gt_t = np.array([10, 20, 20, 20, 30, 40])
    ev_t = np.array([20, 20])            # nur ZWEI der drei 20er geladen
    sl = align_gt_window(gt_t, ev_t)
    assert sl == slice(1, 3)


def test_align_meldet_unpassende_gt():
    with pytest.raises(ValueError):
        align_gt_window(np.array([0, 10, 20]), np.array([5, 15]))


def test_die_beiden_reinheiten_laufen_auseinander():
    """DER Fall, fuer den purity_event_mean existiert.

    Ein grosser reiner Cluster und ein kleiner schmutziger: clustergewichtet
    zaehlen beide gleich, eventgewichtet nicht. Von Hand:
      Cluster 0: 8 Events, alle GT 1               -> 1.00
      Cluster 1: 2 Events, 1x GT 1 + 1x GT 2       -> 0.50
      cluster: (1.00+0.50)/2 = 0.75
      event  : (8+1)/10       = 0.90
    """
    lab = np.array([0] * 8 + [1, 1])
    gt = np.array([1] * 9 + [2])
    st = np.full(10, STATUS_CLEAN, np.uint8)
    m = cluster_metrics(lab, gt, st)
    assert m["purity_cluster_mean"] == pytest.approx(0.75)
    assert m["purity_event_mean"] == pytest.approx(0.90)
    assert m["cluster_size_min"] == 2 and m["cluster_size_max"] == 8


def test_sichtbarkeitsspanne_von_hand():
    """gt_visible_ms_median: Spanne je Partikel, Median darueber.

    GT 1: t 0..4000 µs -> 4 ms ; GT 2: t 1000..9000 µs -> 8 ms
    Median von [4, 8] = 6.0
    """
    lab = np.array([0, 0, 0, 1, 1, 1])
    gt = np.array([1, 1, 1, 2, 2, 2])
    st = np.full(6, STATUS_CLEAN, np.uint8)
    t = np.array([0, 2000, 4000, 1000, 5000, 9000], dtype=np.int64)
    m = cluster_metrics(lab, gt, st, t=t)
    assert m["gt_visible_ms_median"] == pytest.approx(6.0)
    assert "gt_visible_ms_median" not in cluster_metrics(lab, gt, st)

def test_edge_gt_stats_von_hand():
    """Vier Events auf einer Geraden, drei Kanten. Von Hand:

      Event  0    1    2    3
      GT     1    1    2    2
      status sauber sauber sauber MEHRDEUTIG

      Kante 0-1: gleiches Partikel, beide sauber  -> JA
      Kante 1-2: verschieden,       beide sauber  -> NEIN
      Kante 2-3: ein Endpunkt mehrdeutig          -> nicht bewertbar

      edge_ja_frac = 1/2 ; bewertbar 2/3
      dt = 1000 je Kante -> Mittel 1000
      dx = 3, dy = 4 je Kante -> dr = 5 -> Mittel 5
    """
    t = np.array([0.0, 1000.0, 2000.0, 3000.0])
    x = np.array([0.0, 3.0, 6.0, 9.0])
    y = np.array([0.0, 4.0, 8.0, 12.0])
    i = np.array([0, 1, 2])
    j = np.array([1, 2, 3])
    gt = np.array([1, 1, 2, 2])
    st = np.array([STATUS_CLEAN, STATUS_CLEAN, STATUS_CLEAN, STATUS_AMBIG],
                  dtype=np.uint8)
    m = edge_gt_stats(i, j, t, x, y, gt, st)
    assert m["edge_ja_frac"] == pytest.approx(0.5)
    assert m["edge_n_bewertbar"] == 2
    assert m["edge_bewertbar_frac"] == pytest.approx(2 / 3)
    assert m["dt_mean"] == pytest.approx(1000.0)
    assert m["dr_mean"] == pytest.approx(5.0)


def test_edge_gt_stats_ohne_bewertbare_kante():
    """Kein Label ohne zwei saubere Endpunkte -- dt/dr bleiben trotzdem."""
    t = np.array([0.0, 1000.0])
    m = edge_gt_stats(np.array([0]), np.array([1]), t, np.zeros(2),
                      np.zeros(2), np.array([1, 1]),
                      np.array([STATUS_NOISE, STATUS_CLEAN], np.uint8))
    assert "edge_ja_frac" not in m
    assert m["edge_n_bewertbar"] == 0 and "dt_mean" in m

def test_align_weist_gefilterten_stream_zurueck():
    """Die Absicherung fuer ON-only -- und sie traegt.

    Ein gefilterter Eventstrom (nur ON) ist eine Teilfolge der Label-Datei.
    Nach k gefilterten Events steht die echte GT-Position bei etwa 2k, der
    zusammenhaengende Ausschnitt aber bei start+k -- die Zeitstempel laufen
    sofort auseinander. Der Abgleich schlaegt also fehl, statt still einen
    falschen Ausschnitt zu liefern.

    KONSEQUENZ fuer ON-only: erst auf dem UNGEFILTERTEN Strom ausrichten,
    dann dieselbe Maske auf beide Seiten. Vergisst man es, bricht der Lauf
    ab -- die Meldung zeigt allerdings auf die falsche Ursache.
    """
    gt_t = np.arange(0, 1000, 10)
    nur_on = gt_t[::2]
    with pytest.raises(ValueError):
        align_gt_window(gt_t, nur_on)


def test_align_grenzfall_lauter_gleicher_zeitstempel():
    """Die EINE Luecke, dokumentiert statt behoben.

    Tragen alle Events eines Ausschnitts denselben Zeitstempel, stimmt eine
    Teilfolge in den Zeitstempeln zwangslaeufig ueberein -- der Abgleich
    liefert dann Position 0..n-1 statt der tatsaechlichen Indizes. Real
    tritt das nicht auf: der Ausschnitt muesste ueber seine volle Laenge
    zeitstempelkonstant sein, bei ~4 Events je µs also hoechstens eine
    Handvoll Events lang.
    """
    gt_t = np.full(8, 10)
    sl = align_gt_window(gt_t, gt_t[::2])
    assert sl == slice(0, 4)          # trifft Events 0-3 statt 0,2,4,6


def merge_metrics_naiv(labels, track_id, status):
    """Referenz fuer merge_metrics: direkte Schleifen, andere Maske."""
    m = np.asarray(status) != STATUS_NOISE
    lab = np.asarray(labels)[m]
    gt = np.asarray(track_id)[m]
    ok = lab >= 0
    lab, gt = lab[ok], gt[ok]
    if len(lab) == 0:
        return {}
    n_gt, minderheit, groessen, mind_abs = [], [], [], []
    for c in np.unique(lab):
        _, cnt = np.unique(gt[lab == c], return_counts=True)
        n_gt.append(len(cnt))
        minderheit.append(1.0 - cnt.max() / cnt.sum())
        groessen.append(cnt.sum())
        # Absolut gezaehlt, nicht aus dem Anteil zurueckgerechnet -- sonst
        # pruefte die Referenz dieselbe Gleitkommarechnung noch einmal.
        mind_abs.append(int(cnt.sum() - cnt.max()))
    n_gt = np.array(n_gt)
    minderheit, groessen = np.array(minderheit), np.array(groessen)
    return {"merge_n_events": int(groessen.sum()),
            "merge_n_clusters": int(len(np.unique(lab))),
            "merge_n_gt": int(len(np.unique(gt))),
            "merge_frac_strikt": float((n_gt >= 2).mean()),
            "merge_frac_10": float((minderheit >= 0.10).mean()),
            "merge_event_frac": float((minderheit * groessen).sum()
                                      / groessen.sum()),
            "merge_n_minderheit": int(sum(mind_abs)),
            "merge_n_gt_max": int(n_gt.max())}


@pytest.mark.parametrize("seed", range(8))
def test_merge_gegen_naive_referenz(seed):
    rng = np.random.default_rng(seed)
    n = 400
    lab = rng.integers(-1, 25, n)
    gt = rng.integers(1, 12, n)
    st = rng.choice([STATUS_CLEAN, STATUS_AMBIG, STATUS_NOISE], n,
                    p=[0.85, 0.10, 0.05]).astype(np.uint8)
    a = merge_metrics(lab, gt, st)
    b = merge_metrics_naiv(lab, gt, st)
    assert set(a) == set(b), "Feld ungeprueft: " + str(set(a) ^ set(b))
    for k in b:
        assert a[k] == pytest.approx(b[k]), f"Abweichung in {k}"


def test_merge_sieht_mehrdeutige_events():
    """DER Grund fuer die zweite Maske.

    Ein Cluster aus vier Events: zwei saubere von Partikel 1, zwei
    MEHRDEUTIGE von Partikel 2. cluster_metrics sieht nur die sauberen und
    meldet perfekte Reinheit. merge_metrics sieht alle vier und meldet die
    Verschmelzung -- genau die Lage, die im Ueberlappungskern entsteht.
    """
    lab = np.zeros(4, np.int64)
    gt = np.array([1, 1, 2, 2])
    st = np.array([STATUS_CLEAN, STATUS_CLEAN, STATUS_AMBIG, STATUS_AMBIG],
                  dtype=np.uint8)

    assert cluster_metrics(lab, gt, st)["purity_event_mean"] == 1.0
    m = merge_metrics(lab, gt, st)
    assert m["merge_frac_strikt"] == 1.0
    assert m["merge_frac_10"] == 1.0
    assert m["merge_event_frac"] == pytest.approx(0.5)
    assert m["merge_n_events"] == 4 and m["merge_n_gt_max"] == 2


def test_merge_ignoriert_rauschen():
    """Rauschen traegt track_id -1 und gehoert zu keinem Partikel."""
    lab = np.zeros(3, np.int64)
    gt = np.array([1, 1, -1])
    st = np.array([STATUS_CLEAN, STATUS_AMBIG, STATUS_NOISE], np.uint8)
    m = merge_metrics(lab, gt, st)
    assert m["merge_n_events"] == 2 and m["merge_frac_strikt"] == 0.0


def test_strikt_und_zehn_prozent_laufen_auseinander():
    """Der ABSTAND der beiden ist die Diagnose.

    Ein Cluster mit 99 Events von Partikel 1 und einem von Partikel 2:
    strikt zaehlt ihn als verschmolzen, die 10-%-Regel nicht. Ein einziges
    fremdes Event ist ein Ausreisser, keine Verschmelzung.
    """
    lab = np.zeros(100, np.int64)
    gt = np.array([1] * 99 + [2])
    st = np.full(100, STATUS_CLEAN, np.uint8)
    m = merge_metrics(lab, gt, st)
    assert m["merge_frac_strikt"] == 1.0
    assert m["merge_frac_10"] == 0.0
    assert m["merge_event_frac"] == pytest.approx(0.01)


def test_groesstes_fragment_unterscheidet_kruemel_von_teilung():
    """frag_mean ist in beiden Faellen 3.0 -- der Anteil nicht.

    A: 10 Events in drei gleiche Teile -> groesstes Fragment ~40 %
    B: 10 Events als 8 + 1 + 1        -> groesstes Fragment 80 %
    """
    st = np.full(10, STATUS_CLEAN, np.uint8)
    gt = np.ones(10, np.int64)
    a = cluster_metrics(np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2]), gt, st)
    b = cluster_metrics(np.array([0] * 8 + [1, 2]), gt, st)
    assert a["frag_mean"] == b["frag_mean"] == 3.0
    assert a["frag_groesster_anteil_median"] == pytest.approx(0.4)
    assert b["frag_groesster_anteil_median"] == pytest.approx(0.8)


def test_merge_leer():
    st = np.full(3, STATUS_NOISE, np.uint8)
    assert merge_metrics(np.zeros(3), np.ones(3), st) == {}


def _bbox_paar(t_und_positionen, box_w=11.0):
    """Minimales bbox-Array aus (t, [(x_zentrum, y_zentrum), ...])."""
    from conftest import BBOX_DTYPE
    eintraege = [(t, cx - box_w / 2, cy - box_w / 2, k)
                 for t, zentren in t_und_positionen
                 for k, (cx, cy) in enumerate(zentren)]
    b = np.zeros(len(eintraege), dtype=BBOX_DTYPE)
    for i, (t, x, y, tid) in enumerate(eintraege):
        b[i] = (t, x, y, box_w, box_w, 0, tid, 1.0)
    return b


def test_abstand_je_partikel_von_hand():
    """Drei Partikel: zwei nah beieinander, eines weit weg. Das globale
    Minimum waere 5 px fuer alle drei -- je Partikel ist es 5, 5 und 400."""
    from eventpiv.bewerten.evaluate import min_abstand_je_partikel

    bbox = _bbox_paar([(1000, [(100.0, 100.0), (105.0, 100.0),
                               (500.0, 100.0)])])
    raster_t, ids, d = min_abstand_je_partikel(bbox)
    np.testing.assert_array_equal(ids, [0, 1, 2])
    assert d[0, 0] == pytest.approx(5.0)
    assert d[0, 1] == pytest.approx(5.0)
    assert d[0, 2] == pytest.approx(395.0)


def test_einzelnes_partikel_ist_unendlich_weit():
    from eventpiv.bewerten.evaluate import min_abstand_je_partikel

    _, _, d = min_abstand_je_partikel(_bbox_paar([(1000, [(100.0, 100.0)])]))
    assert np.isinf(d[0, 0])


def test_fenster_umschliesst_das_raster():
    """DER Grund fuer die umschliessende statt filternde Auswahl: bei
    16 667 µs Raster und 10 000 µs Fenster enthaelt [20 000, 30 000) keinen
    einzigen Rasterpunkt."""
    from eventpiv.bewerten.evaluate import abstand_im_fenster

    raster_t = np.array([0, 16667, 33334])
    ids = np.array([0, 1])
    d = np.array([[100.0, 100.0], [4.0, 4.0], [100.0, 100.0]])
    np.testing.assert_allclose(
        abstand_im_fenster(raster_t, ids, d, 20_000, 30_000), [4.0, 4.0])


def test_abstaende_kommen_aus_der_w_spalte():
    """Verschieden grosse Boxen: die Mitte folgt der ZEILE.

    Ein gleichmaessiger Versatz aller Zentren aendert die Abstaende nicht
    -- deshalb braucht dieser Test Boxen VERSCHIEDENER Groesse. Partikel 0
    hat w = 4 an der Ecke (98, 98), Zentrum also (100, 100); Partikel 1
    hat w = 10 an der Ecke (105, 95), Zentrum (110, 100). Abstand 10,0 px.
    Mit einer Konstanten von 11 laegen die Zentren bei (103.5, 103.5) und
    (110.5, 100.5), Abstand 7,62 px.

    _bbox_paar wird hier NICHT benutzt: es setzt w und h fuer alle
    Eintraege gleich, und genau davon soll dieser Test unabhaengig sein.
    """
    from conftest import BBOX_DTYPE
    from eventpiv.bewerten.evaluate import min_abstand_je_partikel

    b = np.zeros(2, dtype=BBOX_DTYPE)
    b[0] = (1000, 98.0, 98.0, 4.0, 4.0, 0, 0, 1.0)
    b[1] = (1000, 105.0, 95.0, 10.0, 10.0, 0, 1, 1.0)
    _, ids, d = min_abstand_je_partikel(b)
    np.testing.assert_array_equal(ids, [0, 1])
    np.testing.assert_allclose(d[0], [10.0, 10.0])


def test_nah_maske_je_event():
    """Die Maske haengt am PARTIKEL des Events, nicht an seiner Position."""
    from eventpiv.bewerten.evaluate import nah_maske

    ids = np.array([1, 2, 3])
    d = np.array([5.0, 5.0, 400.0])          # 1 und 2 nah, 3 fern
    track_id = np.array([1, 3, 2, -1, 3])
    np.testing.assert_array_equal(
        nah_maske(track_id, ids, d),
        [True, False, True, False, False])   # -1 gehoert zu keinem Partikel


def test_merge_metrics_mit_teilmenge():
    """extra_maske bildet die Cluster nur ueber die Teilmenge."""
    lab = np.zeros(4, np.int64)
    gt = np.array([1, 1, 2, 2])
    st = np.full(4, STATUS_CLEAN, np.uint8)
    ganz = merge_metrics(lab, gt, st)
    teil = merge_metrics(lab, gt, st, extra_maske=np.array([1, 1, 0, 0], bool))
    assert ganz["merge_event_frac"] == pytest.approx(0.5)
    assert teil["merge_event_frac"] == 0.0     # nur Partikel 1 uebrig
    assert teil["merge_n_events"] == 2
    assert ganz["merge_n_minderheit"] == 2


def test_kreuztabelle_zellen_von_hand():
    """Die ungefaltete Kreuztabelle, von Hand ausgezaehlt.

    _kreuztabelle war bisher nur ueber cluster_metrics und merge_metrics
    mitgeprueft. Die zell_-Felder gehen in keine der beiden ein, waeren
    dort also unbeobachtet.

      Cluster 0: 3x GT 5, 1x GT 7
      Cluster 2: 2x GT 7
    """
    from eventpiv.bewerten.evaluate import _kreuztabelle

    lab = np.array([0, 0, 0, 0, 2, 2])
    gt = np.array([5, 5, 5, 7, 7, 7])
    kt = _kreuztabelle(lab, gt, np.ones(6, bool))

    np.testing.assert_array_equal(kt["lab_u"], [0, 2])
    np.testing.assert_array_equal(kt["gt_u"], [5, 7])
    zellen = dict(zip(zip(kt["zell_lab"], kt["zell_gt"]), kt["zell_anzahl"]))
    assert zellen == {(0, 0): 3, (0, 1): 1, (1, 1): 2}
    assert kt["zell_anzahl"].sum() == 6


def test_kreuztabelle_zellen_ohne_bewertbare_events():
    from eventpiv.bewerten.evaluate import _kreuztabelle

    assert _kreuztabelle(np.array([-1, -1]), np.array([1, 1]),
                         np.ones(2, bool)) is None


def hota_naiv(labels, track_id, status):
    """Referenz nach den DEFINITIONEN des Papers, ueber Detektionspaare.

    Gleichungen 10 bis 12: fuer jedes TP c werden TPA(c), FNA(c) und
    FPA(c) explizit ausgezaehlt. Die produktive Fassung rechnet
    stattdessen auf Zellzahlen der Kreuztabelle -- das ist die
    Abkuerzung, die das Paper selbst nennt, und genau die gehoert
    gegen die Definition geprueft.

    Die fuenf Kennzahlen aus der Verteilung entstehen hier durch
    Gruppierung per Schleife statt per bincount. Die Aggregation danach
    (min, percentile, zaehlen) ist in beiden Fassungen derselbe Aufruf
    und traegt keine eigene Definition.
    """
    labels, track_id = np.asarray(labels), np.asarray(track_id)
    ist_gt = np.asarray(status) != STATUS_NOISE
    ist_pr = labels >= 0
    tp = np.flatnonzero(ist_gt & ist_pr)
    fn = np.flatnonzero(ist_gt & ~ist_pr)
    fp = np.flatnonzero(~ist_gt & ist_pr)
    if len(tp) == 0:
        return {}

    aa, rr, pp = [], [], []
    for c in tp:
        g, p = track_id[c], labels[c]
        n_tpa = sum(1 for k in tp if labels[k] == p and track_id[k] == g)
        n_fna = (sum(1 for k in tp if labels[k] != p and track_id[k] == g)
                 + sum(1 for k in fn if track_id[k] == g))
        n_fpa = (sum(1 for k in tp if labels[k] == p and track_id[k] != g)
                 + sum(1 for k in fp if labels[k] == p))
        aa.append(n_tpa / (n_tpa + n_fna + n_fpa))
        rr.append(n_tpa / (n_tpa + n_fna))
        pp.append(n_tpa / (n_tpa + n_fpa))

    # Dieselben A(c), nach GT-Partikel gruppiert -- per Schleife, damit
    # die Gruppierung selbst geprueft wird und nicht nur die Aggregation.
    a_je_ptkl = {}
    for c, wert in zip(tp, aa):
        a_je_ptkl.setdefault(int(track_id[c]), []).append(wert)
    ass_a_ptkl = np.array([float(np.mean(v)) for v in a_je_ptkl.values()])

    # Effektive Clusterzahl je Partikel: Kehrwert des Simpson-Index ueber
    # die Verteilung seiner zugeordneten Events auf die Cluster.
    eff, pr_je_ptkl = [], {}
    for c, wert in zip(tp, pp):
        pr_je_ptkl.setdefault(int(track_id[c]), []).append(wert)
    for g in a_je_ptkl:
        cl = {}
        for k in tp:
            if track_id[k] == g:
                cl[int(labels[k])] = cl.get(int(labels[k]), 0) + 1
        n_g = sum(cl.values())
        eff.append(n_g ** 2 / sum(v * v for v in cl.values()))
    ass_pr_ptkl = np.array([float(np.mean(v)) for v in pr_je_ptkl.values()])

    det_a = len(tp) / (len(tp) + len(fn) + len(fp))
    ass_a = float(np.mean(aa))
    return {"ass_a_min": float(ass_a_ptkl.min()),
            "ass_a_p10": float(np.percentile(ass_a_ptkl, 10)),
            "ass_a_lt90": int((ass_a_ptkl < 0.90).sum()),
            "ass_a_lt90_frac": float((ass_a_ptkl < 0.90).mean()),
            "ass_pr_min": float(ass_pr_ptkl.min()),
            "ass_pr_lt90": int((ass_pr_ptkl < 0.90).sum()),
            "n_cluster_eff_max": float(max(eff)),
            "event_tp": len(tp), "event_fn": len(fn), "event_fp": len(fp),
            "det_a": float(det_a),
            "det_re": float(len(tp) / (len(tp) + len(fn))),
            "det_pr": float(len(tp) / (len(tp) + len(fp))),
            "ass_a": ass_a, "ass_re": float(np.mean(rr)),
            "ass_pr": float(np.mean(pp)),
            "hota": float(np.sqrt(det_a * ass_a))}


@pytest.mark.parametrize("seed", range(8))
def test_hota_gegen_naive_referenz(seed):
    """Zellrechnung == Definition ueber Detektionspaare, in ALLEN Feldern."""
    from eventpiv.bewerten.evaluate import tracking_metrics

    rng = np.random.default_rng(seed)
    n = 200
    lab = rng.integers(-1, 9, n)
    gt = rng.integers(1, 5, n)
    st = rng.choice([STATUS_CLEAN, STATUS_AMBIG, STATUS_NOISE], n,
                    p=[0.7, 0.15, 0.15]).astype(np.uint8)
    gt = np.where(st == STATUS_NOISE, -1, gt)   # Vertrag aus labels.py
    a, b = tracking_metrics(lab, gt, st), hota_naiv(lab, gt, st)
    assert set(a) == set(b), "Feld ungeprueft: " + str(set(a) ^ set(b))
    for k in b:
        assert a[k] == pytest.approx(b[k]), f"Abweichung in {k}"


def test_hota_perfekte_partition():
    from eventpiv.bewerten.evaluate import tracking_metrics

    m = tracking_metrics(np.array([0, 0, 0, 1, 1, 1]),
                         np.array([7, 7, 7, 9, 9, 9]),
                         np.full(6, STATUS_CLEAN, np.uint8))
    assert m["hota"] == m["det_a"] == m["ass_a"] == 1.0
    assert m["event_fn"] == m["event_fp"] == 0


def test_hota_reduziert_sich_bei_einem_partikel_auf_jaccard():
    """Gleichung 31 des Papers: bei genau EINEM Partikel fallen
    Detektions- und Assoziationsscore zusammen, die Wurzel hebt die
    Doppelzaehlung auf, und HOTA ist der reine Jaccard-Index.

    Zwei von vier Events eines Partikels sind zugeordnet:
    2 / (2 + 2 + 0) = 0.5.
    """
    from eventpiv.bewerten.evaluate import tracking_metrics

    m = tracking_metrics(np.array([0, 0, -1, -1]), np.ones(4, np.int64),
                         np.full(4, STATUS_CLEAN, np.uint8))
    assert m["hota"] == pytest.approx(0.5)
    assert m["det_a"] == pytest.approx(0.5)


def test_mehrdeutige_events_sind_keine_false_positives():
    """DIE Abgrenzung. Ein mehrdeutiges Event traegt eine track_id und
    ist ein gueltiges gtDet -- ein falsch zugeordnetes erzeugt einen
    ASSOZIATIONS-, keinen Detektionsfehler.

    Zwei Partikel in EINEM Cluster, die Haelfte davon mehrdeutig:
    det_a bleibt perfekt, ass_pr faellt.
    """
    from eventpiv.bewerten.evaluate import tracking_metrics

    m = tracking_metrics(np.zeros(4, np.int64), np.array([1, 1, 2, 2]),
                         np.array([STATUS_CLEAN, STATUS_AMBIG,
                                   STATUS_CLEAN, STATUS_AMBIG], np.uint8))
    assert m["event_fp"] == 0 and m["event_fn"] == 0
    assert m["det_a"] == 1.0
    assert m["ass_pr"] == pytest.approx(0.5)   # jeder Cluster halb fremd


def test_rauschen_im_cluster_ist_ein_false_positive():
    """Die einzige FP-Quelle: ein Rauschevent in einer Trajektorie.
    Nach Journal 15.08. sind Rausch-Tracklets in echte Trajektorien
    eingewandert -- bisher erfasst das keine Kennzahl."""
    from eventpiv.bewerten.evaluate import tracking_metrics

    m = tracking_metrics(np.zeros(3, np.int64), np.array([1, 1, -1]),
                         np.array([STATUS_CLEAN, STATUS_CLEAN,
                                   STATUS_NOISE], np.uint8))
    assert m["event_tp"] == 2 and m["event_fp"] == 1
    assert m["det_pr"] == pytest.approx(2 / 3)
    assert m["det_re"] == 1.0


def test_die_sechs_kennzahlensaetze_kollidieren_nicht():
    """cluster_metrics, merge_metrics, tracking_metrics, idf1_metrics,
     idsw_metrics und fa_hota_metrics werden in solve2 in EIN dict gemischt.
    Ein gemeinsamer Schluessel wuerde still ueberschreiben."""
    from eventpiv.bewerten.evaluate import tracking_metrics

    lab = np.array([0, 0, 1, 1])
    gt = np.array([1, 1, 2, 2])
    st = np.full(4, STATUS_CLEAN, np.uint8)
    from eventpiv.bewerten.evaluate import idf1_metrics, idsw_metrics
    t = np.arange(len(lab))
    saetze = [set(cluster_metrics(lab, gt, st, t=t)),
              set(merge_metrics(lab, gt, st)),
              set(tracking_metrics(lab, gt, st)), set(idf1_metrics(lab, gt, st)),
              set(idsw_metrics(lab, gt, st, t)), set(fa_hota_metrics(lab, gt, st, t))]
    for i, x in enumerate(saetze):
        for y in saetze[i + 1:]:
            assert x & y == set(), f"gemeinsamer Schluessel: {x & y}"


def test_hota_leer():
    from eventpiv.bewerten.evaluate import tracking_metrics

    st = np.full(3, STATUS_NOISE, np.uint8)
    assert tracking_metrics(np.full(3, -1), np.full(3, -1), st) == {}


@pytest.mark.parametrize("seed", range(6))
def test_zerlegung_ergibt_das_laufmittel(seed):
    """Die Probe auf die ganze Idee: das eventgewichtete Mittel der
    Partikelwerte IST ass_a. Weicht es ab, ist eine der beiden
    Funktionen falsch."""
    from eventpiv.bewerten.evaluate import (tracking_metrics,
                                            tracking_metrics_je_partikel)
    rng = np.random.default_rng(seed)
    n = 400
    lab = rng.integers(-1, 12, n)
    gt = rng.integers(1, 7, n)
    st = rng.choice([STATUS_CLEAN, STATUS_AMBIG, STATUS_NOISE], n,
                    p=[0.7, 0.15, 0.15]).astype(np.uint8)
    gt = np.where(st == STATUS_NOISE, -1, gt)

    g, jp = tracking_metrics(lab, gt, st), tracking_metrics_je_partikel(lab, gt, st)
    gewicht = jp["n_ptkl"] / g["event_tp"]
    assert gewicht.sum() == pytest.approx(1.0)
    for feld in ("ass_a", "ass_re", "ass_pr"):
        assert (gewicht * jp[f"{feld}_ptkl"]).sum() == pytest.approx(g[feld])


def test_signatur_identitaetswechsel():
    """Zwei Partikel tauschen zur Haelfte die Cluster. BEIDE Masse fallen,
    und beide Partikel bekommen zwei substanzielle Cluster -- die Signatur
    tritt paarweise auf."""
    from eventpiv.bewerten.evaluate import tracking_metrics_je_partikel

    gt = np.array([1]*10 + [2]*10)
    lab = np.array([0]*5 + [1]*5 + [1]*5 + [0]*5)
    jp = tracking_metrics_je_partikel(lab, gt, np.zeros(20, np.uint8))
    assert jp["n_cluster_eff"] == pytest.approx([2.0, 2.0])
    assert jp["ass_re_ptkl"] == pytest.approx([0.5, 0.5])
    assert jp["ass_pr_ptkl"] == pytest.approx([0.5, 0.5])


def test_signatur_kruemel_unterscheidet_sich():
    """Ein Partikel verliert ein einzelnes Event an einen Splitter:
    ass_re faellt leicht, ass_pr bleibt 1, n_cluster_eff bleibt nahe 1.
    Genau das trennt den harmlosen Fall vom Identitaetswechsel."""
    from eventpiv.bewerten.evaluate import tracking_metrics_je_partikel

    gt = np.ones(20, np.int64)
    lab = np.array([0]*19 + [1])
    jp = tracking_metrics_je_partikel(lab, gt, np.zeros(20, np.uint8))
    assert jp["n_cluster_eff"][0] < 1.15    # 19 + 1 Event: 1.108...
    assert jp["ass_pr_ptkl"] == pytest.approx([1.0])
    assert jp["ass_re_ptkl"][0] < 1.0


def test_je_partikel_leer():
    from eventpiv.bewerten.evaluate import tracking_metrics_je_partikel
    assert tracking_metrics_je_partikel(
        np.full(3, -1), np.full(3, -1), np.full(3, STATUS_NOISE, np.uint8)) == {}


def test_viele_kleine_stuecke_werden_nicht_kleingerechnet():
    """Der Fehler, den eine 10-%-Schwelle gemacht haette: dreizehn
    Stuecke zu je 7,7 % lagen ALLE darunter und wurden als null
    substanzielle Cluster gezaehlt -- je schlimmer der Fall, desto
    kleiner die Zahl. n_cluster_eff ist monoton."""
    from eventpiv.bewerten.evaluate import tracking_metrics_je_partikel

    gt = np.ones(26, np.int64)
    st = np.zeros(26, np.uint8)
    zwei = tracking_metrics_je_partikel(np.repeat([0, 1], 13), gt, st)
    dreizehn = tracking_metrics_je_partikel(np.repeat(np.arange(13), 2), gt, st)
    assert zwei["n_cluster_eff"] == pytest.approx([2.0])
    assert dreizehn["n_cluster_eff"] == pytest.approx([13.0])


def idf1_naiv(labels, track_id, status):
    """Referenz: das beste Matching durch vollstaendige AUFZAEHLUNG.

    Alle injektiven Teilzuordnungen GT -> Cluster, Nichtzuordnung
    ausdruecklich erlaubt. Nur fuer winzige Beispiele brauchbar --
    genau deshalb ist es die richtige Probe darauf, dass das
    rechteckige Zwangs-Matching der produktiven Fassung dasselbe
    Optimum trifft.
    """
    import itertools

    labels, track_id = np.asarray(labels), np.asarray(track_id)
    ist_gt = np.asarray(status) != STATUS_NOISE
    ist_pr = labels >= 0
    tp = np.flatnonzero(ist_gt & ist_pr)
    if len(tp) == 0:
        return {}

    gt_u = sorted(set(int(v) for v in track_id[tp]))
    lab_u = sorted(set(int(v) for v in labels[tp]))
    c = {(g, p): sum(1 for k in tp
                     if track_id[k] == g and labels[k] == p)
         for g in gt_u for p in lab_u}

    bestes = 0
    for wahl in itertools.product(lab_u + [None], repeat=len(gt_u)):
        echt = [w for w in wahl if w is not None]
        if len(echt) != len(set(echt)):
            continue
        bestes = max(bestes, sum(c[(g, w)] for g, w in zip(gt_u, wahl)
                                 if w is not None))

    n_gtdet = int(ist_gt.sum())
    n_prdet = int(ist_pr.sum())
    return {"idtp": bestes,
            "idfn": n_gtdet - bestes, "idfp": n_prdet - bestes,
            "idr": float(bestes / n_gtdet), "idp": float(bestes / n_prdet),
            "idf1": float(2 * bestes / (n_gtdet + n_prdet))}


@pytest.mark.parametrize("seed", range(8))
def test_idf1_gegen_vollstaendige_aufzaehlung(seed):
    from eventpiv.bewerten.evaluate import idf1_metrics

    rng = np.random.default_rng(seed)
    n = 40
    lab = rng.integers(-1, 4, n)
    gt = rng.integers(1, 4, n)
    st = rng.choice([STATUS_CLEAN, STATUS_AMBIG, STATUS_NOISE], n,
                    p=[0.7, 0.15, 0.15]).astype(np.uint8)
    gt = np.where(st == STATUS_NOISE, -1, gt)

    a, b = idf1_metrics(lab, gt, st), idf1_naiv(lab, gt, st)
    assert set(a) == set(b), "Feld ungeprueft: " + str(set(a) ^ set(b))
    for k in b:
        assert a[k] == pytest.approx(b[k]), f"Abweichung in {k}"


def test_idf1_perfekte_partition():
    from eventpiv.bewerten.evaluate import idf1_metrics

    m = idf1_metrics(np.array([0, 0, 0, 1, 1, 1]), np.array([7, 7, 7, 9, 9, 9]),
                     np.full(6, STATUS_CLEAN, np.uint8))
    assert m["idf1"] == 1.0 and m["idfn"] == 0 and m["idfp"] == 0


def test_idf1_bestraft_fragmentierung_haerter_als_hota():
    """DIE Kritik am Maß, als Test festgehalten. Ein Partikel in vier
    gleichen Stuecken: IDF1 matcht genau EINEN Cluster, die uebrigen
    drei Viertel der Events zaehlen als IDFP UND als IDFN -- obwohl
    jeder Cluster perfekt rein ist. HOTA sieht dieselbe Partition
    milder, weil es ueber alle Zuordnungen wertet."""
    from eventpiv.bewerten.evaluate import idf1_metrics, tracking_metrics

    lab = np.repeat(np.arange(4), 10)
    gt = np.ones(40, np.int64)
    st = np.zeros(40, np.uint8)
    assert idf1_metrics(lab, gt, st)["idf1"] == pytest.approx(0.25)
    assert tracking_metrics(lab, gt, st)["hota"] > 0.4


def idsw_naiv(labels, track_id, status, t, delta=None):
    """Referenz: Schleife ueber Partikel, Einheiten in Zeitordnung.

    Haelt sich woertlich an Gl. 44 -- `prev` ist die zuletzt VERFOLGTE
    Einheit desselben Partikels, eine Luecke setzt nicht zurueck.
    """
    labels, track_id = np.asarray(labels), np.asarray(track_id)
    t = np.asarray(t)
    ist_gt = np.asarray(status) != STATUS_NOISE
    idx = np.flatnonzero(ist_gt)
    if len(idx) == 0:
        return {}

    einheiten = {}
    for k in idx:
        schl = int(t[k]) if delta is None else int(t[idx].min()) + \
            int((int(t[k]) - int(t[idx].min())) // int(delta))
        einheiten.setdefault(int(track_id[k]), {}).setdefault(schl, []).append(
            int(labels[k]))

    idsw_je, frag_je = [], []
    for g, nach_zeit in einheiten.items():
        idsw = frag = 0
        letzte = None
        war_verfolgt = False
        for schl in sorted(nach_zeit):
            zug = [l for l in nach_zeit[schl] if l >= 0]
            if delta is None:
                # Eventebene: jedes Event ist eine eigene Einheit; bei
                # gleichem Zeitstempel in Stromreihenfolge.
                folge = nach_zeit[schl]
            else:
                # Gleichstand: kleinste Cluster-ID. min() ueber die
                # sortierten Kandidaten macht die Regel sichtbar, statt
                # sie der Iterationsreihenfolge von set() zu ueberlassen.
                folge = [min(sorted(set(zug)), key=lambda l: -zug.count(l))] \
                    if zug else [-1]
            for lab in folge:
                if lab < 0:
                    war_verfolgt = False
                    continue
                if letzte is not None and lab != letzte:
                    idsw += 1
                if letzte is not None and not war_verfolgt:
                    frag += 1
                letzte, war_verfolgt = lab, True
        idsw_je.append(idsw)
        frag_je.append(frag)

    return {"idsw": sum(idsw_je), "idsw_max": max(idsw_je),
            "lhn_frag": sum(frag_je), "lhn_frag_max": max(frag_je)}


@pytest.mark.parametrize("delta", [None, 3, 10])
@pytest.mark.parametrize("seed", range(6))
def test_idsw_gegen_naive_referenz(seed, delta):
    from eventpiv.bewerten.evaluate import idsw_metrics

    rng = np.random.default_rng(seed)
    n = 120
    lab = rng.integers(-1, 4, n)
    gt = rng.integers(1, 4, n)
    st = rng.choice([STATUS_CLEAN, STATUS_AMBIG, STATUS_NOISE], n,
                    p=[0.7, 0.15, 0.15]).astype(np.uint8)
    gt = np.where(st == STATUS_NOISE, -1, gt)
    t = np.sort(rng.integers(0, 60, n))          # mit Zeitgleichstaenden

    a, b = idsw_metrics(lab, gt, st, t, delta), idsw_naiv(lab, gt, st, t, delta)
    for k in b:
        assert a[k] == b[k], f"{k}: {a[k]} != {b[k]} (delta={delta})"


def test_idsw_luecke_setzt_nicht_zurueck():
    """Gl. 44: prev ist das ZULETZT VERFOLGTE Ereignis. Verschwindet ein
    Partikel und kommt mit anderer ID wieder, ist das ein Wechsel -- und
    zugleich eine Fragmentierung. Die beiden sind orthogonal, hier
    treten sie zusammen auf."""
    from eventpiv.bewerten.evaluate import idsw_metrics

    lab = np.array([0, 0, -1, -1, 1, 1])
    gt = np.ones(6, np.int64)
    st = np.zeros(6, np.uint8)
    m = idsw_metrics(lab, gt, st, np.arange(6))
    assert m["idsw"] == 1
    assert m["lhn_frag"] == 1


def test_frag_ohne_idsw():
    """Unterbrechung, danach dieselbe ID: Frag zaehlt, IDSW nicht."""
    from eventpiv.bewerten.evaluate import idsw_metrics

    m = idsw_metrics(np.array([0, 0, -1, 0, 0]), np.ones(5, np.int64),
                     np.zeros(5, np.uint8), np.arange(5))
    assert m["lhn_frag"] == 1 and m["idsw"] == 0


def test_idsw_ohne_frag():
    """Wechsel ohne Unterbrechung: IDSW zaehlt, Frag nicht."""
    from eventpiv.bewerten.evaluate import idsw_metrics

    m = idsw_metrics(np.array([0, 0, 1, 1]), np.ones(4, np.int64),
                     np.zeros(4, np.uint8), np.arange(4))
    assert m["idsw"] == 1 and m["lhn_frag"] == 0


def test_fuehrende_luecke_ist_keine_fragmentierung():
    """Vor dem ersten verfolgten Ereignis existiert nichts, was
    unterbrochen werden koennte."""
    from eventpiv.bewerten.evaluate import idsw_metrics

    m = idsw_metrics(np.array([-1, -1, 0, 0]), np.ones(4, np.int64),
                     np.zeros(4, np.uint8), np.arange(4))
    assert m["lhn_frag"] == 0 and m["idsw"] == 0


def test_delta_glaettet_verschraenkte_tracklets():
    """DER Grund fuer die Rasterung. Zwei Cluster desselben Partikels
    liegen zeitlich verschraenkt: auf Eventebene sind das fuenf Wechsel,
    mit delta = 10 wird daraus einer."""
    from eventpiv.bewerten.evaluate import idsw_metrics

    lab = np.array([0, 1, 0, 1, 0, 1, 1, 1, 1, 1])
    gt = np.ones(10, np.int64)
    st = np.zeros(10, np.uint8)
    t = np.arange(10)
    assert idsw_metrics(lab, gt, st, t)["idsw"] == 5
    assert idsw_metrics(lab, gt, st, t, delta=10)["idsw"] == 0


def test_idsw_leer():
    from eventpiv.bewerten.evaluate import idsw_metrics

    assert idsw_metrics(np.full(3, -1), np.full(3, -1),
                        np.full(3, STATUS_NOISE, np.uint8), np.arange(3)) == {}


def test_gleichstand_im_bin_waehlt_die_kleinste_cluster_id():
    """Zwei Cluster mit derselben Eventzahl in einem Bin. Die Wahl ist
    willkuerlich, aber sie muss FESTGELEGT sein -- ohne Regel entscheidet
    die interne Reihenfolge, und produktive und naive Fassung liefen
    auseinander."""
    from eventpiv.bewerten.evaluate import idsw_metrics

    # Bin 0: je zweimal Cluster 5 und 9. Bin 1: nur Cluster 5.
    lab = np.array([9, 5, 9, 5, 5, 5])
    gt = np.ones(6, np.int64)
    st = np.zeros(6, np.uint8)
    t = np.array([0, 1, 2, 3, 10, 11])
    assert idsw_metrics(lab, gt, st, t, delta=10)["idsw"] == 0

def _bbox_mit_zeiten(t):
    """Minimales bbox-Array: nur das Feld t wird gelesen."""
    return np.array([(tt,) for tt in t], dtype=[("t", np.int64)])


def test_gt_raster_dt_von_hand():
    """60 Hz auf ganzzahligen Zeitstempeln: die Abstaende wechseln
    zwischen 16.666 und 16.667 µs. Eine Gleichheitspruefung scheiterte
    daran, der Median nicht."""
    from eventpiv.bewerten.evaluate import gt_raster_dt

    t = np.round(np.arange(6) * 50_000 / 3).astype(np.int64)
    assert set(np.diff(t)) == {16_666, 16_667}      # wirklich ungleich
    assert gt_raster_dt(_bbox_mit_zeiten(t)) == pytest.approx(16_666.5, abs=1)


def test_gt_raster_dt_meldet_luecke():
    from eventpiv.bewerten.evaluate import gt_raster_dt

    t = np.array([0, 16_666, 33_333, 100_000], np.int64)
    with pytest.raises(ValueError, match="nicht gleichmaessig"):
        gt_raster_dt(_bbox_mit_zeiten(t))


def test_gt_raster_dt_braucht_zwei_punkte():
    from eventpiv.bewerten.evaluate import gt_raster_dt

    with pytest.raises(ValueError, match="nicht bestimmbar"):
        gt_raster_dt(_bbox_mit_zeiten(np.array([0], np.int64)))


def fa_hota_naiv(labels, track_id, status, t):
    """Referenz nach der Definition: Fragmente durch Ablaufen der
    verschmolzenen Zeitreihe je Zelle. Nur fuer winzige Beispiele."""
    labels, track_id = np.asarray(labels), np.asarray(track_id)
    t = np.asarray(t)
    ist_gt = np.asarray(status) != STATUS_NOISE
    ist_pr = labels >= 0
    tp = [int(k) for k in np.flatnonzero(ist_gt & ist_pr)]
    if not tp:
        return {}
    n_tp = len(tp)
    n_fn = int((ist_gt & ~ist_pr).sum())
    n_fp = int((~ist_gt & ist_pr).sum())

    summe_f = summe_af = summe_a = 0.0
    laengen = []
    zellen = sorted({(int(track_id[k]), int(labels[k])) for k in tp})
    for g, p in zellen:
        in_g = [k for k in range(len(labels)) if ist_gt[k] and track_id[k] == g]
        in_p = [k for k in range(len(labels)) if ist_pr[k] and labels[k] == p]
        reihe = sorted(set(in_g) | set(in_p), key=lambda k: (t[k], k))
        c_zelle = set(in_g) & set(in_p)
        nenner = len(in_g) + len(in_p) - len(c_zelle)
        a = len(c_zelle) / nenner
        summe_a += len(c_zelle) * a
        lauf = 0
        for k in reihe + [None]:
            if k is not None and k in c_zelle:
                lauf += 1
                continue
            if lauf:
                laengen.append(lauf)
                f = lauf / nenner
                summe_f += lauf * f
                summe_af += lauf * (a * f) ** 0.5
            lauf = 0
    return {"hota_frag_a": summe_f / n_tp,
            "fa_hota": (summe_af / (n_tp + n_fn + n_fp)) ** 0.5,
            "hota_einheiten": ((n_tp / (n_tp + n_fn + n_fp))
                               * (summe_a / n_tp)) ** 0.5,
            "hota_n_fragmente": len(laengen),
            "hota_fragment_len_median": float(np.median(laengen)),
            "hota_fragment_len_max": int(max(laengen))}


@pytest.mark.parametrize("seed", range(8))
def test_fa_hota_gegen_naive_referenz(seed):
    from eventpiv.bewerten.evaluate import fa_hota_metrics

    rng = np.random.default_rng(seed)
    n = 60
    lab = rng.integers(-1, 4, n)
    gt = rng.integers(1, 4, n)
    st = rng.choice([STATUS_CLEAN, STATUS_AMBIG, STATUS_NOISE], n,
                    p=[0.7, 0.15, 0.15]).astype(np.uint8)
    gt = np.where(st == STATUS_NOISE, -1, gt)
    t = np.sort(rng.integers(0, 40, n))

    a, b = fa_hota_metrics(lab, gt, st, t), fa_hota_naiv(lab, gt, st, t)
    assert set(a) == set(b), "Feld ungeprueft: " + str(set(a) ^ set(b))
    for k in b:
        assert a[k] == pytest.approx(b[k]), f"Abweichung in {k}"


def test_fa_hota_gleich_hota_ohne_zerreissung():
    """Kernaussage des Papiers: FA-HOTA faellt mit HOTA zusammen, wenn
    keine Fragmentierung auftritt (F = A)."""
    from eventpiv.bewerten.evaluate import fa_hota_metrics, tracking_metrics

    lab = np.array([0, 0, 0, 1, 1, 1])
    gt = np.array([7, 7, 7, 9, 9, 9])
    st = np.zeros(6, np.uint8)
    assert fa_hota_metrics(lab, gt, st, np.arange(6))["fa_hota"] == \
        pytest.approx(tracking_metrics(lab, gt, st)["hota"])


def test_fa_hota_unter_hota_bei_zerreissung():
    """Fig. 5 des Papiers: gleiche Assoziation, verschiedene
    Fragmentierung. Zwei Cluster im Wechsel gegen zwei Bloecke -- HOTA
    identisch, FA-HOTA nicht."""
    from eventpiv.bewerten.evaluate import fa_hota_metrics, tracking_metrics

    gt = np.ones(8, np.int64)
    st = np.zeros(8, np.uint8)
    t = np.arange(8)
    bloecke = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    wechsel = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    assert tracking_metrics(bloecke, gt, st)["hota"] == \
        pytest.approx(tracking_metrics(wechsel, gt, st)["hota"])
    assert (fa_hota_metrics(wechsel, gt, st, t)["fa_hota"] < fa_hota_metrics(bloecke, gt, st, t)["fa_hota"])


def test_fa_hota_leer():
    from eventpiv.bewerten.evaluate import fa_hota_metrics

    assert fa_hota_metrics(np.full(3, -1), np.full(3, -1),
                           np.full(3, STATUS_NOISE, np.uint8),
                           np.arange(3)) == {}


@pytest.mark.parametrize("seed", range(6))
def test_fa_hota_gerastert_gegen_naive_referenz(seed):
    """Die Rasterung KOMPONIERT: fa_hota_metrics(..., delta=d) ist
    dasselbe wie die naive Fassung auf dem gerasterten Strom."""
    from eventpiv.bewerten.evaluate import _raster_strom, fa_hota_metrics

    rng = np.random.default_rng(seed)
    n = 80
    lab = rng.integers(-1, 4, n)
    gt = rng.integers(1, 4, n)
    st = rng.choice([STATUS_CLEAN, STATUS_AMBIG, STATUS_NOISE], n,
                    p=[0.7, 0.15, 0.15]).astype(np.uint8)
    gt = np.where(st == STATUS_NOISE, -1, gt)
    t = np.sort(rng.integers(0, 40, n))

    a = fa_hota_metrics(lab, gt, st, t, delta=5)
    b = fa_hota_naiv(*_raster_strom(lab, gt, st, t, 5))
    for k in b:
        assert a[k] == pytest.approx(b[k]), f"Abweichung in {k}"


def test_rasterung_rettet_fa_hota_vor_der_fn_dichte():
    """DER Grund fuer die Rasterung, als Test. Eine ununterbrochene
    Zuordnung mit einem verstreuten unzugeordneten Event je zehn: auf
    Eventebene zerfaellt sie in lauter Kurzfragmente, gerastert bleibt
    sie ein Stueck."""
    from eventpiv.bewerten.evaluate import fa_hota_metrics

    n = 200
    lab = np.zeros(n, np.int64)
    lab[::10] = -1                      # 10 % isolierte FN
    gt = np.ones(n, np.int64)
    st = np.zeros(n, np.uint8)
    t = np.arange(n)

    roh = fa_hota_metrics(lab, gt, st, t)
    fein = fa_hota_metrics(lab, gt, st, t, delta=20)
    assert roh["hota_n_fragmente"] > 15
    assert fein["hota_n_fragmente"] == 1
    # Gerastert ist die Zuordnung perfekt, weil jede Einheit einen
    # zugeordneten Anteil hat; roh zerfaellt sie an den Einzel-FN.
    assert fein["fa_hota"] == pytest.approx(fein["hota_einheiten"])
    assert roh["fa_hota"] < roh["hota_einheiten"]
    assert fein["fa_hota"] > roh["fa_hota"] + 0.3


def test_fa_hota_hoechstens_hota_auf_demselben_strom():
    """Die Invariante des Papiers: F <= A, also fa_hota <= hota_einheiten,
    mit Gleichheit genau dann, wenn nichts zerrissen ist."""
    from eventpiv.bewerten.evaluate import fa_hota_metrics

    gt = np.ones(8, np.int64)
    st = np.zeros(8, np.uint8)
    t = np.arange(8)
    ganz = fa_hota_metrics(np.zeros(8, np.int64), gt, st, t)
    assert ganz["fa_hota"] == pytest.approx(ganz["hota_einheiten"])
    riss = fa_hota_metrics(np.array([0, 0, -1, 0, 0, -1, 0, 0]), gt, st, t)
    assert riss["fa_hota"] < riss["hota_einheiten"]


def test_raster_strom_fuehrt_die_rauschseite_mit():
    """FP muessen als Einheiten erhalten bleiben -- sonst faellt DetPr
    aus der Rechnung und ein Cluster voller Rauschen kostet nichts."""
    from eventpiv.bewerten.evaluate import _raster_strom

    lab = np.array([0, 0, 3, 3])
    gt = np.array([1, 1, -1, -1])
    st = np.array([STATUS_CLEAN, STATUS_CLEAN, STATUS_NOISE, STATUS_NOISE],
                  np.uint8)
    lab_r, gt_r, st_r, _ = _raster_strom(lab, gt, st, np.arange(4), 10)
    assert (st_r == STATUS_NOISE).sum() == 1        # (Bin 0, Cluster 3)
    assert lab_r[st_r == STATUS_NOISE][0] == 3
    assert (st_r != STATUS_NOISE).sum() == 1        # (Bin 0, Partikel 1)