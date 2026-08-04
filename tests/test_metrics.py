#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_metrics.py -- Invarianten der Paar-Kennzahlen.

Ohne Datensatz, ohne Torch, ohne Hydra: von Hand nachrechenbare Beispiele.
Die Probe darauf, dass metrics.py reine Messung ist und kein Framework
kennt.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from eventpiv.metrics import (class_balance, confusion, flatten, pair_report,
                              rates, roc_auc)


# ------------------------------------------------------------------ flatten

def test_flatten_von_hand():
    assert flatten({"epoch": 1, "holdout": {"auc": 0.9, "cm": {"tp": 3}}}) == {
        "epoch": 1, "holdout/auc": 0.9, "holdout/cm/tp": 3}


def test_flatten_laesst_flaches_dict_unveraendert():
    d = {"a": 1, "b": 2.0, "c": None}
    assert flatten(d) == d


def test_flatten_leerer_zweig_erzeugt_keinen_schluessel():
    """Dokumentiert das Verhalten: {"a": {}} liefert nichts, nicht "a"."""
    assert flatten({"a": {}, "b": 1}) == {"b": 1}


def test_flatten_trenner_waehlbar():
    """W&B nimmt den Punkt, wir wollen den Schraegstrich -- beides moeglich."""
    assert flatten({"a": {"b": 1}}, sep=".") == {"a.b": 1}


def test_flatten_listen_bleiben_werte():
    """Nur dicts werden aufgeloest. val_runs ist ein Wert, kein Zweig."""
    assert flatten({"data": {"val_runs": [30, 43]}}) == {
        "data/val_runs": [30, 43]}

# ---------------------------------------------------------------- confusion

def test_confusion_von_hand():
    """Sechs Paare, Matrix von Hand ausgezaehlt.

    Logit   +2   +1   -1   -3   +5   -2
    Label    1    0    1    0    1    0
             TP   FP   FN   TN   TP   TN
    """
    y = [1, 0, 1, 0, 1, 0]
    s = [2.0, 1.0, -1.0, -3.0, 5.0, -2.0]
    assert confusion(y, s) == {"tp": 2, "fp": 1, "fn": 1, "tn": 2}


def test_schwelle_null_zaehlt_als_ja():
    """Logit exakt 0 gilt als JA -- dieselbe Seite wie p >= 0.5."""
    assert confusion([1, 0], [0.0, 0.0]) == {"tp": 1, "fp": 1, "fn": 0,
                                             "tn": 0}


def test_schwelle_verschiebbar():
    """Ein Kostenversatz verschiebt die Matrix -- der vorgesehene Regler fuer
    Stufe 1 des Tracklet-Ansatzes: negativer Versatz kauft Reinheit mit
    Fragmentierung, also FN gegen FP."""
    y, s = [1, 1, 0, 0], [3.0, 1.0, 2.0, -1.0]
    assert confusion(y, s, threshold=0.0)["fp"] == 1
    assert confusion(y, s, threshold=2.5)["fp"] == 0    # FP weggeschnitten
    assert confusion(y, s, threshold=2.5)["fn"] == 1    # zum Preis eines FN


def test_confusion_summe_ist_n():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = rng.normal(size=500)
    assert sum(confusion(y, s).values()) == 500


def test_confusion_lehnt_fremde_labels_ab():
    with pytest.raises(ValueError, match="nur 0 und 1"):
        confusion([0, 1, 2], [0.0, 0.0, 0.0])


def test_confusion_lehnt_laengenfehler_ab():
    with pytest.raises(ValueError, match="Laengen verschieden"):
        confusion([0, 1], [0.0])


def test_float_labels_erlaubt():
    """Der Paar-Datensatz haelt label als float32."""
    y = np.array([1.0, 0.0], dtype=np.float32)
    assert confusion(y, [1.0, -1.0]) == {"tp": 1, "fp": 0, "fn": 0, "tn": 1}


# -------------------------------------------------------------------- rates

def test_rates_von_hand():
    cm = {"tp": 6, "fp": 2, "fn": 3, "tn": 9}          # n = 20
    r = rates(cm)
    assert r["accuracy"] == pytest.approx(15 / 20)
    assert r["precision"] == pytest.approx(6 / 8)
    assert r["tpr"] == pytest.approx(6 / 9)
    assert r["tnr"] == pytest.approx(9 / 11)
    assert r["f1"] == pytest.approx(12 / (12 + 2 + 3))
    assert r["balanced_accuracy"] == pytest.approx(0.5 * (6 / 9 + 9 / 11))
    assert r["pred_neg_frac"] == pytest.approx(12 / 20)


def test_rates_fehlende_klasse_gibt_nan_statt_absturz():
    """Kein JA vorhanden -> tpr undefiniert. NaN ist im Log sichtbar,
    ein Abbruch kostet den Trainingslauf."""
    r = rates({"tp": 0, "fp": 0, "fn": 0, "tn": 7})
    assert math.isnan(r["tpr"])
    assert math.isnan(r["precision"])
    assert r["tnr"] == pytest.approx(1.0)
    assert r["accuracy"] == pytest.approx(1.0)


def test_immer_ja_entlarvt_accuracy():
    """Bei 6.3:1 erreicht 'immer JA' schon 0.863 Accuracy -- genau der Grund,
    warum accuracy allein hier nichts sagt und balanced_accuracy mitlaeuft."""
    cm = {"tp": 63, "fp": 10, "fn": 0, "tn": 0}
    r = rates(cm)
    assert r["accuracy"] == pytest.approx(0.863, abs=1e-3)
    assert r["balanced_accuracy"] == pytest.approx(0.5)


# ------------------------------------------------------------------ roc_auc

def test_auc_perfekte_trennung():
    assert roc_auc([1, 1, 0, 0], [3.0, 2.0, -1.0, -2.0]) == pytest.approx(1.0)


def test_auc_perfekt_falsch_herum():
    assert roc_auc([1, 1, 0, 0], [-3.0, -2.0, 1.0, 2.0]) == pytest.approx(0.0)


def test_auc_von_hand():
    """Ein JA (1.0) gegen zwei NEIN (0.0, 2.0): das JA schlaegt genau eines
    der beiden -> AUC = 1/2."""
    assert roc_auc([1, 0, 0], [1.0, 0.0, 2.0]) == pytest.approx(0.5)


def test_auc_nan_wenn_eine_klasse_fehlt():
    assert math.isnan(roc_auc([1, 1, 1], [1.0, 2.0, 3.0]))
    assert math.isnan(roc_auc([0, 0, 0], [1.0, 2.0, 3.0]))


def test_auc_invariant_unter_sigmoid():
    """AUC ist rangbasiert, sigmoid ist streng monoton -> identisch.

    DAS ist die Rechtfertigung dafuer, direkt am Logit zu schwellen statt an
    p: die Kennzahl aendert sich dadurch nicht.
    """
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 400)
    logit = rng.normal(scale=3.0, size=400)
    p = 1.0 / (1.0 + np.exp(-logit))
    assert roc_auc(y, logit) == pytest.approx(roc_auc(y, p), abs=1e-12)


def test_auc_gleichstaende_kleiner_extremfall():
    """Haelt den IST-Zustand fest, KEIN Soll-Verhalten.

    Ein JA und ein NEIN mit demselben Score: korrekt waere 0.5 ("nicht
    unterscheidbar"), die Rangvergabe ohne mittlere Raenge liefert 0.0.
    Auf realistischer Groesse mittelt sich der Effekt weg -- siehe den
    naechsten Test und den Docstring von roc_auc.
    """
    assert roc_auc([1, 0], [0.5, 0.5]) == pytest.approx(0.0)


def test_auc_gleichstaende_mitteln_sich_bei_grossen_mengen_weg():
    """Der Grund, warum der Extremfall oben folgenlos bleibt.

    Stark quantisierte Scores (wie (dt, dr) sie erzeugen), aber viele Paare:
    np.argsort ist per Default instabil, die Gleichstaende werden durchmischt
    und das Ergebnis liegt praktisch auf dem korrekten Wert mit mittleren
    Raengen. Dieser Test ist die Absicherung des Satzes "die dokumentierten
    0.792 sind nicht betroffen".
    """
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 50_000)
    s = np.round((rng.normal(size=50_000) + 0.8 * y) * 8) / 8
    assert len(np.unique(s)) < 100                 # massive Gleichstaende
    assert roc_auc(y, s) == pytest.approx(_auc_mittlere_raenge(y, s),
                                          abs=2e-3)


def _auc_mittlere_raenge(y, s):
    """Referenz nach Definition: Mann-Whitney mit mittleren Raengen.

    Nur fuer den Test -- bewusst NICHT in metrics.py. Die produktive Funktion
    auf mittlere Raenge umzustellen waere eine inhaltliche Aenderung und
    muesste zusammen mit der kind=-Frage entschieden werden.
    """
    y, s = np.asarray(y), np.asarray(s)
    o = np.argsort(s, kind="stable")
    r = np.empty(len(s), float)
    r[o] = np.arange(1, len(s) + 1)
    ss, i = s[o], 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


# ------------------------------------------------------------ class_balance

def test_class_balance_von_hand():
    b = class_balance([1, 1, 1, 0])
    assert b == {"n": 4, "n_ja": 3, "n_nein": 1,
                 "ja_frac": pytest.approx(0.75),
                 "ratio_ja_nein": pytest.approx(3.0)}


def test_class_balance_ohne_nein_gibt_nan():
    """Saubere Runs liefern per Konstruktion NULL NEIN-Paare -- der Quotient
    ist dort undefiniert, nicht unendlich, und darf den Lauf nicht
    abbrechen."""
    b = class_balance([1, 1, 1])
    assert b["n_nein"] == 0
    assert math.isnan(b["ratio_ja_nein"])


# ------------------------------------------------------------- pair_report

def test_pair_report_ist_flach_und_vollstaendig():
    r = pair_report([1, 0, 1, 0], [2.0, -1.0, -2.0, 1.0])
    for k in ("tp", "fp", "fn", "tn", "accuracy", "precision", "tpr", "tnr",
              "f1", "balanced_accuracy", "pred_neg_frac", "auc"):
        assert k in r, f"{k} fehlt"
    assert all(not isinstance(v, dict) for v in r.values())


def test_pair_report_stimmt_mit_einzelaufrufen_ueberein():
    y, s = [1, 1, 0, 0, 1], [1.0, -1.0, -2.0, 0.5, 3.0]
    r = pair_report(y, s)
    cm = confusion(y, s)
    assert {k: r[k] for k in cm} == cm
    assert r["auc"] == pytest.approx(roc_auc(y, s))