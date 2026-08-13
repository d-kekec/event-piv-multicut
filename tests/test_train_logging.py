#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Haelt die Namenskonvention des Loggings fest.

Die W&B-Schluessel entstehen aus der Verschachtelung in _epochenzeile plus
metrics.flatten -- an keiner Stelle im Code steht eine Liste der Feldnamen.
DIESER TEST IST DIE LISTE. Wer die Verschachtelung aendert, aendert sichtbar
auch ihn, und alte Laeufe in W&B bleiben vergleichbar oder der Bruch faellt
beim Testlauf auf statt in sechs Wochen im Browser.
"""
from __future__ import annotations

from eventpiv.bewerten.metrics import flatten
from eventpiv.train import _epochenzeile

_METRIKEN = ("loss", "accuracy", "auc", "precision", "tpr", "tnr", "f1",
             "balanced_accuracy", "pred_neg_frac")
_CM = ("tp", "fp", "fn", "tn")

ERWARTETE_SCHLUESSEL = (
    {"epoch", "diag/sec_epoch", "train/loss_running"}
    | {f"train/{m}" for m in _METRIKEN} | {f"train/cm/{c}" for c in _CM}
    | {f"holdout/{m}" for m in _METRIKEN} | {f"holdout/cm/{c}" for c in _CM}
)


def _beispiel(tp=4) -> dict:
    """Wie pair_report plus Holdout-Loss es liefert."""
    return {"loss": 0.3201, "accuracy": 0.8245, "precision": 0.926,
            "tpr": 0.859, "tnr": 0.644, "f1": 0.892,
            "balanced_accuracy": 0.751, "pred_neg_frac": 0.222,
            "auc": 0.8647, "tp": tp, "fp": 1, "fn": 2, "tn": 3}


def test_epochenzeile_erzeugt_genau_die_vereinbarten_schluessel():
    zeile = flatten(_epochenzeile(7, 0.3637, _beispiel(), _beispiel(), 0.382))
    assert set(zeile) == ERWARTETE_SCHLUESSEL


def test_konfusionsmatrix_liegt_unter_holdout_nicht_daneben():
    """cm/ allein waere mehrdeutig, sobald es eine Trainingsmatrix gibt:
    die oberste Ebene muss sagen, WELCHE Menge gemessen wurde."""
    zeile = flatten(_epochenzeile(1, 0.4, _beispiel(), _beispiel(), 0.4))
    assert zeile["holdout/cm/tp"] == 4
    assert "cm/tp" not in zeile


def test_epoch_liegt_flach():
    """define_metric("*", step_metric="epoch") braucht den Schluessel genau
    so -- verschachtelt waere er "x/epoch" und die x-Achse waere kaputt."""
    zeile = flatten(_epochenzeile(7, 0.4, _beispiel(), _beispiel(), 0.4))
    assert zeile["epoch"] == 7

def test_beide_mengen_getrennt():
    """Ohne die oberste Ebene waere "tp" mehrdeutig, seit es zwei Matrizen
    gibt. Das ist der Grund fuer die Verschachtelung."""
    zeile = flatten(_epochenzeile(1, 0.4, _beispiel(tp=1), _beispiel(tp=9), 0.4))
    assert zeile["train/cm/tp"] == 1
    assert zeile["holdout/cm/tp"] == 9
    assert "cm/tp" not in zeile


def test_laufender_verlust_nur_auf_der_trainingsseite():
    """holdout/loss_running gibt es nicht -- auf dem Holdout wird nicht
    schrittweise gerechnet."""
    zeile = flatten(_epochenzeile(1, 0.4, _beispiel(), _beispiel(), 0.4))
    assert "train/loss_running" in zeile
    assert "holdout/loss_running" not in zeile