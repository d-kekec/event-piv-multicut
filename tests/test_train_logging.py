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

from eventpiv.metrics import flatten
from eventpiv.train import _epochenzeile

ERWARTETE_SCHLUESSEL = {
    "epoch",
    "train/loss",
    "holdout/loss", "holdout/accuracy", "holdout/auc", "holdout/precision",
    "holdout/tpr", "holdout/tnr", "holdout/f1", "holdout/balanced_accuracy",
    "holdout/pred_neg_frac",
    "holdout/cm/tp", "holdout/cm/fp", "holdout/cm/fn", "holdout/cm/tn",
    "diag/sec_epoch",
}


def _beispiel_hold() -> dict:
    """Wie pair_report plus Holdout-Loss es liefert."""
    return {"loss": 0.3201, "accuracy": 0.8245, "precision": 0.926,
            "tpr": 0.859, "tnr": 0.644, "f1": 0.892,
            "balanced_accuracy": 0.751, "pred_neg_frac": 0.222,
            "auc": 0.8647, "tp": 4, "fp": 1, "fn": 2, "tn": 3}


def test_epochenzeile_erzeugt_genau_die_vereinbarten_schluessel():
    zeile = flatten(_epochenzeile(7, 0.3637, _beispiel_hold(), 0.382))
    assert set(zeile) == ERWARTETE_SCHLUESSEL


def test_konfusionsmatrix_liegt_unter_holdout_nicht_daneben():
    """cm/ allein waere mehrdeutig, sobald es eine Trainingsmatrix gibt:
    die oberste Ebene muss sagen, WELCHE Menge gemessen wurde."""
    zeile = flatten(_epochenzeile(1, 0.4, _beispiel_hold(), 0.4))
    assert zeile["holdout/cm/tp"] == 4
    assert "cm/tp" not in zeile


def test_epoch_liegt_flach():
    """define_metric("*", step_metric="epoch") braucht den Schluessel genau
    so -- verschachtelt waere er "x/epoch" und die x-Achse waere kaputt."""
    zeile = flatten(_epochenzeile(7, 0.4, _beispiel_hold(), 0.4))
    assert zeile["epoch"] == 7