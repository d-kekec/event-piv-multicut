#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests fuer die datenunabhaengigen Teile von pipeline.py.

Die Orchestrierung selbst (Datensatz -> Training -> Bewertung) braucht den
Datensatz und wird deshalb NICHT getestet -- wohl aber die reinen
Hilfsfunktionen, insbesondere der Fall "kein Run loesbar", der real auftritt
(VRAM-Abstuerze, BA_NN_Kantenkosten.md §3.8).
"""
import math

from eventpiv.report import summarize


def info(purity, frag, frag_max=2):
    return {"purity_mean": purity, "frag_mean": frag, "frag_max": frag_max}


def test_summarize_mittelt_nur_loesbare():
    z = summarize("X", [info(1.0, 1.5), None, info(0.9, 2.5)], verbose=False)
    assert z["n_ok"] == 2 and z["n_total"] == 3
    assert z["purity"] == 0.95 and z["frag"] == 2.0


def test_summarize_ohne_loesbare_runs():
    """Alle abgestuerzt -- darf nicht crashen, sondern NaN melden."""
    z = summarize("X", [None, None], verbose=False)
    assert z["n_ok"] == 0 and z["n_total"] == 2
    assert math.isnan(z["purity"]) and math.isnan(z["frag"])
    assert z["frag_max"] is None


