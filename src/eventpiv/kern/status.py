#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
status.py -- Qualitaetsstatus eines Events, das gemeinsame Vokabular der
Label-Datei.

    STATUS_CLEAN = 0  SAUBER      eindeutig einem Partikel zugeordnet
    STATUS_AMBIG = 1  MEHRDEUTIG  fast gleich weit zu zwei Boxen
    STATUS_NOISE = 2  RAUSCH      keiner Box nah genug -- track_id = -1

Vergeben werden sie in labels.py [1], und WIE, steht dort. Sie wohnen
trotzdem nicht dort: evaluate.py liest sie, und eine Bewertung, die eine
Stufe importiert, ist nicht mehr unabhaengig von ihr. Die Codes beschreiben
ohnehin das DATEIFORMAT der Label-Datei und nicht den Schritt, der sie
hineinschreibt.
"""
STATUS_CLEAN = 0
STATUS_AMBIG = 1
STATUS_NOISE = 2