#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Die Zusicherungen des Paar-Encoders.

Drei davon sind Eigenschaften, die die KONSTRUKTION garantieren soll --
Permutationsinvarianz, Padding-Invarianz, Wirkung des Segment-Etiketts.
Genau solche Eigenschaften brechen still: das Netz laeuft weiter und
liefert nur leicht falsche Zahlen. Deshalb stehen sie hier.
"""
import numpy as np
import pytest
import torch

from eventpiv.kette.transformer import (PaarTransformer, baue_tokens,
                                        load_checkpoint, save_checkpoint)

ALPHA, R_MAX, N = 30_000.0, 15.0, 8


@pytest.fixture
def netz():
    torch.manual_seed(0)
    return PaarTransformer(d_model=32, n_layers=2, n_heads=4,
                           d_ff=64, dropout=0.0).eval()


def _knotentensor(n_echt=(N, N)):
    """Zwei Knoten mit vorgegebener Zahl echter Events."""
    g = torch.Generator().manual_seed(1)
    ev = torch.zeros(2, N, 3)
    maske = torch.zeros(2, N, dtype=torch.bool)
    for k, ne in enumerate(n_echt):
        ev[k, :ne, 0] = torch.rand(ne, generator=g) * 9_000
        ev[k, :ne, 1] = torch.rand(ne, generator=g) * 10 + 100 * k
        ev[k, :ne, 2] = torch.rand(ne, generator=g) * 10 + 50
        maske[k, :ne] = True
    t_min = torch.tensor([0.0, 12_000.0])
    x_c = ev[:, :, 1].sum(1) / maske.sum(1)
    y_c = ev[:, :, 2].sum(1) / maske.sum(1)
    return ev, maske, t_min, x_c, y_c


def _logit(netz, ev, maske, t_min, x_c, y_c, i=0, j=1):
    tok, tm = baue_tokens(ev, maske, torch.tensor([i]), torch.tensor([j]),
                          t_min, x_c, y_c, alpha=ALPHA, r_max=R_MAX)
    with torch.no_grad():
        return netz(tok, tm).item()


def test_permutationsinvariant_in_der_ziehreihenfolge(netz):
    """DIE zentrale Zusicherung: ein Tracklet ist eine Punktwolke, keine
    Sequenz. Dieselben Events in anderer Reihenfolge muessen denselben
    Logit geben -- sonst haenge das Ergebnis an der Stichprobenziehung."""
    ev, maske, t_min, x_c, y_c = _knotentensor()
    a = _logit(netz, ev, maske, t_min, x_c, y_c)

    perm = torch.randperm(N, generator=torch.Generator().manual_seed(2))
    ev2 = ev.clone()
    ev2[0] = ev[0][perm]
    b = _logit(netz, ev2, maske, t_min, x_c, y_c)
    assert a == pytest.approx(b, abs=1e-5)


def test_padding_aendert_den_logit_nicht(netz):
    """Ein Knoten mit vier echten Events muss denselben Logit geben, egal
    ob auf acht oder auf sechzehn Spalten aufgefuellt. Ohne greifende
    src_key_padding_mask zoegen die Fuell-Token echte Aufmerksamkeit."""
    ev, maske, t_min, x_c, y_c = _knotentensor(n_echt=(4, N))
    a = _logit(netz, ev, maske, t_min, x_c, y_c)

    # Dieselben Daten, doppelt so breit aufgefuellt.
    ev2 = torch.zeros(2, 2 * N, 3)
    m2 = torch.zeros(2, 2 * N, dtype=torch.bool)
    ev2[:, :N], m2[:, :N] = ev, maske
    b = _logit(netz, ev2, m2, t_min, x_c, y_c)
    assert a == pytest.approx(b, abs=1e-5)


def test_fuelltoken_sind_null_und_maske_stimmt():
    ev, maske, t_min, x_c, y_c = _knotentensor(n_echt=(3, N))
    tok, tm = baue_tokens(ev, maske, torch.tensor([0]), torch.tensor([1]),
                          t_min, x_c, y_c, alpha=ALPHA, r_max=R_MAX)
    assert tok.shape == (1, 2 * N, 3) and tm.shape == (1, 2 * N)
    assert int(tm.sum()) == 3 + N
    assert torch.all(tok[0][~tm[0]] == 0)


def test_zeitversatz_steht_in_den_tokens():
    """Die eigentliche Information des Paares. Der zweite Knoten muss um
    (t_min[j] - t_min[i]) / alpha verschoben sein -- ohne das saehen zwei
    Tracklets aus benachbarten Fenstern aus wie zwei aus demselben."""
    ev, maske, t_min, x_c, y_c = _knotentensor()
    tok, _ = baue_tokens(ev, maske, torch.tensor([0]), torch.tensor([1]),
                         t_min, x_c, y_c, alpha=ALPHA, r_max=R_MAX)
    erwartet = (ev[1, :, 0] + (t_min[1] - t_min[0])) / ALPHA
    torch.testing.assert_close(tok[0, N:, 0], erwartet)
    torch.testing.assert_close(tok[0, :N, 0], ev[0, :, 0] / ALPHA)


def test_ursprung_liegt_auf_dem_frueheren_knoten():
    """Nicht auf dem Bildursprung: absolute Koordinaten sind der Fehler,
    an dem raw6 in Stufe 1 scheiterte."""
    ev, maske, t_min, x_c, y_c = _knotentensor()
    tok, _ = baue_tokens(ev, maske, torch.tensor([0]), torch.tensor([1]),
                         t_min, x_c, y_c, alpha=ALPHA, r_max=R_MAX)
    # Zentroid des ersten Knotens liegt danach im Ursprung.
    assert tok[0, :N, 1].mean().item() == pytest.approx(0.0, abs=1e-5)
    assert tok[0, :N, 2].mean().item() == pytest.approx(0.0, abs=1e-5)


def test_orientierung_ist_tragend(netz):
    """Das Netz ist NICHT symmetrisch unter dem Tausch der Knoten --
    Segment-Etikett und Ursprung haengen daran, wer Nummer 1 ist.
    c_ij = c_ji gilt, WEIL kandidatenpaare jede Kante genau einmal mit dem
    frueheren t_min als Nummer 1 liefert. Die kanonische Ordnung ERSETZT
    die Symmetrie; dieser Test haelt fest, dass sie tragend ist."""
    ev, maske, t_min, x_c, y_c = _knotentensor()
    a = _logit(netz, ev, maske, t_min, x_c, y_c, i=0, j=1)
    b = _logit(netz, ev, maske, t_min, x_c, y_c, i=1, j=0)
    assert a != pytest.approx(b, abs=1e-4)


def test_segment_etikett_wirkt(netz):
    """Ohne Etikett waere die Eingabe eine ununterscheidbare Wolke aus 2n
    Punkten. Setzt man beide Etiketten gleich, muss sich der Logit
    aendern -- sonst wird das Etikett gar nicht gelesen."""
    ev, maske, t_min, x_c, y_c = _knotentensor()
    a = _logit(netz, ev, maske, t_min, x_c, y_c)
    with torch.no_grad():
        netz.segment[1] = netz.segment[0]
    b = _logit(netz, ev, maske, t_min, x_c, y_c)
    assert a != pytest.approx(b, abs=1e-4)


@pytest.mark.parametrize("pooling", ["cls", "mean"])
def test_checkpoint_rundreise(tmp_path, pooling):
    """mu/sig heissen hier alpha/r_max: das Netz hat auf DIESER Skala
    gelernt, und ohne sie ist der Checkpoint im Einsatz unbrauchbar."""
    from eventpiv.kern.runconfig import Kandidaten, Transformer

    tf = Transformer(d_model=32, n_heads=4, d_ff=64, pooling=pooling)
    kd = Kandidaten()
    torch.manual_seed(0)
    netz = PaarTransformer(d_model=tf.d_model, n_layers=tf.n_layers,
                           n_heads=tf.n_heads, d_ff=tf.d_ff,
                           dropout=tf.dropout, pooling=tf.pooling).eval()

    p = save_checkpoint(tmp_path / "model.pt", netz, tf, kd, notiz="test")
    geladen, norm, ckpt = load_checkpoint(p)
    assert norm == {"alpha": float(kd.alpha), "r_max": float(kd.r_max),
                    "n_events": int(kd.n_events)}
    assert ckpt["meta"]["notiz"] == "test"

    ev, maske, t_min, x_c, y_c = _knotentensor()
    assert _logit(netz, ev, maske, t_min, x_c, y_c) == pytest.approx(
        _logit(geladen, ev, maske, t_min, x_c, y_c), abs=1e-6)


def test_stapel_ist_zeilenweise_unabhaengig(netz):
    """Ein Paar im Stapel darf keinen Einfluss auf ein anderes haben --
    ein vertauschtes Achsenpaar in baue_tokens faellt sonst nicht auf."""
    ev, maske, t_min, x_c, y_c = _knotentensor()
    tok, tm = baue_tokens(ev, maske, torch.tensor([0, 1]),
                          torch.tensor([1, 0]), t_min, x_c, y_c,
                          alpha=ALPHA, r_max=R_MAX)
    with torch.no_grad():
        stapel = netz(tok, tm)
    assert stapel[0].item() == pytest.approx(
        _logit(netz, ev, maske, t_min, x_c, y_c, 0, 1), abs=1e-5)
    assert stapel[1].item() == pytest.approx(
        _logit(netz, ev, maske, t_min, x_c, y_c, 1, 0), abs=1e-5)