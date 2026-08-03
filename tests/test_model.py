#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Invariantentests fuer model.build_features.

Der wichtigste Test ist test_numpy_und_torch_identisch: er belegt, dass
Training (NumPy) und Einsatz (Torch/GPU) DIESELBEN Features sehen. Frueher
waren das zwei getrennte Implementierungen -- genau die Art Duplikat, die
still auseinanderlaeuft.
"""
import numpy as np
import pytest
import torch

from eventpiv.model import (FEATURE_MODES, EdgeMLP, build_features,
                            feature_names, is_symmetric, n_features)


@pytest.fixture
def deltas():
    rng = np.random.default_rng(7)
    n = 50
    dt = rng.uniform(0, 10_000, n).astype(np.float32)      # >= 0, Kontrakt
    dx = rng.uniform(-7, 7, n).astype(np.float32)
    dy = rng.uniform(-7, 7, n).astype(np.float32)
    ti = rng.uniform(0, 8e6, n).astype(np.float32)
    xi = rng.uniform(0, 1280, n).astype(np.float32)
    yi = rng.uniform(0, 720, n).astype(np.float32)
    return dt, dx, dy, ti, xi, yi


@pytest.mark.parametrize("mode", FEATURE_MODES)
def test_form_passt_zu_n_features(deltas, mode):
    dt, dx, dy, ti, xi, yi = deltas
    f = build_features(dt, dx, dy, mode=mode, ti=ti, xi=xi, yi=yi)
    assert f.shape == (len(dt), n_features(mode))
    assert len(feature_names(mode)) == n_features(mode)


@pytest.mark.parametrize("mode", FEATURE_MODES)
def test_numpy_und_torch_identisch(deltas, mode):
    """Training rechnet mit NumPy, solve mit Torch auf der GPU -- die
    Features muessen bis auf Rundung identisch sein, sonst sieht das Netz
    im Einsatz etwas anderes als im Training.

    Toleranz statt Bitgleichheit: in dr entsteht ein Unterschied von 1 ULP
    (float32-eps = 1.19e-7), weil Torch fuer dx*dx+dy*dy FMA nutzt (eine
    Rundung) und NumPy getrennte Mul/Add (zwei Rundungen). Beide Ergebnisse
    sind korrekt gerundet. absxy und raw6 sind exakt gleich (nur abs bzw. +).
    Auswirkung auf die Kantenkosten: ~1e-6 im Logit -- irrelevant, ausser
    exakt an der Nullstelle, wo die Kante ohnehin maximal unsicher ist.
    """
    dt, dx, dy, ti, xi, yi = deltas
    f_np = build_features(dt, dx, dy, mode=mode, ti=ti, xi=xi, yi=yi, xp=np)
    f_t = build_features(*[torch.tensor(a) for a in (dt, dx, dy)], mode=mode,
                         ti=torch.tensor(ti), xi=torch.tensor(xi),
                         yi=torch.tensor(yi), xp=torch)
    np.testing.assert_allclose(f_np, f_t.numpy(), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("mode", ["dr", "absxy"])
def test_symmetrie_unter_vertauschung(deltas, mode):
    """c_ij == c_ji: Vertauschen von i und j spiegelt dx,dy -- die Features
    duerfen sich nicht aendern (dt bleibt >= 0 per Konstruktion)."""
    dt, dx, dy, *_ = deltas
    a = build_features(dt, dx, dy, mode=mode)
    b = build_features(dt, -dx, -dy, mode=mode)
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)
    assert is_symmetric(mode)


def test_raw6_ist_nicht_symmetrisch(deltas):
    dt, dx, dy, ti, xi, yi = deltas
    assert not is_symmetric("raw6")
    f = build_features(dt, dx, dy, mode="raw6", ti=ti, xi=xi, yi=yi)
    # Spalten 3..5 sind das SPAETERE Event
    np.testing.assert_allclose(f[:, 3], ti + dt, rtol=1e-5)
    np.testing.assert_allclose(f[:, 4], xi + dx, rtol=1e-5)


def test_dr_ist_euklidisch(deltas):
    dt, dx, dy, *_ = deltas
    f = build_features(dt, dx, dy, mode="dr")
    np.testing.assert_allclose(f[:, 1], np.hypot(dx, dy), rtol=1e-5)


def test_raw6_ohne_absolutkoordinaten_bricht_ab(deltas):
    dt, dx, dy, *_ = deltas
    with pytest.raises(ValueError):
        build_features(dt, dx, dy, mode="raw6")


def test_unbekannter_modus_bricht_ab(deltas):
    dt, dx, dy, *_ = deltas
    with pytest.raises(ValueError):
        build_features(dt, dx, dy, mode="quatsch")


def test_netz_gibt_logit_je_zeile(deltas):
    dt, dx, dy, *_ = deltas
    f = torch.tensor(build_features(dt, dx, dy, mode="dr"))
    out = EdgeMLP(2, 8)(f)
    assert out.shape == (len(dt), 1)
    assert sum(p.numel() for p in EdgeMLP(2, 8).parameters()) == 33


def test_checkpoint_rundreise(tmp_path, deltas):
    """save -> load muss Netz UND Standardisierung unveraendert zurueckgeben.

    mu/sig sind der stille Fehlerfall: geht die Standardisierung verloren
    oder wird sie vertauscht, sieht das Netz im Einsatz andere Zahlen als
    im Training -- ohne Absturz, nur mit schlechteren Kanten.
    """
    from eventpiv.model import load_checkpoint, save_checkpoint

    dt, dx, dy, *_ = deltas
    f = torch.tensor(build_features(dt, dx, dy, mode="dr"))
    mu = np.array([5000.0, 2.0], dtype=np.float32)
    sig = np.array([3000.0, 1.5], dtype=np.float32)

    modell = EdgeMLP(2, 8)
    modell.eval()
    with torch.no_grad():
        vorher = modell((f - torch.tensor(mu)) / torch.tensor(sig))

    pfad = save_checkpoint(tmp_path / "tief" / "model.pt", modell,
                           mu, sig, "dr", 8, quelle="test")
    assert pfad.exists()          # legt das Verzeichnis selbst an

    geladen, mu_l, sig_l, mode_l, ckpt = load_checkpoint(pfad)
    assert mode_l == "dr" and ckpt["n_in"] == 2 and ckpt["hidden"] == 8
    assert ckpt["meta"]["quelle"] == "test"
    np.testing.assert_array_equal(mu_l.numpy(), mu)
    np.testing.assert_array_equal(sig_l.numpy(), sig)

    with torch.no_grad():
        nachher = geladen((f - mu_l) / sig_l)
    torch.testing.assert_close(vorher, nachher)