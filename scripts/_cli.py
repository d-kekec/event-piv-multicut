#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bruecke Hydra -> Dataclass. EINZIGE Stelle in scripts/, die OmegaConf kennt.

Warum nicht einfach OmegaConf.to_object: Hydra erlaubt mit '+' das Anhaengen
beliebiger Schluessel auf oberster Ebene (dafuer ist die Syntax da), und
to_object wirft sie danach still weg. '+alpha=20000' -- flach statt gruppiert --
wuerde damit weder auffallen noch wirken. Gemessen 27.07.2026.
"""
from dataclasses import asdict

from omegaconf import DictConfig, OmegaConf


def _flat(d: dict, prefix: str = "") -> set[str]:
    out: set[str] = set()
    for k, v in d.items():
        key = f"{prefix}{k}"
        out |= _flat(v, key + ".") if isinstance(v, dict) else {key}
    return out


def to_cfg(cfg: DictConfig):
    """DictConfig -> Dataclass, mit hartem Abbruch bei unbekannten Schluesseln."""
    obj = OmegaConf.to_object(cfg)
    fremd = _flat(OmegaConf.to_container(cfg, resolve=True)) - _flat(asdict(obj))
    if fremd:
        raise SystemExit(
            f"Unbekannte Konfigurationsschluessel: {sorted(fremd)}\n"
            "Mit '+' angehaengt? Parameter gehoeren in ihre Gruppe "
            "(neighborhood.alpha=..., nicht alpha=...).")
    
    # Abgeleitete Defaults (win_dt aus alpha/r_max) einsetzen, BEVOR
    # irgendein Slug berechnet wird -- sonst bekaemen win_dt=None und
    # win_dt=110763 verschiedene Identitaeten fuer denselben Datensatz.
    if hasattr(obj, "resolve_derived"):
        obj.resolve_derived()
        
    return obj

def konfusionstabelle(cm: dict):
    """Die 2x2-Matrix als W&B-Tabelle.

    Aus den VIER ZAHLEN gebaut, nicht ueber wandb.plot.confusion_matrix --
    das verlangt die Vorhersage je Datenpunkt und waere bei Millionen Paaren
    absurd.

    Geloggt wird sie ueber run.log, nicht ueber run.summary: nur der
    Log-Eintrag traegt den Deskriptor (_type, ncols, nrows, artifact_path),
    aus dem die Oberflaeche ein Panel baut (geprueft 31.07.2026).
    """
    import wandb
    t = wandb.Table(columns=["", "Predicted JA", "Predicted NEIN"])
    t.add_data("GT JA", cm["tp"], cm["fn"])
    t.add_data("GT NEIN", cm["fp"], cm["tn"])
    return t