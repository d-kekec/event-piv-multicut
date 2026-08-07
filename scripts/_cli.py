#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bruecke Hydra -> Dataclass. EINZIGE Stelle in scripts/, die OmegaConf kennt.

Warum nicht einfach OmegaConf.to_object: Hydra erlaubt mit '+' das Anhaengen
beliebiger Schluessel, und to_object wirft sie danach still weg.
'+alpha=20000' -- flach statt gruppiert -- wuerde damit weder auffallen noch
wirken.
"""
from dataclasses import asdict
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf


def _flat(d: dict, prefix: str = "") -> set[str]:
    out: set[str] = set()
    for k, v in d.items():
        key = f"{prefix}{k}"
        out |= _flat(v, key + ".") if isinstance(v, dict) else {key}
    return out


def to_cfg(cfg: DictConfig) -> Any:
    """DictConfig -> Dataclass, mit hartem Abbruch bei unbekannten Schluesseln.

    Rueckgabetyp Any, keine Nachlaessigkeit: welche Dataclass herauskommt,
    entscheidet das im ConfigStore registrierte Schema, also der Aufrufer --
    statisch nicht bestimmbar. Die Zusicherung steht deshalb an der
    Aufrufstelle ("cfg: TrainCfg = to_cfg(...)").
    """
    obj: Any = OmegaConf.to_object(cfg)
    # cast, weil to_container statisch eine Union liefert (ein Config
    # koennte auch Liste oder Skalar sein). Dass hier immer ein DictConfig
    # ankommt, sichert das Framework zu, nicht das Typsystem.
    roh = cast(dict, OmegaConf.to_container(cfg, resolve=True))
    fremd = _flat(roh) - _flat(asdict(obj))
    if fremd:
        raise SystemExit(
            f"Unbekannte Konfigurationsschluessel: {sorted(fremd)}\n"
            "Mit '+' angehaengt? Parameter gehoeren in ihre Gruppe "
            "(neighborhood.alpha=..., nicht alpha=...).")
    return obj


def konfusionstabelle(cm: dict):
    """Die 2x2-Matrix als W&B-Tabelle.

    Aus den VIER ZAHLEN gebaut, nicht ueber wandb.plot.confusion_matrix --
    das verlangt die Vorhersage je Datenpunkt und waere bei Millionen Paaren
    absurd.

    Geloggt wird sie ueber run.log, nicht ueber run.summary: nur der
    Log-Eintrag traegt den Deskriptor (_type, ncols, nrows, artifact_path),
    aus dem die Oberflaeche ein Panel baut. GEMESSEN an wandb 0.28.1.
    """
    import wandb
    t = wandb.Table(columns=["", "Predicted JA", "Predicted NEIN"])
    t.add_data("GT JA", cm["tp"], cm["fn"])
    t.add_data("GT NEIN", cm["fp"], cm["tn"])
    return t