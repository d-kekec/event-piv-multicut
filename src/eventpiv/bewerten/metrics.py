#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics.py -- Kennzahlen der PAAR-Ebene (Stufe [3]).

Reines NumPy: keine Torch-, keine Hydra-Abhaengigkeit. Damit ohne Datensatz
und ohne Framework testbar; der Aufrufer wandelt Tensoren vorher in Arrays um.

Bewusst NUR die Paar-Ebene. Reinheit und Fragmentierung sind Cluster-Ebene
und bleiben in evaluate.py -- zwei Zustaendigkeiten, zwei Module.

WARUM DIE SCHWELLE 0 UND KEINE ANDERE
-------------------------------------
Das Netz gibt einen LOGIT aus, und dieser Logit geht in solve.py UNVERAENDERT
als Kantenkosten an RAMA. RAMA liest das Vorzeichen: positiv = anziehend,
negativ = abstossend. 
Die Schwelle 0 ist deshalb nicht ein Arbeitspunkt unter vielen, sondern GENAU
der, den der Solver sieht. Wegen p = sigmoid(logit) ist "logit >= 0"
gleichbedeutend mit "p >= 0.5"; wir schwellen direkt am Logit, weil das den
Bezug zu RAMA sichtbar macht und eine Transformation spart.

Damit hat jede Zelle der Matrix eine Bedeutung in der Pipeline:

    TP  gleiches Partikel, Kosten positiv  -> zieht korrekt zusammen
    TN  verschiedene,      Kosten negativ  -> schneidet korrekt
    FP  verschiedene,      Kosten positiv  -> Druck Richtung REINHEITSFEHLER
    FN  gleiches Partikel, Kosten negativ  -> Druck Richtung FRAGMENTIERUNG

FP und FN sind also die Paar-Ebenen-Vorboten der beiden Cluster-Kennzahlen.
Gemessen: imbalance="weight" (TPR 0.57 / TNR 0.89) erzeugt viele FN und
fragmentiert stark; "sample" (0.75 / 0.65) ist ausgewogen.

ABGRENZUNG zu frac_neg (Solve): pred_neg_frac unten ist der Anteil negativ
bewerteter Paare auf dem HOLDOUT, frac_neg derselbe Anteil auf den echten
Kanten eines Solve-Fensters. Verwandt, aber nicht dasselbe -- der
Paar-Datensatz ist keine Stichprobe der Einsatzverteilung, weil die beiden
Klassen aus verschiedenen Runs stammen und gedeckelt werden. Ein Vergleich
ist eine Plausibilitaetspruefung; eine grosse Abweichung ist ein Befund,
kein Fehler.

NaN STATT AUSNAHME
------------------
Fehlt eine Klasse ganz, sind Precision/TPR/TNR/AUC nicht definiert. Diese
Funktionen liefern dann float("nan") und werfen NICHT: ein Trainingslauf
soll nicht daran scheitern, dass eine Kennzahl in einer Epoche undefiniert
ist. Wo eine fehlende Klasse WIRKLICH ein Fehler ist -- beim Holdout-Split --
wacht train.split_runs.
"""
from __future__ import annotations

import numpy as np

__all__ = ["confusion", "rates", "roc_auc", "class_balance", "pair_report",
           "flatten"]


def _div(a, b) -> float:
    """a/b, aber nan statt ZeroDivisionError. Siehe Modul-Docstring."""
    return float(a) / float(b) if b else float("nan")


def _as_labels(y) -> np.ndarray:
    """Labels als flaches Array, mit Pruefung auf {0, 1}.

    Der Paar-Datensatz haelt die Labels als float32 (0.0/1.0), Tests bauen
    sie als int -- beides ist erlaubt. Alles andere ist ein Aufruffehler und
    soll auffallen, statt still eine falsche Matrix zu erzeugen.
    """
    y = np.asarray(y).ravel()
    if y.size and not np.all((y == 0) | (y == 1)):
        raise ValueError("y_true darf nur 0 und 1 enthalten, gefunden: "
                         f"{np.unique(y)[:5]}")
    return y


def flatten(d: dict, sep: str = "/", prefix: str = "") -> dict:
    """Verschachteltes dict -> flach, Ebenen durch sep getrennt.

    Reine Strukturumformung: benennt nichts um, entscheidet nichts. Die
    Namen entstehen aus der Verschachtelung des Aufrufers.

    Gebraucht an der Logging-Grenze: W&B verflacht verschachtelte dicts zwar
    selbst, aber mit einem PUNKT (holdout.cm.tp), waehrend die
    Panel-Gruppierung sich nach dem SCHRAEGSTRICH richtet. Ohne die
    Umformung gaebe es keine Gruppen, sondern eine flache Wand aus Kacheln.
    GEMESSEN an wandb 0.28.1.

    Steht hier und nicht im Skript, damit sie testbar ist.
    """
    out: dict = {}
    for k, v in d.items():
        name = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, sep, f"{name}{sep}"))
        else:
            out[name] = v
    return out


def confusion(y_true, y_score, threshold: float = 0.0) -> dict[str, int]:
    """Konfusionsmatrix am OPERATIONSPUNKT des Solvers (Schwelle 0).

    y_score sind LOGITS (nicht sigmoid-transformiert). ">=" statt ">":
    ein Logit von exakt 0 gilt als JA.

    Rueckgabe: {"tp", "fp", "fn", "tn"} als int (nicht np.int64, damit
    json.dumps sie ohne Sonderbehandlung schreibt).
    """
    y = _as_labels(y_true)
    s = np.asarray(y_score).ravel()
    if y.shape != s.shape:
        raise ValueError(f"Laengen verschieden: y_true {y.shape}, "
                         f"y_score {s.shape}")
    pred = s >= threshold
    ist_ja = y == 1
    return {"tp": int(np.count_nonzero(pred & ist_ja)),
            "fp": int(np.count_nonzero(pred & ~ist_ja)),
            "fn": int(np.count_nonzero(~pred & ist_ja)),
            "tn": int(np.count_nonzero(~pred & ~ist_ja))}


def rates(cm: dict[str, int]) -> dict[str, float]:
    """Abgeleitete Kennzahlen aus einer Konfusionsmatrix.

    tpr und tnr heissen bewusst so und nicht recall/specificity: unter
    diesen Namen laufen sie durch Doku und W&B-Felder.

    balanced_accuracy = (tpr+tnr)/2 bleibt bei Imbalance aussagekraeftig --
    accuracy allein erreicht bei 6.3:1 schon durch "immer JA" den Wert 0.86.

    pred_neg_frac ist der Anteil negativ bewerteter Paare; Abgrenzung zu
    frac_neg siehe Modul-Docstring.
    """
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    n = tp + fp + fn + tn
    tpr = _div(tp, tp + fn)
    tnr = _div(tn, tn + fp)
    return {"accuracy": _div(tp + tn, n),
            "precision": _div(tp, tp + fp),
            "tpr": tpr,                       # = Recall / Sensitivitaet
            "tnr": tnr,                       # = Specificity
            "f1": _div(2 * tp, 2 * tp + fp + fn),
            "balanced_accuracy": 0.5 * (tpr + tnr),
            "pred_neg_frac": _div(fn + tn, n)}


def roc_auc(y_true, y_score) -> float:
    """AUC ueber die Rangsumme (Mann-Whitney-U). Ohne sklearn-Abhaengigkeit.

    GLEICHSTAENDE -- INVARIANTE: kind= hier NICHT setzen.

    Die Funktion vergibt jedem Wert einen EIGENEN Rang, auch bei gleichem
    Score; die Definition verlangt dort den MITTLEREN Rang. Das ist keine
    Randfrage: (dt, dr) ist quantisiert, und am Arbeitspunkt teilen sich
    98,8 % der Paare ihren Logit mit anderen.

    Trotzdem stimmen naive und mittlere Raenge auf sechs Nachkommastellen
    ueberein (gemessen an 11,6 Mio. Paaren: 0.870959, Delta unter 5e-7) --
    weil np.argsort per Default INSTABIL sortiert und die Gleichstaende sich
    wegmitteln. Die Funktion ist also richtig, aber sie haengt an einem
    Implementierungsdetail von NumPy, nicht an der Definition.

    Deshalb die Invariante: die naheliegende "Verbesserung" kind="stable"
    senkt die AUC um ~0.015, weil die JA-Seite in der Verkettung vorne steht
    und systematisch die niedrigeren Raenge bekaeme. Determinismus ist nur
    ZUSAMMEN mit mittleren Raengen zu haben, nie einzeln.
    """
    y_true = _as_labels(y_true)
    y_score = np.asarray(y_score).ravel()
    if y_true.shape != y_score.shape:
        raise ValueError(f"Laengen verschieden: y_true {y_true.shape}, "
                         f"y_score {y_score.shape}")
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) \
        / (len(pos) * len(neg))


def class_balance(label) -> dict[str, float]:
    """Klassenverhaeltnis EINER Menge von Paar-Labels.

    Wird mehrfach je Lauf aufgerufen: auf dem rohen Datensatz, nach dem
    Holdout-Split (getrennt fuer beide Seiten) und nach dem Untersampeln.
    Erst die Zahlen NEBENEINANDER zeigen, was Split und Balancing
    tatsaechlich getan haben -- eine allein sagt nichts.

    ratio_ja_nein ist JA/NEIN, also die "6.3" in "6.3 : 1".
    """
    lab = _as_labels(label)
    n_ja = int(np.count_nonzero(lab == 1))
    n_nein = int(np.count_nonzero(lab == 0))
    return {"n": int(lab.size), "n_ja": n_ja, "n_nein": n_nein,
            "ja_frac": _div(n_ja, lab.size),
            "ratio_ja_nein": _div(n_ja, n_nein)}


def pair_report(y_true, y_score, threshold: float = 0.0) -> dict:
    """Konfusionsmatrix + abgeleitete Raten + AUC in EINEM flachen Dict.

    Flach, weil das Ergebnis unveraendert an eine Logging-Senke gereicht
    werden soll. Die Zuordnung zu Panel-Praefixen ("cm/tp", "holdout/auc")
    passiert bewusst NICHT hier: das ist Darstellung und gehoert an die
    Aufrufstelle, nicht in die Messung.
    """
    cm = confusion(y_true, y_score, threshold)
    out: dict = dict(cm)
    out.update(rates(cm))
    out["auc"] = roc_auc(y_true, y_score)
    return out
