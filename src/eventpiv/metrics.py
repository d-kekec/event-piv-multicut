#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics.py -- Kennzahlen der PAAR-Ebene (Stufe [3]).

Reines NumPy: keine Torch-, keine Hydra-Abhaengigkeit. Die Funktionen sind
damit ohne Datensatz und ohne Framework testbar (BA_NN_Pipeline_Dateien.md
§2.4); der Aufrufer wandelt Tensoren vorher in Arrays um.

Bewusst NUR die Paar-Ebene. Reinheit und Fragmentierung sind Cluster-Ebene
und bleiben in evaluate.py -- zwei Zustaendigkeiten, zwei Module.

WARUM DIE SCHWELLE 0 UND KEINE ANDERE
-------------------------------------
Das Netz gibt einen LOGIT aus, und dieser Logit geht in solve.py UNVERAENDERT
als Kantenkosten an RAMA (model.py-Docstring, BA_RAMA_Verstaendnis.md §1.4).
RAMA liest das Vorzeichen: positiv = anziehend, negativ = abstossend. Die
Schwelle 0 ist deshalb nicht ein Arbeitspunkt unter vielen, sondern GENAU
der, den der Solver sieht. Wegen p = sigmoid(logit) ist "logit >= 0"
gleichbedeutend mit "p >= 0.5"; wir schwellen direkt am Logit, weil das den
Bezug zu RAMA sichtbar macht und eine Transformation spart.

Damit hat jede Zelle der Matrix eine Bedeutung in der Pipeline:

    TP  gleiches Partikel, Kosten positiv  -> zieht korrekt zusammen
    TN  verschiedene,      Kosten negativ  -> schneidet korrekt
    FP  verschiedene,      Kosten positiv  -> Druck Richtung REINHEITSFEHLER
    FN  gleiches Partikel, Kosten negativ  -> Druck Richtung FRAGMENTIERUNG

FP und FN sind also die Paar-Ebenen-Vorboten der beiden Cluster-Metriken.
Das erklaert die Zahlen aus BA_NN_Kantenkosten.md §3.6: imbalance="weight"
(TPR 0.57 / TNR 0.89) erzeugt viele FN und fragmentiert stark; "sample"
(0.75 / 0.65) ist ausgewogen.

ABGRENZUNG zu frac_neg (Solve): pred_neg_frac unten ist der Anteil negativ
bewerteter Paare auf dem Holdout, frac_neg derselbe Anteil auf den echten
Kanten eines Solve-Fensters. Verwandt, aber NICHT dasselbe: der Paar-
Datensatz ist keine Stichprobe der Einsatzverteilung (JA aus sauberen Runs,
NEIN aus Overlap-Runs, je gedeckelt -> 6.3:1 ist ein Artefakt der
Sammelstrategie, BA_NN_Kantenkosten.md §4.1). Ein Vergleich der beiden ist
eine Plausibilitaetspruefung, keine Identitaet; eine grosse Abweichung ist
ein Befund, kein Fehler.

NaN STATT AUSNAHME
------------------
Fehlt eine Klasse ganz, sind Precision/TPR/TNR/AUC nicht definiert. Diese
Funktionen liefern dann float("nan") und werfen NICHT. Grund: ein
Trainingslauf soll nicht daran scheitern, dass eine Kennzahl in einer Epoche
undefiniert ist -- NaN ist im Log sichtbar, ein Abbruch kostet den Lauf.
Die Stelle, an der eine fehlende Klasse WIRKLICH ein Fehler ist, ist der
Holdout-Split; dort wacht train.split_runs_stratified.
"""
from __future__ import annotations

import numpy as np

__all__ = ["confusion", "rates", "roc_auc", "class_balance", "pair_report"]


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


def confusion(y_true, y_score, threshold: float = 0.0) -> dict[str, int]:
    """Konfusionsmatrix am OPERATIONSPUNKT des Solvers (Schwelle 0).

    y_score sind LOGITS (nicht sigmoid-transformiert). ">=" statt ">":
    ein Logit von exakt 0 gilt als JA -- konsistent mit dem bisherigen
    "p >= 0.5" in train.evaluate.

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
    diesen Namen stehen sie in BA_NN_Kantenkosten.md §3.6, und Kontinuitaet
    zur Doku ist hier mehr wert als der Lehrbuchname.

    balanced_accuracy = (tpr+tnr)/2 ist die Kennzahl, die bei Imbalance
    aussagekraeftig bleibt -- accuracy allein waere bei 6.3:1 schon durch
    "immer JA" bei 0.86.

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

    WORTGLEICH aus train.py uebernommen (Commit C1, 29.07.2026) -- die
    Verschiebung darf das Ergebnis nicht veraendern.

    GLEICHSTAENDE -- Befund 29.07.2026, gemessen statt vermutet
    -----------------------------------------------------------
    Die Funktion vergibt je einen EIGENEN Rang, auch bei gleichen Scores;
    die Mann-Whitney-Definition verlangt dort den MITTLEREN Rang. Das ist
    hier keine Randfrage: (dt, dr) ist quantisiert (dt ganzzahlig in µs,
    dr aus ganzzahligen dx, dy bei r_max=3), viele Paare teilen sich also
    exakt denselben Logit.

    Am Extrembeispiel ist der Fehler drastisch -- ein JA und ein NEIN mit
    demselben Score liefern 0.0 statt korrekt 0.5. Auf realistischer Groesse
    ist er es NICHT: bei 200 000 Paaren mit 98 % Gleichstandsanteil liegt die
    Abweichung unter 1e-4. Grund: np.argsort benutzt per Default quicksort
    und ist damit INSTABIL, die Gleichstaende werden also durchmischt und
    mitteln sich weg.

    DIE FUNKTION IST ALSO RICHTIG -- AUS DEM FALSCHEN GRUND. Sie haengt an
    einem Implementierungsdetail von NumPy, nicht an der Definition.
    Gemessene Folge: dieselben Daten mit kind="stable" ergeben 0.698 statt
    0.713, weil die JA-Seite in der Verkettung vorne steht und bei stabiler
    Sortierung systematisch die niedrigeren Raenge bekaeme.

    KONSEQUENZ: kind= hier NICHT setzen. Die naheliegende "Verbesserung"
    (deterministisch sortieren) wuerde die AUC still um ~0.015 senken. Wer
    Determinismus will, muss zugleich auf mittlere Raenge umstellen -- beides
    zusammen, nie einzeln. Die dokumentierten 0.792 aus
    BA_NN_Kantenkosten.md §3.6 sind nach dieser Messung nicht betroffen.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
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

    Wird an drei Stellen aufgerufen (vom Prof angefordert): auf dem rohen
    Datensatz, nach dem Holdout-Split (getrennt fuer beide Seiten) und nach
    dem Untersampeln. Erst die drei Zahlen NEBENEINANDER zeigen, was das
    Balancing tatsaechlich getan hat -- eine allein sagt nichts.

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
