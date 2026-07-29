#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py -- BAUSTEIN [3]: Training des Kantenkosten-Netzes.

p = sigmoid(logit), BCE gegen das Paar-Label "gleiches Partikel?".
Metrik ist AUC: rangbasiert, schwellenunabhaengig und bei starker Imbalance
ehrlich -- Accuracy waere bei 266:1 wertlos.

WICHTIG (BA_NN_Kantenkosten.md §3.6, Nachtrag 19.07.):
AUC bewertet die Trennschaerfe des NETZES und ist dafuer die richtige Metrik
(Feature-Ablation, Kapazitaetsdiagnose, Trainingsverlauf). Sie taugt aber
NICHT zur Wahl von alpha/r_max: kleinere Nachbarschaften machen die
Paar-Trennung trivial (AUC steigt), waehrend die Cluster-Fragmentierung
steigt. Fuer die Parameterwahl zaehlen Cluster-Metriken, nicht AUC.

Val-Split NACH RUNS, stratifiziert (beide Klassen in val, sonst NaN bei
AUC/TPR). Split innerhalb eines Runs waere Leakage: Paare desselben Runs sind
hochgradig abhaengig.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from eventpiv import config
from eventpiv.model import (EdgeMLP, SWAP_RAW6, build_features, check_mode,
                            feature_names, is_symmetric, n_features,
                            save_checkpoint)
from eventpiv.metrics import pair_report
from eventpiv.runconfig import TrainCfg, identity_meta

IMBALANCE_MODES = ("sample", "weight")


def _epochenzeile(ep: int, train_loss: float, hold: dict, sec: float) -> dict:
    """Baut die Logzeile einer Epoche. VERSCHACHTELT, nicht flach.

    Die oberste Ebene sagt, WELCHE MENGE gemessen wurde -- "loss" heisst auf
    train etwas anderes als auf holdout, und "tp" waere ohne diese Ebene
    mehrdeutig, sobald es eine Trainings-Konfusionsmatrix gibt. Das ist eine
    semantische Trennung, keine Darstellungsfrage.

    Dass W&B verschachtelte dicts mit "/" verflacht (holdout/cm/tp), ist ein
    willkommener Nebeneffekt -- nicht der Grund fuer die Form.
    """
    cm = {k: hold[k] for k in ("tp", "fp", "fn", "tn")}
    rest = {k: v for k, v in hold.items() if k not in cm}
    return {"epoch": ep,
            "train": {"loss": train_loss},
            "holdout": {**rest, "cm": cm},
            "diag": {"sec_epoch": round(sec, 3)}}


def _json_default(o):
    """numpy-Skalare JSON-tauglich machen.

    Geprueft 29.07.2026: np.float64 ist zufaellig eine float-Unterklasse und
    ginge auch ohne -- np.int64 und np.float32 NICHT. Ein Lauf ueber viele
    Minuten darf nicht an der letzten Zeile sterben, weil irgendwo ein
    numpy-Skalar durchgerutscht ist. Unbekannte Typen fliegen weiterhin auf.
    """
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"nicht JSON-serialisierbar: {type(o).__name__}")



def split_runs_stratified(label, run_arr, val_frac, rng):
    """Val-Runs so waehlen, dass BEIDE Klassen vertreten sind.

    Ein Run ist eine JA- oder eine NEIN-Quelle (so wurde gesammelt); aus
    jeder Gruppe wandert val_frac in die Validierung.
    """
    ist_ja = {r: (label[run_arr == r].mean() >= 0.5)
              for r in np.unique(run_arr)}
    ja = np.array([r for r, v in ist_ja.items() if v])
    nein = np.array([r for r, v in ist_ja.items() if not v])
    rng.shuffle(ja)
    rng.shuffle(nein)

    def nimm(runs):
        # max(1, ...): mindestens EIN Run je Klasse muss in die Validierung,
        # sonst fehlt dort eine Klasse und AUC/TPR werden NaN.
        # min(len-1, ...): mindestens EIN Run je Klasse muss im TRAINING
        # bleiben. Ohne diese Schranke landeten bei nur einem Run je Klasse
        # beide im Holdout -- der Trainingssatz war leer und die Schleife
        # brach mit ZeroDivisionError ab (gefunden 27.07.2026).
        if len(runs) < 2:
            raise ValueError(
                f"Nur {len(runs)} Run(s) in einer Klasse. Ein run-getrennter "
                "Holdout braucht mindestens zwei je Klasse -- sonst ist "
                "entweder das Training oder die Validierung leer. "
                "clean_runs bzw. overlap_runs erweitern.")
        k = min(len(runs) - 1, max(1, int(round(len(runs) * val_frac)))) 
        return set(runs[:k].tolist())

    val_runs = nimm(ja) | nimm(nein)
    return val_runs, np.array([r in val_runs for r in run_arr])


def evaluate(model, X, y, mu, sig, symmetric=True, loss_fn=None) -> dict:
    """Kennzahlen auf einem Datensatz. Rueckgabe: FLACHES dict.

    Frueher ein 4-Tupel (acc, auc, tpr, tnr). Ein dict, weil die Kennzahlen
    unveraendert an eine Logging-Senke gereicht werden sollen und ein Tupel
    bei jeder neuen Groesse alle Aufrufstellen bricht.

    SCHWELLE AM LOGIT, nicht an p: mathematisch dasselbe (sigmoid ist streng
    monoton), aber in float32 rundet sigmoid(x) fuer winzige negative x auf
    exakt 0.5 -- die alte Schwelle zaehlte solche Kanten als JA, waehrend
    RAMA sie am Vorzeichen des Logits als abstossend liest. Gemessen
    29.07.2026: ueber 20 Laeufe liegt der Unterschied in acc/auc/tpr/tnr
    unter 1.3e-7, die Uebereinstimmung mit dem Solver ist dafuer exakt.
    Das Band, in dem beide Schwellen sich widersprechen, ist 1.8e-7 breit;
    auf 20 Mio. Logits realistischer Streuung faellt dort 0 bis 1 Kante
    hinein (gemessen 29.07.2026). Die Umstellung ist Korrektheit im
    Prinzip, keine Fehlerbehebung mit messbarer Folge.

    loss_fn : optional. Mit derselben Verlustfunktion wie im Training
        aufgerufen, damit train/loss und holdout/loss vergleichbar sind.
        VORSICHT bei imbalance="sample": der Trainingssatz ist untersampelt,
        der Holdout nicht -- die beiden Kurven liegen auf verschieden
        verteilten Daten. Die Richtung (laeuft der Holdout-Loss weg?) bleibt
        aussagekraeftig, der absolute Abstand nicht.
    """
    model.eval()
    with torch.no_grad():
        logit = model(X)
        if not symmetric:
            roh = X * sig + mu                      # zurueck in Rohgroessen
            getauscht = (roh[:, SWAP_RAW6] - mu) / sig
            logit = 0.5 * (logit + model(getauscht))
        out = {"loss": loss_fn(logit, y).item()} if loss_fn is not None else {}
        # squeeze(1) statt squeeze(): bei genau einem Paar wuerde squeeze()
        # einen 0-dimensionalen Tensor liefern und die Kennzahlen brechen.
        y_np = y.squeeze(1).numpy()
        s_np = logit.squeeze(1).numpy()
    out.update(pair_report(y_np, s_np))
    return out


def train_edge_nn(cfg: TrainCfg, verbose=True, on_epoch=None):
    """Trainiert das Kanten-Netz und schreibt model.pt an den abgeleiteten Pfad.

    neighborhood und pairs werden nicht gerechnet -- sie leiten den EINGABE-
    pfad ab. Die Geometrie bleibt trotzdem eine Eigenschaft des DATENSATZES:
    sie wird aus dessen Metadaten gelesen und gegen cfg.neighborhood geprueft.
    Zwei unabhaengige Rechnungen desselben Werts; stimmen sie nicht ueberein,
    luegt eine von beiden. Frueher baute diese Funktion den Pfad selbst neu --
    ohne geom, und lud damit still den Ellipsoid-Datensatz (Befund 27.07.2026).

    on_epoch : optional Callable, bekommt je Epoche EIN verschachteltes dict.
        Der Anschluss fuer W&B (scripts/train_edge_nn.py reicht wandb.log
        durch), ohne dass dieses Modul das Framework kennt -- dieselbe Grenze
        wie bei Hydra. Ein Test haengt stattdessen an eine Liste an.

    Rueckgabe: (pfad, kennzahlen-dict). Die vollstaendige Historie steht
    daneben in model.metrics.json, nicht im Checkpoint: 30 Epochen x 16
    Felder haben in einer Gewichtsdatei nichts verloren.
    """
    nb, ps, tr = cfg.neighborhood, cfg.pairs, cfg.training
    check_mode(tr.feature_mode)
    if tr.imbalance not in IMBALANCE_MODES:
        raise ValueError(f"imbalance muss in {IMBALANCE_MODES} liegen")
    
    if tr.epochs < 1:
        raise ValueError("training.epochs muss mindestens 1 sein -- sonst "
                         "gibt es keine Kennzahlen und keinen Anker.")
    
    torch.manual_seed(tr.seed)
    rng = np.random.default_rng(tr.seed)

    data_path = (Path(cfg.io.data_path) if cfg.io.data_path
                 else config.pairs_file(nb, ps, create=False))

    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} fehlt -- erst scripts/build_pair_dataset.py laufen "
            "lassen (gleiche neighborhood/pairs-Gruppe).")
    d = np.load(data_path, allow_pickle=False)

    dt = d["dt"].astype(np.float32)
    dx = d["dx"].astype(np.float32)
    dy = d["dy"].astype(np.float32)
    label = d["label"].astype(np.float32)
    run_arr = d["run"]

    ds_nb = json.loads(str(d["meta"])).get("neighborhood", {})
    if ds_nb and ds_nb.get("geom") != nb.geom:
        raise ValueError(
            f"Datensatz {data_path.name} wurde mit geom={ds_nb.get('geom')!r} "
            f"gebaut, Konfiguration sagt {nb.geom!r} -- der Pfad passt nicht "
            "zum Inhalt.")

    if verbose:
        print(f"Datensatz: {data_path}")
        print(f"  {len(label):,} Paare "
              f"(JA {int(label.sum()):,} / NEIN {int((label == 0).sum()):,})")
        print(f"  alpha={nb.alpha}, r_max={nb.r_max}, geom={nb.geom}")

    kwargs = {}
    if tr.feature_mode == "raw6":
        kwargs = {"ti": d["ti"].astype(np.float32),
                  "xi": d["xi"].astype(np.float32),
                  "yi": d["yi"].astype(np.float32)}
    feats = build_features(dt, dx, dy, mode=tr.feature_mode, xp=np,
                           **kwargs).astype(np.float32)
    sym = is_symmetric(tr.feature_mode)
    if verbose:
        print(f"Feature-Modus '{tr.feature_mode}': {feature_names(tr.feature_mode)} "
              f"({'symmetrisch' if sym else 'NICHT symm. -> Mittelung'})")

    val_runs, is_val = split_runs_stratified(label, run_arr, tr.val_frac, rng)
    tr_idx = np.flatnonzero(~is_val)
    val_idx = np.flatnonzero(is_val)
    if verbose:
        vlab = label[is_val]
        print(f"Holdout-Runs (aus demselben Split wie der Paar-Datensatz): "
              f"{sorted(val_runs)}"
              f"(train {len(tr_idx):,} / val {len(val_idx):,} Paare)")
        print(f"  Val-Klassen: JA {int((vlab == 1).sum()):,} / "
              f"NEIN {int((vlab == 0).sum()):,}")

    # --- Imbalance: Sampling ODER Gewichtung, nie beides ---
    pos_weight = None
    if tr.imbalance == "sample":
        tr_lab = label[tr_idx]
        ja_i = tr_idx[tr_lab == 1]
        nein_i = tr_idx[tr_lab == 0]
        n_keep = min(len(ja_i), int(len(nein_i) * tr.sample_ratio))
        tr_idx = np.concatenate([rng.choice(ja_i, n_keep, replace=False),
                                 nein_i])
        rng.shuffle(tr_idx)
        if verbose:
            print(f"Sampling: JA {n_keep:,} : NEIN {len(nein_i):,} "
                  f"(~{n_keep / max(len(nein_i), 1):.1f}:1)")
    else:
        tr_lab = label[tr_idx]
        n_pos = (tr_lab == 1).sum()
        n_neg = (tr_lab == 0).sum()
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        if verbose:
            print(f"Gewichtung: pos_weight = {pos_weight.item():.5f}")

    # Standardisierung NUR aus den Trainingsdaten (sonst Leakage)
    mu = feats[tr_idx].mean(axis=0)
    sig = feats[tr_idx].std(axis=0) + 1e-8

    def std(a):
        return (a - mu) / sig

    Xtr = torch.tensor(std(feats[tr_idx]))
    ytr = torch.tensor(label[tr_idx]).unsqueeze(1)
    Xva = torch.tensor(std(feats[val_idx]))
    yva = torch.tensor(label[val_idx]).unsqueeze(1)
    mu_t = torch.tensor(mu)
    sig_t = torch.tensor(sig)

    model = EdgeMLP(n_features(tr.feature_mode), tr.hidden)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Netz: {n_features(tr.feature_mode)}->{tr.hidden}->1, "
              f"{n_params} Parameter")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=tr.lr)

    n = len(Xtr)
    historie = []
    for ep in range(1, tr.epochs + 1):
        t_ep = time.perf_counter()
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for b in range(0, n, tr.batch):
            bi = perm[b:b + tr.batch]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[bi]), ytr[bi])
            loss.backward()
            opt.step()
            tot += loss.item() * len(bi)

        # Auswertung in JEDER Epoche -- die Kurve IST der Zweck des Loggings.
        # Kosten auf dem echten Holdout (2,38 Mio. Paare) 0.22s je Epoche
        # (gemessen 29.07.2026), ueber 30 Epochen rund 7 Sekunden. Aendert
        # das Ergebnis nicht: evaluate() zieht keine Zufallszahlen, rechnet
        # unter no_grad, und EdgeMLP hat weder Dropout noch BatchNorm.
        hold = evaluate(model, Xva, yva, mu_t, sig_t, sym, loss_fn)
        zeile = _epochenzeile(ep, tot / n, hold, time.perf_counter() - t_ep)
        historie.append(zeile)
        if on_epoch is not None:
            on_epoch(zeile)

        if verbose and (ep % 5 == 0 or ep == 1):
            h = zeile["holdout"]
            print(f"  ep {ep:3d}  loss {zeile['train']['loss']:.4f} | "
                  f"holdout loss {h['loss']:.4f} acc {h['accuracy']:.4f} "
                  f"AUC {h['auc']:.4f} TPR(JA) {h['tpr']:.3f} "
                  f"TNR(NEIN) {h['tnr']:.3f} [{zeile['diag']['sec_epoch']}s]")

    
    # Die Auswertung der LETZTEN Epoche ist die Schlussauswertung: zwischen
    # ihr und diesem Punkt aendert sich am Modell nichts mehr. Der frueher
    # hier stehende zweite evaluate()-Aufruf war eine exakte Wiederholung.
    kennzahlen = {**hold, "n_params": n_params, "val_runs": sorted(val_runs)}
    if verbose:
        print(f"\nFinal: loss {hold['loss']:.4f} | acc {hold['accuracy']:.4f} | "
              f"AUC {hold['auc']:.4f} | TPR(JA) {hold['tpr']:.3f} | "
              f"TNR(NEIN) {hold['tnr']:.3f}")
        # Schwelle 0 = das Vorzeichen, das RAMA liest. FP ist der Druck
        # Richtung Reinheitsfehler, FN der Richtung Fragmentierung.
        print(f"  Konfusion @ Logit 0:  TP {hold['tp']:,}  FP {hold['fp']:,}  "
              f"FN {hold['fn']:,}  TN {hold['tn']:,}")
        print(f"  Praezision {hold['precision']:.3f} | F1 {hold['f1']:.3f} | "
              f"ausgew. Acc {hold['balanced_accuracy']:.3f} | "
              f"negativ bewertet {hold['pred_neg_frac']*100:.1f}%")

    out_path = (Path(cfg.io.out_path) if cfg.io.out_path
                else config.model_file(nb, ps, tr))
    path = save_checkpoint(
        out_path, model, mu, sig, tr.feature_mode, tr.hidden,
        **json.loads(config.meta_json(
            stage="edge_nn",
            **identity_meta(neighborhood=nb, pairs=ps, training=tr),
            data_path=str(data_path), **kennzahlen)))
    mpath = path.with_suffix(".metrics.json")
    mpath.write_text(json.dumps(
        {"identity": identity_meta(neighborhood=nb, pairs=ps, training=tr),
         "data_path": str(data_path),
         "model_path": str(path),
         "history": historie,
         "final": kennzahlen},
        indent=2, default=_json_default), encoding="utf-8")
    if verbose:
        print(f"-> {path}")
        print(f"-> {mpath}  ({len(historie)} Epochen)")
    return path, kennzahlen
