#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py -- BAUSTEIN [3]: Training und Bewertung des Kantenkosten-Netzes.

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
import resource
import time

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from eventpiv import config
from eventpiv.metrics import class_balance, pair_report
from eventpiv.model import (EdgeMLP, SWAP_RAW6, build_features, check_mode,
                            feature_names, is_symmetric, load_checkpoint,
                            n_features, save_checkpoint)
from eventpiv.runconfig import TrainCfg, identity_meta

IMBALANCE_MODES = ("sample", "weight")

def check_dataset_geom(meta: dict, nb, name: str = "") -> None:
    """Prueft, ob der Datensatz mit DIESER Geometrie gebaut wurde.

    Zwei unabhaengige Rechnungen desselben Werts: der abgeleitete Pfad
    behauptet eine Geometrie, die Metadaten der Datei nennen die
    tatsaechliche. Frueher baute train.py den Pfad ohne geom neu und lud
    damit still den Ellipsoid-Datensatz (Befund 27.07.2026).

    Aeltere Dateien ohne Metadatenblock laufen UNGEPRUEFT durch -- dieselbe
    Entscheidung wie in solve.check_consistency.
    """
    ds_nb = (meta or {}).get("neighborhood") or {}
    if ds_nb and ds_nb.get("geom") != nb.geom:
        raise ValueError(
            f"Datensatz {name} wurde mit geom={ds_nb.get('geom')!r} gebaut, "
            f"Konfiguration sagt {nb.geom!r} -- der Pfad passt nicht zum "
            "Inhalt.")


def load_features(path, feature_mode: str):
    """Laedt einen Paar-Datensatz und baut daraus die Feature-Matrix.

    EINE Stelle, an der aus pairs.npz Merkmale werden. Vorher stand die Kette
    (laden, casten, build_features) inline in train_edge_nn; ein zweiter
    Verbraucher haette sie dupliziert -- genau die Doppelung, gegen die
    model.build_features gebaut wurde.

    Liegt in train.py und nicht in pairs.py, obwohl der Name dorthin zeigt:
    pairs.py darf nicht von model.py abhaengen. Stufe [2] speichert rohe
    dt/dx/dy und baut nie Merkmale -- ein Import von model zoege torch in
    eine Stufe, die es nicht braucht.

    Rueckgabe: (feats float32 (n, n_in), label float32, run, meta dict)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} fehlt -- erst scripts/build_pair_dataset.py laufen "
            "lassen (gleiche neighborhood/pairs-Gruppe).")
    d = np.load(path, allow_pickle=False)

    kwargs = {}
    if feature_mode == "raw6":
        kwargs = {"ti": d["ti"].astype(np.float32),
                  "xi": d["xi"].astype(np.float32),
                  "yi": d["yi"].astype(np.float32)}
    feats = build_features(d["dt"].astype(np.float32),
                           d["dx"].astype(np.float32),
                           d["dy"].astype(np.float32),
                           mode=feature_mode, xp=np,
                           **kwargs).astype(np.float32)
    return (feats, d["label"].astype(np.float32), d["run"],
            json.loads(str(d["meta"])))

def _mit_cm(m: dict) -> dict:
    """Trennt die Konfusionsmatrix in einen eigenen Zweig.

    Aus dem flachen pair_report wird {..., "cm": {tp, fp, fn, tn}}. Der
    Grund ist Darstellung im Panel, aber die Trennung ist auch inhaltlich
    richtig: Zaehlungen und Raten sind verschiedene Groessen.
    """
    cm = {k: m[k] for k in ("tp", "fp", "fn", "tn")}
    return {**{k: v for k, v in m.items() if k not in cm}, "cm": cm}


def _epochenzeile(ep: int, loss_running: float, trn: dict, hold: dict,
                  sec: float) -> dict:
    """Baut die Logzeile einer Epoche. VERSCHACHTELT, nicht flach.

    Die oberste Ebene sagt, WELCHE MENGE gemessen wurde -- ohne sie waere
    "tp" mehrdeutig, sobald es Trainings- UND Holdout-Matrix gibt. Genau das
    ist seit E1 der Fall.

    ZWEI VERLUSTE auf der Trainingsseite, und der Unterschied ist kein Detail:
      train/loss_running  laufender Mittel ueber die Minibatches -- gerechnet
                          bei WECHSELNDEN Gewichten, also nicht mit
                          holdout/loss vergleichbar.
      train/loss          bei den Gewichten am Epochenende, also derselbe
                          Massstab wie holdout/loss.
    Erst der zweite macht den Vergleich ehrlich. Der erste bleibt, weil er
    nichts kostet und den Verlauf INNERHALB einer Epoche mittelt.

    Dass W&B verschachtelte dicts mit "/" verflacht, ist ein willkommener
    Nebeneffekt -- nicht der Grund fuer die Form.
    """
    return {"epoch": ep,
            "train": {"loss_running": loss_running, **_mit_cm(trn)},
            "holdout": _mit_cm(hold),
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


def train_edge_nn(cfg: TrainCfg, verbose=True, on_epoch=None, on_start=None):
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

    on_start : optional Callable, bekommt EINMAL vor der ersten Epoche das
        Datenblock-dict (Klassenverhaeltnisse, Standardisierung, Holdout).
        Getrennt von on_epoch, weil diese Werte konstant sind: als Verlauf
        geloggt waeren es 30 identische Punkte. In W&B gehoeren sie nach
        config/summary, nicht in die Historie. Und sie stehen fest, BEVOR
        die erste Epoche laeuft -- bricht der Job ab, sind sie schon da.

    Rueckgabe: (pfad, kennzahlen-dict). Die vollstaendige Historie steht
    daneben in model.metrics.json, nicht im Checkpoint: 30 Epochen x 16
    Felder haben in einer Gewichtsdatei nichts verloren.
    """
    t_start = time.perf_counter()
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
    feats, label, run_arr, ds_meta = load_features(data_path, tr.feature_mode)
    check_dataset_geom(ds_meta, nb, data_path.name)
    sym = is_symmetric(tr.feature_mode)

    if verbose:
        print(f"Datensatz: {data_path}")
        print(f"  {len(label):,} Paare "
              f"(JA {int(label.sum()):,} / NEIN {int((label == 0).sum()):,})")
        print(f"  alpha={nb.alpha}, r_max={nb.r_max}, geom={nb.geom}")
        print(f"Feature-Modus '{tr.feature_mode}': "
              f"{feature_names(tr.feature_mode)} "
              f"({'symmetrisch' if sym else 'NICHT symm. -> Mittelung'})")

    val_runs, is_val = split_runs_stratified(label, run_arr, tr.val_frac, rng)
    tr_idx = np.flatnonzero(~is_val)
    val_idx = np.flatnonzero(is_val)

    # Klassenverhaeltnis an VIER Stellen statt der drei angeforderten. Eine
    # Zahl allein sagt nichts -- erst der Vergleich zeigt, was Split und
    # Balancing tatsaechlich getan haben. Der Split ist run-getrennt, die
    # beiden Seiten haben deshalb NICHT automatisch dasselbe Verhaeltnis.
    daten = {
        "raw": class_balance(label),
        "train_split": class_balance(label[tr_idx]),
        "holdout": class_balance(label[val_idx]),
        "holdout_runs": sorted(val_runs),
        "holdout_hinweis": ("run-getrennter Holdout INNERHALB des "
                            "train-Splits -- NICHT der offizielle val-Split "
                            "(BA_NN_Kantenkosten.md §3.6)"),
        "pairs_path": str(data_path),
        "dataset_neighborhood": ds_meta.get("neighborhood", {})
    }
    if verbose:
        print(f"Holdout-Runs (aus demselben Split wie der Paar-Datensatz): "
              f"{daten['holdout_runs']}  (train {daten['train_split']['n']:,} "
              f"/ holdout {daten['holdout']['n']:,} Paare)")
        print(f"  Holdout-Klassen: JA {daten['holdout']['n_ja']:,} / "
              f"NEIN {daten['holdout']['n_nein']:,} "
              f"({daten['holdout']['ratio_ja_nein']:.2f} : 1)")

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
    # train_used ist die Menge, die WIRKLICH ins Training geht. Bei
    # imbalance="weight" ist sie identisch mit train_split -- und dass man
    # das im Log sieht, ist der Punkt: die Gewichtung sampelt nicht.
    daten["imbalance"] = tr.imbalance
    daten["train_used"] = class_balance(label[tr_idx])
    daten["pos_weight"] = (float(pos_weight.item())
                           if pos_weight is not None else None)

    # Standardisierung NUR aus den Trainingsdaten (sonst Leakage)
    mu = feats[tr_idx].mean(axis=0)
    sig = feats[tr_idx].std(axis=0) + 1e-8

    # Je Feature benannt statt als Array: so wird daraus data/mu/dt und
    # data/mu/dr statt eines Vektors, den kein Diagramm zeigen kann.
    # Diagnosewert: mu["dt"] ist der mittlere Zeitabstand des Datensatzes.
    # Aendert man alpha und mu["dt"] bleibt stehen, wurde der falsche
    # Datensatz geladen -- zwei Zahlen, die eine ganze Fehlerklasse fangen.
    namen = feature_names(tr.feature_mode)
    daten["mu"] = {k: float(v) for k, v in zip(namen, mu)}
    daten["sig"] = {k: float(v) for k, v in zip(namen, sig)}
    if verbose:
        print("Standardisierung: " + ",  ".join(
            f"{k}: mu={daten['mu'][k]:.4g} sig={daten['sig'][k]:.4g}"
            for k in namen))
    if on_start is not None:
        on_start(daten)

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

        # Auswertung auf BEIDEN Mengen, je Epoche. Der Trainingssatz kostet
        # etwa doppelt so viel wie der Holdout (4,83 gegen 2,38 Mio. Paare);
        # der Lauf waechst dadurch von rund 14 auf rund 27 Sekunden. Das ist
        # der Preis dafuer, Ueberanpassung in JEDER Kennzahl zu sehen statt
        # nur im Verlust -- und dafuer, dass die vom Betreuer angeforderte
        # Konfusionsmatrix fuer beide Mengen existiert.
        hold = evaluate(model, Xva, yva, mu_t, sig_t, sym, loss_fn)
        trn = evaluate(model, Xtr, ytr, mu_t, sig_t, sym, loss_fn)
        zeile = _epochenzeile(ep, tot / n, trn, hold,
                              time.perf_counter() - t_ep)
        historie.append(zeile)
        if on_epoch is not None:
            on_epoch(zeile)

        if verbose and (ep % 5 == 0 or ep == 1):
            h, tr_ = zeile["holdout"], zeile["train"]
            print(f"  ep {ep:3d}  loss {tr_['loss_running']:.4f} | "
                  f"train  loss {tr_['loss']:.4f} AUC {tr_['auc']:.4f} | "
                  f"holdout loss {h['loss']:.4f} AUC {h['auc']:.4f} "
                  f"TPR {h['tpr']:.3f} TNR {h['tnr']:.3f} "
                  f"[{zeile['diag']['sec_epoch']}s]")
    
    # diag misst den LAUF, nicht die Herkunft der Datei -- die steht in
    # config.build_meta. Zwei Zeitangaben, weil sie verschiedene Fragen
    # beantworten: sec_epochs ist Rechnen, sec_total enthaelt zusaetzlich
    # das Laden von 11,6 Mio. Paaren ueber NFS. Klaffen sie auseinander,
    # ist nicht das Netz langsam, sondern die Platte.
    #
    # ru_maxrss ist unter Linux KILObytes. Selbst messen ist hier Pflicht,
    # nicht Bequemlichkeit: SLURM buchhaltet Speicher auf diesem Cluster
    # nicht, sacct liefert eine leere MaxRSS-Spalte (gemessen 29.07.2026,
    # BA_Umgebung_Setup.md §0.9).
    diag = {
        "device": str(Xtr.device),
        "torch": torch.__version__,
        "max_rss_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "sec_epochs": round(sum(z["diag"]["sec_epoch"] for z in historie), 2),
        "sec_total": round(time.perf_counter() - t_start, 2),
    }
    # Verschachtelt wie die Epochenzeile -- daraus wird final/train/cm/tp und
    # final/holdout/auc, also dieselben Namen wie im Verlauf. Vorher lag die
    # Matrix im Verlauf unter holdout/cm/, in der Zusammenfassung aber flach
    # unter final/tp; das war eine unnoetige Stolperstelle.
    kennzahlen = {"train": _mit_cm(trn), "holdout": _mit_cm(hold),
                  "n_params": n_params, "val_runs": sorted(val_runs),
                  "diag": diag}
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
        print(f"  Diagnose: {diag['device']}, torch {diag['torch']}, "
              f"Spitzenspeicher {diag['max_rss_mb']:.0f} MB, "
              f"{diag['sec_epochs']:.1f}s rechnen / "
              f"{diag['sec_total']:.1f}s gesamt")

    out_path = (Path(cfg.io.out_path) if cfg.io.out_path
                else config.model_file(nb, ps, tr, create=True))
    path = save_checkpoint(
        out_path, model, mu, sig, tr.feature_mode, tr.hidden,
        **json.loads(config.meta_json(
            stage="edge_nn",
            **identity_meta(neighborhood=nb, pairs=ps, training=tr),
            data_path=str(data_path), **kennzahlen)))
    mpath = path.with_suffix(".metrics.json")
    mpath.write_text(json.dumps(
        {"identity": identity_meta(neighborhood=nb, pairs=ps, training=tr),
         "data": daten,
         "data_path": str(data_path),
         "model_path": str(path),
         "history": historie,
         "final": kennzahlen},
        indent=2, default=_json_default), encoding="utf-8")
    if verbose:
        print(f"-> {path}")
        print(f"-> {mpath}  ({len(historie)} Epochen)")
    return path, kennzahlen

def evaluate_pairs_on_dataset(cfg: TrainCfg, verbose=True) -> dict:
    """Haelt ein FERTIG trainiertes Netz gegen einen Paar-Datensatz.

    Kein Training, kein Split -- bewertet wird der GANZE Datensatz. Wozu:
    der Holdout in train_edge_nn liegt INNERHALB des train-Splits. Ein
    Datensatz aus val-Runs ist die erste unabhaengige Messung auf
    Paar-Ebene, und genau die hat der Betreuer angefragt.

    Der FREMDE Datensatz kommt ueber cfg.io.data_path herein: Netz aus
    train_c5_o10, Daten aus dem val-Zweig. Beide Pfade sind sonst abgeleitet.

    Das Praefix kommt aus den METADATEN des Datensatzes, nicht aus cfg --
    die Datei sagt selbst, aus welchem Split sie stammt. Ein val-Datensatz
    meldet val/..., ein test-Datensatz test/..., ohne dass der Aufrufer es
    richtig setzen muss.

    IN EINEM STUECK gerechnet, nicht stapelweise: bei hidden=8 ist die
    versteckte Schicht 11,6 Mio. x 8 float32, also rund 370 MB. Bei
    deutlich groesserem hidden waere das der Punkt, an dem solve.edge_costs
    stapelt -- dann hier ebenso.
    """
    nb, ps, tr = cfg.neighborhood, cfg.pairs, cfg.training
    check_mode(tr.feature_mode)

    mpath = (Path(cfg.io.model_path) if cfg.io.model_path
             else config.model_file(nb, ps, tr, create=False))
    if not mpath.exists():
        raise FileNotFoundError(f"{mpath} fehlt -- erst trainieren.")
    model, mu_t, sig_t, feature_mode, _ = load_checkpoint(mpath, device="cpu")
    if feature_mode != tr.feature_mode:
        raise ValueError(
            f"Checkpoint hat feature_mode={feature_mode!r}, Konfiguration "
            f"{tr.feature_mode!r} -- Pfad passt nicht zum Modell.")

    data_path = (Path(cfg.io.data_path) if cfg.io.data_path
                 else config.pairs_file(nb, ps, create=False))
    feats, label, _, ds_meta = load_features(data_path, feature_mode)
    check_dataset_geom(ds_meta, nb, data_path.name)
    split = ((ds_meta.get("pairs") or {}).get("split") or ps.split)

    X = (torch.tensor(feats) - mu_t) / sig_t
    y = torch.tensor(label).unsqueeze(1)
    m = evaluate(model, X, y, mu_t, sig_t, is_symmetric(feature_mode))

    kennzahlen = {"split": split, "metrics": _mit_cm(m),
                  "data": {**class_balance(label),
                           "path": str(data_path),
                           "model_path": str(mpath)}}
    if verbose:
        b = kennzahlen["data"]
        print(f"Netz     : {mpath}")
        print(f"Datensatz: {data_path}")
        print(f"  Split '{split}': {b['n']:,} Paare "
              f"(JA {b['n_ja']:,} / NEIN {b['n_nein']:,}, "
              f"{b['ratio_ja_nein']:.2f} : 1)")
        print(f"\n{split}: acc {m['accuracy']:.4f} | AUC {m['auc']:.4f} | "
              f"TPR(JA) {m['tpr']:.3f} | TNR(NEIN) {m['tnr']:.3f}")
        print(f"  Konfusion @ Logit 0:  TP {m['tp']:,}  FP {m['fp']:,}  "
              f"FN {m['fn']:,}  TN {m['tn']:,}")
    return kennzahlen