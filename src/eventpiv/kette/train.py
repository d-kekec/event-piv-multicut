#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py -- BAUSTEIN [3]: Training und Bewertung des Kantenkosten-Netzes.

p = sigmoid(logit), BCE gegen das Paar-Label "gleiches Partikel?".

WOFUER AUC TAUGT UND WOFUER NICHT: sie bewertet die Trennschaerfe des NETZES
und ist dafuer richtig (Merkmals-Ablation, Kapazitaetsdiagnose,
Trainingsverlauf). Sie taugt NICHT zur Wahl von alpha/r_max -- kleinere
Nachbarschaften machen die Paar-Trennung trivial, die AUC steigt, waehrend
die Cluster zerfallen. Die Hierarchie lautet Fragmentierung > Reinheit >>
AUC.

Holdout NACH RUNS, stratifiziert: beide Klassen muessen drin sein, sonst
werden AUC und TPR NaN. Ein Schnitt INNERHALB eines Runs waere Leakage --
Paare desselben Runs zeigen dieselben Partikel auf denselben Bahnen.
"""
from __future__ import annotations

import json
import resource
import time

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from eventpiv.bewerten.metrics import class_balance, pair_report
from eventpiv.kern import config
from eventpiv.kern.runconfig import TrainCfg, identity_meta
from eventpiv.kette.model import (EdgeMLP, build_features, check_mode,
                            feature_names, load_checkpoint,
                            n_features, save_checkpoint)


def check_dataset_neighborhood(meta: dict, nb, name: str = "") -> None:
    """Prueft, ob der Datensatz mit DIESER Nachbarschaft gebaut wurde.

    Zwei unabhaengige Rechnungen desselben Werts: der Pfad behauptet eine
    Nachbarschaft, die Metadaten der Datei nennen die tatsaechliche.

    Bei ABGELEITETEM Pfad ist das redundant -- slug(nb) deckt alle Felder ab.
    Wirksam wird die Pruefung genau dann, wenn io.data_path von Hand gesetzt
    ist (evaluate_pairs gegen einen val-Datensatz); dort garantiert nichts
    mehr die Uebereinstimmung.

    Verglichen wird gegen identity_meta und nicht ueber same_neighborhood:
    letzteres erwartet zwei Dataclasses, hier steht links ein dict aus JSON.

    Aeltere Dateien ohne Metadatenblock laufen UNGEPRUEFT durch -- dieselbe
    Entscheidung wie in solve.check_consistency.
    """
    ds_nb = (meta or {}).get("neighborhood") or {}
    jetzt = identity_meta(neighborhood=nb)["neighborhood"]
    if ds_nb and ds_nb != jetzt:
        raise ValueError(
            f"Datensatz {name} wurde mit neighborhood={ds_nb} gebaut, "
            f"Konfiguration sagt {jetzt} -- der Pfad passt nicht zum Inhalt.")


def load_features(path, feature_mode: str):
    """Laedt einen Paar-Datensatz und baut daraus die Feature-Matrix.

    EINE Stelle, an der aus pairs.npz Merkmale werden -- train_edge_nn und
    evaluate_pairs_on_dataset teilen sie sich.

    Liegt in train.py und nicht in pairs.py, obwohl der Name dorthin zeigt:
    Stufe [2] speichert rohe dt/dx/dy und baut nie Merkmale. Ein Import von
    model zoege torch in eine Stufe, die es nicht braucht.

    Rueckgabe: (feats float32 (n, n_in), label float32, run, meta dict)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} fehlt -- erst scripts/build_pair_dataset.py laufen "
            "lassen (gleiche neighborhood/pairs-Gruppe).")
    
    with np.load(path, allow_pickle=False) as d:
        if feature_mode == "drjac" and "jac" not in d:
            raise ValueError(
                f"{path} traegt keine jac-Spalte, feature_mode='drjac' "
                "braucht sie -- Datensatz mit dem aktuellen Stand von [2] "
                "neu bauen.")
        jac = d["jac"].astype(np.float32) if "jac" in d else None
        feats = build_features(d["dt"].astype(np.float32),
                               d["dx"].astype(np.float32),
                               d["dy"].astype(np.float32),
                               mode=feature_mode, xp=np,
                               jac=jac).astype(np.float32)
        return (feats, d["label"].astype(np.float32), d["run"],
                json.loads(str(d["meta"])))

def _mit_cm(m: dict) -> dict:
    """Trennt die Konfusionsmatrix in einen eigenen Zweig.

    Aus dem flachen pair_report wird {..., "cm": {tp, fp, fn, tn}}. Grund
    ist die Panel-Gruppierung, aber die Trennung ist auch inhaltlich
    richtig: Zaehlungen und Raten sind verschiedene Groessen.
    """
    cm = {k: m[k] for k in ("tp", "fp", "fn", "tn")}
    return {**{k: v for k, v in m.items() if k not in cm}, "cm": cm}


def _epochenzeile(ep: int, loss_running: float, trn: dict, hold: dict,
                  sec: float) -> dict:
    """Baut die Logzeile einer Epoche. VERSCHACHTELT, nicht flach.

    Die oberste Ebene sagt, WELCHE MENGE gemessen wurde -- ohne sie waere
    "tp" mehrdeutig, seit es eine Trainings- UND eine Holdout-Matrix gibt.

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

    np.float64 ist zufaellig eine float-Unterklasse und ginge auch ohne --
    np.int64 und np.float32 NICHT. Ein Lauf ueber viele Minuten darf nicht
    an der letzten Zeile sterben, weil irgendwo ein numpy-Skalar
    durchgerutscht ist. Unbekannte Typen fliegen weiterhin auf.
    """
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"nicht JSON-serialisierbar: {type(o).__name__}")



def split_runs(label, run_arr, val_frac, rng):
    """Run-getrennter Holdout, mit Pruefung auf beide Klassen.

    Getrennt nach RUNS und nicht nach Paaren: Paare desselben Runs sind
    stark korreliert (dieselben Partikel, dieselbe Wolke), ein zufaelliger
    Paar-Split maesse Auswendiglernen statt Verallgemeinerung.

    Frueher war ein Run eine JA- ODER eine NEIN-Quelle, und die Auswahl
    schichtete danach. Jetzt traegt JEDER Run beide Klassen -- beide stammen
    aus denselben Fenstern -- und es gibt nichts mehr zu schichten. An die
    Stelle der Schichtung tritt die Pruefung des Ergebnisses: ein Run kann
    sehr wenige NEIN tragen, und dann ist die Frage nicht, WIE gewaehlt
    wurde, sondern OB es gereicht hat.

    min(len-1, ...): mindestens EIN Run bleibt im Training. Ohne diese
    Schranke landeten bei zwei Runs beide im Holdout, der Trainingssatz
    waere leer und die Schleife braeche mit ZeroDivisionError ab.
    """
    runs = np.unique(run_arr)
    if len(runs) < 2:
        raise ValueError(
            f"Nur {len(runs)} Run(s) im Datensatz. Ein run-getrennter "
            "Holdout braucht mindestens zwei -- sonst ist entweder das "
            "Training oder die Validierung leer. pairs.runs erweitern.")

    perm = runs.copy()
    rng.shuffle(perm)
    k = min(len(perm) - 1, max(1, int(round(len(perm) * val_frac))))
    val_runs = set(perm[:k].tolist())
    is_val = np.array([r in val_runs for r in run_arr])

    for name, m in (("Holdout", is_val), ("Training", ~is_val)):
        vorhanden = np.unique(label[m])
        if len(vorhanden) < 2:
            raise ValueError(
                f"{name} enthaelt nur die Klasse(n) {vorhanden.tolist()}. "
                f"Gewaehlte Val-Runs: {sorted(val_runs)}. AUC und TPR waeren "
                "dort NaN. Mehr Runs in pairs.runs, oder Runs mit zu wenig "
                "NEIN aussortieren.")
    return val_runs, is_val


def evaluate(model, X, y, loss_fn=None) -> dict:
    """Kennzahlen auf einem Datensatz. Rueckgabe: FLACHES dict.

    Ein dict und kein Tupel, weil die Kennzahlen unveraendert an eine
    Logging-Senke gereicht werden -- ein Tupel braeche bei jeder neuen
    Groesse alle Aufrufstellen.

    SCHWELLE AM LOGIT, nicht an p: mathematisch dasselbe (sigmoid ist streng
    monoton), aber in float32 rundet sigmoid(x) fuer winzige negative x auf
    exakt 0.5 -- die alte Schwelle zaehlte solche Kanten als JA, waehrend
    RAMA sie am Vorzeichen als abstossend liest. Gemessener Unterschied ueber
    20 Laeufe: unter 1.3e-7. Korrektheit im Prinzip, keine Fehlerbehebung
    mit messbarer Folge -- die Uebereinstimmung mit dem Solver ist dafuer
    exakt.

    X kommt STANDARDISIERT herein. mu/sig standen frueher in der Signatur,
    weil ein unsymmetrischer Merkmalsmodus daraus die Rohgroessen
    zurueckrechnete, um ueber den Spaltentausch zu mitteln. Den Modus gibt
    es nicht mehr, und rueckwaerts rechnen muss diese Funktion nie.

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
    pfad ab. Die Nachbarschaft bleibt trotzdem eine Eigenschaft des
    DATENSATZES: sie wird aus dessen Metadaten gelesen und gegen
    cfg.neighborhood geprueft. Zwei unabhaengige Rechnungen desselben Werts;
    stimmen sie nicht ueberein, luegt eine von beiden.

    on_epoch : optional Callable, bekommt je Epoche EIN verschachteltes dict.
        Der Anschluss fuer W&B, ohne dass dieses Modul das Framework kennt --
        dieselbe Grenze wie bei Hydra.

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
    
    if tr.epochs < 1:
        raise ValueError("training.epochs muss mindestens 1 sein -- sonst "
                         "gibt es keine Kennzahlen und keinen Anker.")
    
    torch.manual_seed(tr.seed)
    rng = np.random.default_rng(tr.seed)

    data_path = (Path(cfg.io.data_path) if cfg.io.data_path
                 else config.pairs_file(nb, ps, create=False))
    feats, label, run_arr, ds_meta = load_features(data_path, tr.feature_mode)
    check_dataset_neighborhood(ds_meta, nb, data_path.name)

    if verbose:
        print(f"Datensatz: {data_path}")
        print(f"  {len(label):,} Paare "
              f"(JA {int(label.sum()):,} / NEIN {int((label == 0).sum()):,})")
        print(f"  alpha={nb.alpha}, r_max={nb.r_max}, geom={nb.geom}")
        print(f"Feature-Modus '{tr.feature_mode}': "
              f"{feature_names(tr.feature_mode)}")

    val_runs, is_val = split_runs(label, run_arr, tr.val_frac, rng)
    tr_idx = np.flatnonzero(~is_val)
    val_idx = np.flatnonzero(is_val)

    # Klassenverhaeltnis an VIER Stellen. Eine Zahl allein sagt nichts --
    # erst der Vergleich zeigt, was Split und Balancing getan haben. Der
    # Split ist run-getrennt, die beiden Seiten haben deshalb NICHT
    # automatisch dasselbe Verhaeltnis.
    daten = {
        "raw": class_balance(label),
        "train_split": class_balance(label[tr_idx]),
        "holdout": class_balance(label[val_idx]),
        "holdout_runs": sorted(val_runs),
        "holdout_hinweis": ("run-getrennter Holdout INNERHALB des "
                            "train-Splits -- NICHT der offizielle val-Split "),
        "pairs_path": str(data_path),
        "dataset_neighborhood": ds_meta.get("neighborhood", {}),
        # Aus dem Datensatz uebernommen, nicht aus der Konfiguration:
        # Labeling geht in keinen Slug ein, zwei Labelings landen also auf
        # demselben Pfad. Nur so laesst sich einem Netz nachtraeglich
        # ansehen, welche Wahrheit es gelernt hat.
        "dataset_labeling": ds_meta.get("labeling", {}),
        "dataset_label_format": ds_meta.get("label_format")
    }
    if verbose:
        print(f"Holdout-Runs (aus demselben Split wie der Paar-Datensatz): "
              f"{daten['holdout_runs']}  (train {daten['train_split']['n']:,} "
              f"/ holdout {daten['holdout']['n']:,} Paare)")
        print(f"  Holdout-Klassen: JA {daten['holdout']['n_ja']:,} / "
              f"NEIN {daten['holdout']['n_nein']:,} "
              f"({daten['holdout']['ratio_ja_nein']:.2f} : 1)")

    # --- Klassenverhaeltnis: JA untersampeln ---
    tr_lab = label[tr_idx]
    ja_i = tr_idx[tr_lab == 1]
    nein_i = tr_idx[tr_lab == 0]
    n_verlangt = int(len(nein_i) * tr.sample_ratio)
    n_keep = min(len(ja_i), n_verlangt)
    tr_idx = np.concatenate([rng.choice(ja_i, n_keep, replace=False),
                             nein_i])
    rng.shuffle(tr_idx)
    # Das Klassenverhaeltnis SETZT die Entscheidungsschwelle des Netzes:
    # bei 3:1 lernt es log-Odds gegen P(JA)=0.75, bei 1:1 gegen 0.5, und
    # die Differenz verschiebt jeden Logit um logit(0.75)-logit(0.5).
    # Reicht der Datensatz nicht, folgt die Schwelle also aus ja_vorrat
    # und ja_frac statt aus sample_ratio -- ein Betriebspunkt, den
    # niemand gewaehlt hat. Das muss auffallen.
    erreicht = n_keep >= n_verlangt
    if verbose:
        print(f"Sampling: JA {n_keep:,} : NEIN {len(nein_i):,} "
              f"(~{n_keep / max(len(nein_i), 1):.1f}:1)")
        if not erreicht:
            print(f"  WARN: sample_ratio={tr.sample_ratio} verlangt "
                  f"{n_verlangt:,} JA, vorhanden sind {len(ja_i):,}. Das "
                  f"Verhaeltnis folgt damit aus dem DATENSATZ, nicht aus "
                  f"sample_ratio -- pairs.ja_vorrat oder pairs.ja_frac "
                  f"erhoehen, sonst ist der Betriebspunkt geerbt statt "
                  f"gewaehlt.")
    # train_used ist die Menge, die WIRKLICH ins Training geht -- der
    # Vergleich mit train_split zeigt, was das Untersampeln getan hat.
    daten["imbalance"] = tr.imbalance
    daten["sample_ratio_erreicht"] = bool(erreicht)
    daten["train_used"] = class_balance(label[tr_idx])

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

    loss_fn = nn.BCEWithLogitsLoss()
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

        # Auswertung auf BEIDEN Mengen, je Epoche. Kostet etwa die doppelte
        # Laufzeit -- der Preis dafuer, Ueberanpassung in JEDER Kennzahl zu
        # sehen statt nur im Verlust, und dafuer, dass es die
        # Konfusionsmatrix fuer beide Mengen gibt.
        hold = evaluate(model, Xva, yva, loss_fn)
        trn = evaluate(model, Xtr, ytr, loss_fn)
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
    # config.build_meta. Zwei Zeitangaben: sec_epochs ist Rechnen, sec_total
    # enthaelt zusaetzlich das Laden ueber NFS. Klaffen sie auseinander, ist
    # nicht das Netz langsam, sondern die Platte.
    #
    # ru_maxrss ist unter Linux KILObytes. Selbst messen ist Pflicht: SLURM
    # buchhaltet Speicher auf diesem Cluster nicht, sacct liefert eine leere
    # MaxRSS-Spalte.
    diag = {
        "device": str(Xtr.device),
        "torch": torch.__version__,
        "max_rss_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "sec_epochs": round(sum(z["diag"]["sec_epoch"] for z in historie), 2),
        "sec_total": round(time.perf_counter() - t_start, 2),
    }
    # Verschachtelt wie die Epochenzeile -- daraus wird final/train/cm/tp und
    # final/holdout/auc, also dieselben Namen wie im Verlauf.
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
            labeling=ds_meta.get("labeling"),
            label_format=ds_meta.get("label_format"),
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
    Datensatz aus val-Runs ist die erste unabhaengige Messung auf Paar-Ebene.

    Der FREMDE Datensatz kommt ueber cfg.io.data_path herein; beide Pfade
    sind sonst abgeleitet.

    Das Praefix kommt aus den METADATEN des Datensatzes, nicht aus cfg --
    die Datei sagt selbst, aus welchem Split sie stammt, und der Aufrufer
    kann es nicht falsch setzen.

    IN EINEM STUECK gerechnet, nicht stapelweise: bei hidden=8 bleibt die
    versteckte Schicht unter einem Gigabyte. Bei deutlich groesserem hidden
    waere hier zu stapeln wie in solve.edge_costs.
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
    check_dataset_neighborhood(ds_meta, nb, data_path.name)
    split = ((ds_meta.get("pairs") or {}).get("split") or ps.split)

    X = (torch.tensor(feats) - mu_t) / sig_t
    y = torch.tensor(label).unsqueeze(1)
    m = evaluate(model, X, y)

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