#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train2.py -- BAUSTEIN [6b]: Training des Paar-Transformers.

  paar_ziele        Zielwert und Gueltigkeit je Paar, drei Modi
  lade_kandidaten   die [6a]-Artefakte aller Runs zu EINEM Datensatz
  split_nach_runs   Holdout nach RUNS
  train_stufe2      die Schleife

Benannt nach train.py, eine Abstraktionsebene hoeher: formal dieselbe
Aufgabe -- Paar rein, ein Logit raus, Schwelle 0, RAMA.

HOLDOUT NACH RUNS, nicht nach Paaren. Zwei Tracklets desselben Partikels
sind hochgradig abhaengig; ein Paar-Split legte fast jedes
Validierungspaar neben ein Trainingspaar derselben Trajektorie und
meldete eine Guete, die im Einsatz niemand sieht.

DIE TOKENS ENTSTEHEN JE STAPEL auf dem Geraet. 23,7 Mio. Paare mal 65
Tokens vorzurechnen ist unmoeglich; der KNOTENtensor dagegen passt --
rund 61 MB je Run bei cost_offset = -0,5, 610 MB fuer zehn. Genau dafuer
nimmt transformer.baue_tokens Indizes statt fertiger Stapel.

UNTERSAMPELN UND VALIDIERUNGSSTICHPROBE GESCHEHEN EINMAL, nicht je
Epoche: sonst schwankt die Verlustkurve mit der Ziehung statt mit dem
Lernfortschritt, und zwei Epochen waeren nicht vergleichbar.
"""
from __future__ import annotations

import resource
import time
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn as nn

from eventpiv.bewerten.metrics import class_balance, pair_report
from eventpiv.kern import config
from eventpiv.kern.runconfig import Selection, Stufe2Cfg, identity_meta
from eventpiv.kette.transformer import (PaarTransformer, baue_tokens,
                                        save_checkpoint)

# Obergrenze JEDER Bewertungsstichprobe -- Holdout wie Trainingsmenge. Der
# Holdout traegt bei zwei Runs rund 4,5 Mio. Paare, die Trainingsmenge bei
# dreissig Runs rund 12 Mio.; sie alle je Epoche zu bewerten kostete so
# viel wie das Training selbst. KONSTANTE und kein Feld: die Zahl
# beeinflusst nur die Genauigkeit einer Kennzahl, nicht das Ergebnis, und
# gehoert damit nicht in die Identitaet -- der Slug bleibt unberuehrt.
VAL_MAX = 1_000_000


@dataclass
class Daten:
    """Alle Runs in EINEM Datensatz. Knoten auf dem Geraet, Paare im RAM.

    Die Knotenindizes sind GLOBAL: beim Laden wird je Run ein Versatz
    addiert, wie solve_run es fuer die Tracklet-IDs tut. Damit ist ein
    Paar durch zwei Zahlen bestimmt, ohne dass irgendwo eine Run-Nummer
    nachgeschlagen werden muss.
    """
    ev: torch.Tensor          # (M, n, 3) auf dem Geraet
    maske: torch.Tensor       # (M, n)
    t_min: torch.Tensor       # (M,)
    x_c: torch.Tensor
    y_c: torch.Tensor
    paar_i: np.ndarray        # (P,) int32, global
    paar_j: np.ndarray
    ziel: np.ndarray          # (P,) float32 in [0, 1]
    gueltig: np.ndarray       # (P,) bool
    run: np.ndarray           # (P,) int16 -- die Einheit des Holdouts


def paar_ziele(gt, n_gt, gt_anteil, i, j, *, modus, min_anteil):
    """Zielwert und Gueltigkeit je Paar. Reine Funktion.

    Rueckgabe: (ziel float32 in [0,1], gueltig bool).

    streng    Beide Knoten genau ein GT-Partikel. Rauschknoten (n_gt = 0)
              und teilverschmolzene (n_gt >= 2) fallen heraus -- fuer
              beide gibt es keine Wahrheit, gegen die man messen koennte.
    mehrheit  Auch verschmolzene, hart nach Mehrheit, ab
              gt_anteil >= min_anteil. Rund 58 % der verschmolzenen
              Tracklets liegen darunter; ein hartes Label auf einer
              knappen Mehrheit waere nahe an einem Muenzwurf.

    Die Maske ist eine LOSS-Maske. Im Einsatz laufen alle Paare mit -- ein
    weggelassener Knoten ist ein Stueck Trajektorie, das niemand mehr
    verbindet.
    """
    gleich = gt[i] == gt[j]
    if modus == "streng":
        gueltig = (n_gt[i] == 1) & (n_gt[j] == 1)
        ziel = (gleich & gueltig).astype(np.float32)
    elif modus == "mehrheit":
        brauchbar = (n_gt >= 1) & (gt_anteil >= min_anteil)
        gueltig = brauchbar[i] & brauchbar[j]
        ziel = (gleich & gueltig).astype(np.float32)
    else:
        raise ValueError(f"Unbekannter label_modus {modus!r}")
    return ziel, gueltig


def lade_kandidaten(cfg: Stufe2Cfg, device="cpu") -> Daten:
    """Die [6a]-Artefakte aller Runs zu einem Datensatz.

    Knotentensoren wandern auf das Geraet, Paar-Arrays bleiben im RAM: die
    ersten werden je Stapel indiziert, die zweiten nur geschnitten.
    """
    kd, tf = cfg.kandidaten, cfg.transformer
    teile_ev, teile_m = [], []
    tmin, xc, yc = [], [], []
    pi, pj, ziel, gueltig, run = [], [], [], [], []
    versatz = 0

    for r in kd.runs:
        sel = Selection(split=kd.split, run_nr=r)
        pfad = config.kandidaten_file(cfg.neighborhood, cfg.pairs,
                                      cfg.training, sel, cfg.solver, kd)
        if not pfad.exists():
            raise SystemExit(
                f"{pfad} fehlt. Erst scripts/build_kandidaten.py mit "
                "denselben Gruppen laufen lassen.")
        with np.load(pfad) as f:
            if "gt_track_id" not in f.files:
                raise SystemExit(
                    f"{pfad.name} traegt keine GT -- ohne Label kein Training.")
            teile_ev.append(torch.from_numpy(f["ev"]))
            teile_m.append(torch.from_numpy(f["maske"]))
            tmin.append(f["t_min"]); xc.append(f["x_c"]); yc.append(f["y_c"])
            # int32 DURCHHALTEN statt aufweiten: paar_i und paar_j sind die
            # beiden groessten Arrays des Datensatzes, und weil die
            # Teillisten bis zum Funktionsende leben, zaehlt der Unterschied
            # doppelt. Bei 1,2 Mrd. Paaren sind es 9,5 statt 19 GB, im
            # Spitzenbedarf 18 statt 36. Torch indiziert mit int32 -- Tokens,
            # Logits und Gradienten sind bitgleich (geprueft, torch 2.13).
            i = f["paar_i"]
            j = f["paar_j"]
            z, g = paar_ziele(f["gt_track_id"], f["n_gt"], f["gt_anteil"],
                              i, j, modus=tf.label_modus,
                              min_anteil=tf.gt_min_anteil)
            # Versatz wie bei den Tracklet-IDs in solve_run: danach ist ein
            # Paar durch zwei globale Zahlen bestimmt.
            pi.append(i + versatz); pj.append(j + versatz)
            ziel.append(z); gueltig.append(g)
            run.append(np.full(len(i), r, np.int16))
            versatz += len(f["t_min"])
            # Ein Ueberlauf waere STILL: numpy liefert bei int32 + zu
            # grossem Summanden ein negatives Ergebnis ohne Warnung, und
            # jedes Paar zeigte danach auf den falschen Knoten.
            if versatz > np.iinfo(np.int32).max:
                raise SystemExit(
                    f"Knotenversatz {versatz:,} sprengt int32 -- paar_i/paar_j "
                    "in lade_kandidaten wieder auf int64 stellen.")

    def _dev(arrays, dtype):
        return torch.as_tensor(np.concatenate(arrays), dtype=dtype).to(device)

    return Daten(
        ev=torch.cat(teile_ev).to(device),
        maske=torch.cat(teile_m).to(device),
        t_min=_dev(tmin, torch.float32), x_c=_dev(xc, torch.float32),
        y_c=_dev(yc, torch.float32),
        paar_i=np.concatenate(pi), paar_j=np.concatenate(pj),
        ziel=np.concatenate(ziel), gueltig=np.concatenate(gueltig),
        run=np.concatenate(run))


def split_nach_runs(run, ziel, gueltig, val_frac, seed, fest=()):
    """Holdout nach RUNS. Rueckgabe: (idx_train, idx_val, val_runs).

    `fest` nennt Runs, die im Training bleiben muessen. Die Groesse des
    Holdouts folgt weiterhin ALLEN Runs -- sonst schruempfte er mit jedem
    festgesetzten Run und val_frac hiesse hier etwas anderes als in
    train.py.

    Beide Seiten muessen beide Klassen tragen, sonst sind TPR, TNR und AUC
    dort undefiniert -- und anders als bei einer einzelnen Epoche waere
    das kein Schoenheitsfehler, sondern eine wertlose Kennzahl ueber den
    ganzen Lauf. Deshalb bricht die Funktion ab statt nan zu liefern.

    NICHT train.split_runs wiederverwendet: die Funktion arbeitet auf dem
    Paar-Datensatz von Stufe 1, und ich habe sie in diesem Stand nicht
    gelesen. Ob sie passt, ist zu pruefen -- dann faellt diese hier weg.
    """
    runs = np.unique(run)
    fest = np.asarray(fest, dtype=runs.dtype).reshape(-1)
    unbekannt = np.setdiff1d(fest, runs)
    if len(unbekannt):
        # Ein Tippfehler bliebe sonst still folgenlos: der Run waere weder
        # fest noch im Holdout, weil es ihn im Datensatz nicht gibt.
        raise SystemExit(
            f"train_fest nennt Runs {list(unbekannt)}, die nicht im "
            f"Datensatz sind. Vorhanden: {list(runs)}")
    waehlbar = np.setdiff1d(runs, fest)
    n_val = max(1, int(round(len(runs) * val_frac)))
    if n_val >= len(runs):
        raise SystemExit(f"val_frac={val_frac} nimmt alle {len(runs)} Runs.")
    if n_val > len(waehlbar):
        raise SystemExit(
            f"val_frac={val_frac} verlangt {n_val} Holdout-Runs, aber nur "
            f"{len(waehlbar)} von {len(runs)} sind waehlbar "
            f"({len(fest)} in train_fest).")
    rng = np.random.default_rng(seed)
    val_runs = np.sort(rng.choice(waehlbar, n_val, replace=False))

    ist_val = np.isin(run, val_runs)
    idx_t = np.flatnonzero(gueltig & ~ist_val)
    idx_v = np.flatnonzero(gueltig & ist_val)
    for name, idx in (("train", idx_t), ("val", idx_v)):
        ja = int((ziel[idx] >= 0.5).sum())
        if ja == 0 or ja == len(idx):
            raise SystemExit(
                f"Holdout-Split {name}: nur eine Klasse ({ja} JA von "
                f"{len(idx)}). Anderen seed oder val_frac waehlen.")
    return idx_t, idx_v, val_runs


def _epochengroesse(n_ja_topf, n_nein_topf, ratio, max_paare):
    """Wieviel JA und NEIN eine Epoche zieht. Ohne rng, ohne Ziehung.

    NEIN ist die knappe Klasse und gibt die Groesse vor, JA folgt mit
    ratio. Der Deckel verkleinert beide im selben Verhaeltnis und laesst
    den Betriebspunkt unberuehrt.

    Rueckgabe: (n_ja, n_nein, erreicht). erreicht=False heisst, der
    JA-Topf reicht fuer ratio nicht -- dann folgt der Betriebspunkt den
    Daten statt der Konfiguration.
    """
    n_nein = n_nein_topf
    if max_paare:
        n_nein = min(n_nein, max(int(max_paare / (1.0 + ratio)), 1))
    n_ja = int(n_nein * ratio)
    return min(n_ja, n_ja_topf), n_nein, n_ja <= n_ja_topf


def _ziehe_epoche(ja, nein, n_ja, n_nein, rng):
    """Trainingsmenge EINER Epoche, je Klasse ohne Zuruecklegen.

    Je Epoche neu: bei einem Deckel von Faktor 50 saehe das Netz sonst
    ueber den ganzen Lauf dieselben 2 % des Datensatzes.
    """
    return np.sort(np.concatenate([
        rng.choice(ja, n_ja, replace=False) if n_ja < len(ja) else ja,
        rng.choice(nein, n_nein, replace=False) if n_nein < len(nein)
        else nein]))


def _durchlauf(model, d, idx, batch, device, ziel_t=None, opt=None, lossfn=None):
    """Ein Durchlauf ueber idx. Mit opt wird trainiert, ohne nur bewertet.

    Rueckgabe: (logits als numpy, mittlerer Verlust).
    """
    train = opt is not None
    model.train(train)
    logits, verlust, n = [], 0.0, 0
    for k in range(0, len(idx), batch):
        teil = idx[k:k + batch]
        i = torch.from_numpy(d.paar_i[teil]).to(device)
        j = torch.from_numpy(d.paar_j[teil]).to(device)
        tok, tm = baue_tokens(d.ev, d.maske, i, j, d.t_min, d.x_c, d.y_c,
                              alpha=ziel_t["alpha"], r_max=ziel_t["r_max"])
        with torch.set_grad_enabled(train):
            out = model(tok, tm)
            y = torch.from_numpy(d.ziel[teil]).to(device)
            l = lossfn(out, y)
        if train:
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
        logits.append(out.detach().float().cpu().numpy())
        # .detach().item() statt float(): float() auf einem Tensor mit
        # requires_grad haelt den Autograd-Graphen bis zum Zeilenende am
        # Leben. Hier folgenlos, aber torch warnt zu Recht.
        verlust += l.detach().item() * len(teil)
        n += len(teil)
    return np.concatenate(logits), verlust / max(n, 1)


def _diag(t0, device) -> dict:
    """Ressourcen, nie Ergebnis -- dieselbe Rolle wie lauf["diag"] in
    solve_run.

    gpu_peak_mb ist PyTorchs Hochwassermarke fuer TENSOREN, nicht der
    Verbrauch der Karte: der Caching-Allocator haelt mehr, gibt es aber
    wieder her. Fuer die Frage "passt der naechste Stapel" ist die Marke
    die richtige Zahl, fuer "passt noch ein zweiter Job" nicht.
    """
    aus = {"sec_total": round(time.time() - t0, 1),
           "max_rss_mb": round(
               resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
    if str(device).startswith("cuda") and torch.cuda.is_available():
        aus["gpu_peak_mb"] = round(
            torch.cuda.max_memory_allocated() / 1024**2, 1)
    return aus


def train_stufe2(cfg: Stufe2Cfg, on_start=None, on_epoch=None):
    """Trainiert den Paar-Transformer. Rueckgabe: (pfad, kennzahlen).

    on_start/on_epoch sind Senken wie in train.py: ein Test haengt an eine
    Liste an, das Skript reicht wandb durch. Dieses Modul kennt weder
    Hydra noch W&B.
    """
    tf, kd, cp = cfg.transformer, cfg.kandidaten, cfg.compute
    # VOR dem Rechnen, nicht danach: ein Abbruch nach einer Stunde Training
    # waere die teuerste Art, ueber einen Tippfehler zu stolpern. create
    # bleibt False -- save_checkpoint legt das Verzeichnis selbst an.
    pfad = config.transformer_file(cfg.neighborhood, cfg.pairs, cfg.training,
                                   cfg.solver, kd, tf)
    if pfad.exists() and not cfg.io.overwrite:
        raise SystemExit(
            f"{pfad} existiert bereits. Andere Konfiguration waehlen oder "
            "io.overwrite=true setzen.")
    device = cp.device
    t0 = time.time()
    torch.manual_seed(tf.seed)
    rng = np.random.default_rng(tf.seed)

    d = lade_kandidaten(cfg, device=device)
    idx_t, idx_v, val_runs = split_nach_runs(
        d.run, d.ziel, d.gueltig, tf.val_frac, tf.seed, cfg.io.train_fest)

    daten = {
        "n_knoten": int(d.ev.shape[0]), "n_paare": int(len(d.paar_i)),
        "n_gueltig": int(d.gueltig.sum()),
        "gueltig_frac": float(d.gueltig.mean()),
        "val_runs": [int(r) for r in val_runs],
        "roh": class_balance((d.ziel[d.gueltig] >= 0.5).astype(np.int8)),
        "train_vor": class_balance((d.ziel[idx_t] >= 0.5).astype(np.int8)),
        "val": class_balance((d.ziel[idx_v] >= 0.5).astype(np.int8)),
    }

    # Die Klassentoepfe EINMAL trennen -- der Gather ueber die ganze
    # Trainingsmenge faellt sonst je Epoche an.
    ja_topf = idx_t[d.ziel[idx_t] >= 0.5]
    nein_topf = idx_t[d.ziel[idx_t] < 0.5]
    # Eigener Generator: eine Ziehung aus `rng` verschoebe jede folgende
    # Permutation und damit das Training selbst.
    rng_zieh = np.random.default_rng(tf.seed + 1)

    pos_weight = None
    if tf.imbalance == "sample":
        ratio = float(tf.sample_ratio)
    else:
        # Bei "weight" bleibt das natuerliche Verhaeltnis; als Ziel
        # uebergeben wirkt der Deckel damit klassenneutral.
        ratio = len(ja_topf) / max(len(nein_topf), 1)
        pos_weight = torch.tensor([1.0 / max(ratio, 1e-9)], device=device)
        daten["pos_weight"] = float(pos_weight)

    n_ja, n_nein, erreicht = _epochengroesse(
        len(ja_topf), len(nein_topf), ratio, tf.train_max)
    n_ep = n_ja + n_nein
    daten["train_je_epoche"] = {
        "n": n_ep, "n_ja": n_ja, "n_nein": n_nein,
        "ja_frac": n_ja / max(n_ep, 1),
        "ratio_ja_nein": n_ja / max(n_nein, 1)}
    daten["sample_ratio_erreicht"] = bool(erreicht)
    if not erreicht:
        print(f"  WARN: sample_ratio={ratio} nicht erreichbar "
              f"({len(ja_topf):,} JA, {len(nein_topf):,} NEIN) -- der "
              "Betriebspunkt folgt den Daten, nicht der Konfiguration.",
              flush=True)

    # Die Validierungsstichprobe EINMAL ziehen: eine je Epoche neu gezogene
    # liesse die Kurve mit der Ziehung schwanken statt mit dem Lernen.
    if len(idx_v) > VAL_MAX:
        idx_v = np.sort(rng.choice(idx_v, VAL_MAX, replace=False))
        daten["val_gestutzt_auf"] = VAL_MAX

    if n_ep > VAL_MAX:
        daten["train_eval_gestutzt_auf"] = VAL_MAX

    model = PaarTransformer(d_model=tf.d_model, n_layers=tf.n_layers,
                            n_heads=tf.n_heads, d_ff=tf.d_ff,
                            dropout=tf.dropout, pooling=tf.pooling).to(device)
    daten["n_parameter"] = sum(p.numel() for p in model.parameters())
    if on_start is not None:
        on_start(daten)
    print(f"  {daten['n_knoten']:,} Knoten, {n_ep:,} Trainingspaare je Epoche, "
          f"{len(idx_v):,} Validierungspaare, "
          f"{daten['n_parameter']:,} Parameter", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=tf.lr)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    norm = {"alpha": float(kd.alpha), "r_max": float(kd.r_max)}

    if str(device).startswith("cuda") and torch.cuda.is_available():
        # Die Hochwassermarke NACH dem Laden zuruecksetzen: sonst misst sie
        # den Knotentensor und nicht den Trainingsstapel.
        torch.cuda.reset_peak_memory_stats()

    kennzahlen = {}
    for epoche in range(1, tf.epochs + 1):
        t_ep = time.time()
        # Je Epoche neu gezogen, dann permutiert.
        idx_t = _ziehe_epoche(ja_topf, nein_topf, n_ja, n_nein, rng_zieh)
        idx_te = idx_t
        if len(idx_te) > VAL_MAX:
            idx_te = np.sort(rng_zieh.choice(idx_t, VAL_MAX, replace=False))
        idx_e = idx_t[rng.permutation(len(idx_t))]
        _, verlust_lauf = _durchlauf(model, d, idx_e, tf.batch, device,
                                     norm, opt, lossfn)
        t_train = time.time() - t_ep
        # BEIDE Mengen nach dem Durchlauf, mit festen Gewichten. Der Wert
        # AUS dem Trainingsdurchlauf entsteht bei wechselnden Gewichten und
        # ist mit holdout/loss nicht vergleichbar -- er bleibt als
        # loss_running erhalten, weil er den Verlauf INNERHALB einer Epoche
        # mittelt und nichts kostet.
        lg_t, verlust = _durchlauf(model, d, idx_te, tf.batch * 4, device,
                                   norm, None, lossfn)
        lg_v, verlust_v = _durchlauf(model, d, idx_v, tf.batch * 4, device,
                                     norm, None, lossfn)
        # Hart geschwellt: die Konfusionsmatrix ist eine Aussage ueber
        # Entscheidungen, und RAMA entscheidet am Vorzeichen.
        zeile = {
            "epoch": epoche,
            "sec": round(time.time() - t_ep, 1),
            "sec_train": round(t_train, 1),
            "train": {"loss": verlust, "loss_running": verlust_lauf,
                      **pair_report((d.ziel[idx_te] >= 0.5).astype(np.int8),
                                    lg_t)},
            "holdout": {"loss": verlust_v,
                        **pair_report((d.ziel[idx_v] >= 0.5).astype(np.int8),
                                      lg_v)},
        }
        if on_epoch is not None:
            on_epoch(zeile)
        print(f"  Epoche {epoche:>3d}/{tf.epochs}  "
              f"loss {verlust:.4f}/{verlust_v:.4f}  "
              f"AUC {zeile['holdout']['auc']:.4f}  "
              f"TPR {zeile['holdout']['tpr']:.3f}  "
              f"TNR {zeile['holdout']['tnr']:.3f}  "
              f"[{zeile['sec']:.0f}s]", flush=True)
        kennzahlen = {k: v for k, v in zeile.items() if k != "epoch"}

    # EINMAL berechnen: zweimal gerufen lieferte zwei verschiedene
    # Sekundenzahlen, und Checkpoint und Kennzahlen widersprechen sich.
    diag = _diag(t0, device)
    save_checkpoint(pfad, model, tf, kd,
                    **identity_meta(neighborhood=cfg.neighborhood,
                                    pairs=cfg.pairs, training=cfg.training,
                                    solver=cfg.solver, kandidaten=kd,
                                    transformer=tf),
                    daten=daten, diag=diag)
    kennzahlen["daten"] = daten
    kennzahlen["diag"] = diag
    return pfad, kennzahlen