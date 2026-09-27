#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train2.py -- BAUSTEIN [6b]: Training des Paar-Transformers.

  paar_ziele        Zielwert und Gueltigkeit je Paar, zwei Modi
  lade_kandidaten   die [6a]-Artefakte aller Runs zu EINEM Datensatz
  split_nach_runs   Holdout nach RUNS
  train_stufe2      die Schleife

Benannt nach train.py, eine Abstraktionsebene hoeher: formal dieselbe
Aufgabe -- Paar rein, ein Logit raus, Schwelle 0, RAMA.

HOLDOUT NACH RUNS, nicht nach Paaren. Zwei Tracklets desselben Partikels
sind hochgradig abhaengig; ein Paar-Split legte fast jedes
Validierungspaar neben ein Trainingspaar derselben Trajektorie und
meldete eine Guete, die im Einsatz niemand sieht.

DIE TOKENS ENTSTEHEN JE STAPEL auf dem Geraet. Alle Paare mal 2n+1 Tokens
vorzurechnen ist unmoeglich, der KNOTENtensor dagegen passt. Genau dafuer
nimmt transformer.baue_tokens Indizes statt fertiger Stapel.

GEZOGEN WIRD JE EPOCHE, der HOLDOUT dagegen einmal vor der Schleife.
Ueber die Epochen sieht das Netz damit mehr Paare als train_max; der
feste Holdout haelt holdout/loss zwischen Epochen vergleichbar.
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

# Obergrenze JEDER Bewertungsstichprobe -- Holdout wie Trainingsmenge.
# Beide tragen weit mehr Paare, als sich je Epoche zu bewerten lohnt; alle
# zu bewerten kostete so viel wie das Training selbst. KONSTANTE und kein
# Feld: die Zahl beeinflusst nur die Genauigkeit einer Kennzahl, nicht das
# Ergebnis, und gehoert damit nicht in die Identitaet -- der Slug bleibt
# unberuehrt.
VAL_MAX = 1_000_000


@dataclass
class Daten:
    """Alle Runs in EINEM Datensatz. Knoten auf dem Geraet, Paare im RAM.

    Die Knotenindizes sind GLOBAL: beim Laden wird je Run ein Versatz
    addiert, wie solve_run es fuer die Tracklet-IDs tut. Damit ist ein
    Paar durch zwei Zahlen bestimmt, ohne dass irgendwo eine Run-Nummer
    nachgeschlagen werden muss.


    NUR GUELTIGE PAARE: die uebrigen bekommen nie einen Gradienten und
    werden nie bewertet. Sie liegen je Run zusammen, in der Reihenfolge von
    kandidaten.runs und darin in der des Artefakts; grenzen[k]:grenzen[k+1]
    gehoert zu runs[k].
    """
    ev: torch.Tensor          # (M, n, 3) auf dem Geraet
    maske: torch.Tensor       # (M, n)
    t_min: torch.Tensor       # (M,)
    x_c: torch.Tensor
    y_c: torch.Tensor
    paar_i: np.ndarray        # (P,) int32, global
    paar_j: np.ndarray
    ziel: np.ndarray          # (P,) bool
    runs: np.ndarray          # (R,) Run-Nummern in Ladereihenfolge
    grenzen: np.ndarray       # (R+1,) Paar-Offsets je Run
    n_roh: np.ndarray         # (R,) Kandidatenpaare je Run, auch ungueltige


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


def _npz_form(f, name):
    """(Form, dtype) eines Arrays im npz, aus dem npy-Kopf -- ohne das Array
    zu entpacken."""
    with f.zip.open(f"{name}.npy") as fp:
        version = np.lib.format.read_magic(fp)
        lies = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                else np.lib.format.read_array_header_2_0)
        form, _, dtype = lies(fp)
    return form, dtype


def lade_kandidaten(cfg: Stufe2Cfg, device="cpu") -> Daten:
    """Die [6a]-Artefakte aller Runs zu einem Datensatz.

    Knotentensoren wandern auf das Geraet, Paar-Arrays bleiben im RAM: die
    ersten werden je Stapel indiziert, die zweiten nur geschnitten.

    ZWEI DURCHGAENGE, damit nichts doppelt liegt. Der erste liest nur die
    npy-Koepfe und kennt danach alle Groessen. Der zweite fuellt Tensoren
    und Arrays, die von Anfang an ihre endgueltige Groesse haben -- Knoten
    je Run direkt auf das Geraet, Paare je Run in die Paar-Arrays. Keine
    Teillisten neben einem zusammengefuegten Array.
    """
    kd, tf = cfg.kandidaten, cfg.transformer
    pfade = []
    for r in kd.runs:
        sel = Selection(split=kd.split, run_nr=r)
        pfad = config.kandidaten_file(cfg.neighborhood, cfg.pairs,
                                      cfg.training, sel, cfg.solver, kd)
        if not pfad.exists():
            raise SystemExit(
                f"{pfad} fehlt. Erst scripts/build_kandidaten.py mit "
                "denselben Gruppen laufen lassen.")
        pfade.append(pfad)

    m_r, p_r = [], []
    for pfad in pfade:
        with np.load(pfad) as f:
            if "gt_track_id" not in f.files:
                raise SystemExit(
                    f"{pfad.name} traegt keine GT -- ohne Label kein Training.")
            (m,), _ = _npz_form(f, "t_min")
            (p,), _ = _npz_form(f, "paar_i")
            form_ev, typ_ev = _npz_form(f, "ev")
            _, typ_m = _npz_form(f, "maske")
        m_r.append(m)
        p_r.append(p)
    n_knoten = sum(m_r)
    # int32 DURCHHALTEN statt aufweiten: paar_i und paar_j sind die beiden
    # groessten Arrays. Torch indiziert mit int32 -- Tokens, Logits und
    # Gradienten sind bitgleich (geprueft, torch 2.13). Ein Ueberlauf waere
    # STILL: int32 plus zu grossem Versatz ergibt ohne Warnung negative
    # Indizes.
    if n_knoten > np.iinfo(np.int32).max:
        raise SystemExit(
            f"{n_knoten:,} Knoten sprengen int32 -- paar_i/paar_j in "
            "lade_kandidaten auf int64 stellen.")

    def _auf_geraet(form, typ):
        typ_t = torch.from_numpy(np.empty(0, typ)).dtype
        return torch.empty(form, dtype=typ_t, device=device)

    ev = _auf_geraet((n_knoten, *form_ev[1:]), typ_ev)
    maske = _auf_geraet((n_knoten, form_ev[1]), typ_m)
    t_min, x_c, y_c = (_auf_geraet(n_knoten, np.float32) for _ in range(3))
    # Obergrenze ist die Zahl ALLER Kandidatenpaare; belegt werden nur die
    # gueltigen, der Rest bleibt ungenutzt.
    paar_i = np.empty(sum(p_r), np.int32)
    paar_j = np.empty(sum(p_r), np.int32)
    ziel = np.empty(sum(p_r), bool)
    grenzen = np.zeros(len(pfade) + 1, np.int64)

    a = b = 0
    for k, (pfad, m) in enumerate(zip(pfade, m_r)):
        with np.load(pfad) as f:
            ev[a:a + m].copy_(torch.from_numpy(f["ev"]))
            maske[a:a + m].copy_(torch.from_numpy(f["maske"]))
            for spalte, name in ((t_min, "t_min"), (x_c, "x_c"),
                                 (y_c, "y_c")):
                spalte[a:a + m].copy_(
                    torch.from_numpy(f[name]).to(torch.float32))
            i = f["paar_i"]
            j = f["paar_j"]
            z, g = paar_ziele(f["gt_track_id"], f["n_gt"], f["gt_anteil"],
                              i, j, modus=tf.label_modus,
                              min_anteil=tf.gt_min_anteil)
        n = int(g.sum())
        # Versatz wie bei den Tracklet-IDs in solve_run: danach ist ein
        # Paar durch zwei globale Zahlen bestimmt.
        paar_i[b:b + n] = i[g] + a
        paar_j[b:b + n] = j[g] + a
        ziel[b:b + n] = z[g] >= 0.5
        a += m
        b += n
        grenzen[k + 1] = b

    return Daten(ev=ev, maske=maske, t_min=t_min, x_c=x_c, y_c=y_c,
                 paar_i=paar_i[:b], paar_j=paar_j[:b], ziel=ziel[:b],
                 runs=np.asarray(kd.runs, np.int64), grenzen=grenzen,
                 n_roh=np.asarray(p_r, np.int64))


def split_nach_runs(runs, n_ja, n_nein, val_frac, seed, fest=()):
    """Holdout nach RUNS. Rueckgabe: val_runs, aufsteigend.

    runs, n_ja und n_nein sind parallel: je Run die Zahl gueltiger JA- und
    NEIN-Paare. Entschieden wird je Run, die Paare folgen ihrem Run.

    `fest` nennt Runs, die im Training bleiben muessen. Die Groesse des
    Holdouts folgt weiterhin ALLEN Runs -- sonst schruempfte er mit jedem
    festgesetzten Run und val_frac hiesse hier etwas anderes als in
    train.py.

    Beide Seiten muessen beide Klassen tragen, sonst sind TPR, TNR und AUC
    dort undefiniert -- und anders als bei einer einzelnen Epoche waere
    das kein Schoenheitsfehler, sondern eine wertlose Kennzahl ueber den
    ganzen Lauf. Deshalb bricht die Funktion ab statt nan zu liefern.
    """
    runs = np.asarray(runs)
    n_ja, n_nein = np.asarray(n_ja), np.asarray(n_nein)
    alle = np.unique(runs)
    fest = np.asarray(fest, dtype=alle.dtype).reshape(-1)
    unbekannt = np.setdiff1d(fest, alle)
    if len(unbekannt):
        # Ein Tippfehler bliebe sonst still folgenlos: der Run waere weder
        # fest noch im Holdout, weil es ihn im Datensatz nicht gibt.
        raise SystemExit(
            f"train_fest nennt Runs {list(unbekannt)}, die nicht im "
            f"Datensatz sind. Vorhanden: {list(alle)}")
    waehlbar = np.setdiff1d(alle, fest)
    n_val = max(1, int(round(len(alle) * val_frac)))
    if n_val >= len(alle):
        raise SystemExit(f"val_frac={val_frac} nimmt alle {len(alle)} Runs.")
    if n_val > len(waehlbar):
        raise SystemExit(
            f"val_frac={val_frac} verlangt {n_val} Holdout-Runs, aber nur "
            f"{len(waehlbar)} von {len(alle)} sind waehlbar "
            f"({len(fest)} in train_fest).")
    rng = np.random.default_rng(seed)
    val_runs = np.sort(rng.choice(waehlbar, n_val, replace=False))

    ist_val = np.isin(runs, val_runs)
    for name, seite in (("train", ~ist_val), ("val", ist_val)):
        ja, nein = int(n_ja[seite].sum()), int(n_nein[seite].sum())
        if ja == 0 or nein == 0:
            raise SystemExit(
                f"Holdout-Split {name}: nur eine Klasse ({ja} JA von "
                f"{ja + nein}). Anderen seed oder val_frac waehlen.")
    return val_runs


def _bloecke(grenzen, auswahl):
    """(lo, hi) der gewaehlten Runs in den Paar-Arrays, in Ladereihenfolge."""
    return [(int(grenzen[k]), int(grenzen[k + 1]))
            for k in np.flatnonzero(auswahl)]


def _holdout_index(grenzen, ist_val):
    """Alle Holdout-Paare, aufsteigend wie die Paare selbst."""
    teile = [np.arange(lo, hi, dtype=np.int64)
             for lo, hi in _bloecke(grenzen, ist_val)]
    return np.concatenate(teile) if teile else np.empty(0, np.int64)


def _toepfe(ziel, grenzen, ist_val):
    """JA- und NEIN-Topf des Trainings, als Indizes in die Paar-Arrays.

    Je Run gebaut, ohne eine Liste aller Trainingspaare daneben. Aufsteigend
    wie die Paare selbst: die Ziehung greift nach Position, dieselbe
    Reihenfolge heisst also dieselbe Ziehung. uint32, solange die Paarzahl
    es zulaesst -- halber Speicher gegenueber int64.
    """
    typ = np.uint32 if len(ziel) <= 2**32 else np.int64
    bloecke = _bloecke(grenzen, ~np.asarray(ist_val))
    n_ja = sum(int(ziel[lo:hi].sum()) for lo, hi in bloecke)
    n_alle = sum(hi - lo for lo, hi in bloecke)
    ja = np.empty(n_ja, typ)
    nein = np.empty(n_alle - n_ja, typ)
    a = b = 0
    for lo, hi in bloecke:
        z = ziel[lo:hi]
        k_ja, k_nein = np.flatnonzero(z), np.flatnonzero(~z)
        ja[a:a + len(k_ja)] = lo + k_ja
        nein[b:b + len(k_nein)] = lo + k_nein
        a += len(k_ja)
        b += len(k_nein)
    return ja, nein


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


def _durchlauf(model, d, idx, batch, device, norm=None, opt=None, lossfn=None):
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
                              alpha=norm["alpha"], r_max=norm["r_max"])
        with torch.set_grad_enabled(train):
            out = model(tok, tm)
            y = torch.from_numpy(d.ziel[teil]).to(device=device,
                                                   dtype=torch.float32)
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
    n_ja_r = np.array([int(d.ziel[lo:hi].sum())
                       for lo, hi in zip(d.grenzen[:-1], d.grenzen[1:])],
                      np.int64)
    n_nein_r = np.diff(d.grenzen) - n_ja_r
    mit = d.n_roh > 0            # ein Run ohne Kandidaten nimmt nicht teil
    val_runs = split_nach_runs(d.runs[mit], n_ja_r[mit], n_nein_r[mit],
                               tf.val_frac, tf.seed, cfg.io.train_fest)
    ist_val = np.isin(d.runs, val_runs)

    def _bilanz(auswahl):
        teile = [d.ziel[lo:hi] for lo, hi in _bloecke(d.grenzen, auswahl)]
        return class_balance(np.concatenate(teile).astype(np.int8))

    n_paare = int(d.n_roh.sum())
    daten = {
        "n_knoten": int(d.ev.shape[0]), "n_paare": n_paare,
        "n_gueltig": int(len(d.ziel)),
        "gueltig_frac": float(len(d.ziel) / n_paare),
        "val_runs": [int(r) for r in val_runs],
        "roh": class_balance(d.ziel.astype(np.int8)),
        "train_vor": _bilanz(~ist_val),
        "val": _bilanz(ist_val),
    }

    # Die Validierungsstichprobe EINMAL ziehen: eine je Epoche neu gezogene
    # liesse die Kurve mit der Ziehung schwanken statt mit dem Lernen. VOR
    # den Toepfen, damit die volle Holdout-Liste nie neben ihnen liegt.
    idx_v = _holdout_index(d.grenzen, ist_val)
    val_gestutzt = len(idx_v) > VAL_MAX
    if val_gestutzt:
        idx_v = np.sort(rng.choice(idx_v, VAL_MAX, replace=False))

    # Die Klassentoepfe EINMAL trennen -- der Gather ueber die ganze
    # Trainingsmenge faellt sonst je Epoche an.
    ja_topf, nein_topf = _toepfe(d.ziel, d.grenzen, ist_val)
    # Eigener Generator: eine Ziehung aus `rng` verschoebe jede folgende
    # Permutation und damit das Training selbst.
    rng_zieh = np.random.default_rng(tf.seed + 1)

    ratio = float(tf.sample_ratio)

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

    if val_gestutzt:
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
    lossfn = nn.BCEWithLogitsLoss()
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
                      **pair_report(d.ziel[idx_te].astype(np.int8), lg_t)},
            "holdout": {"loss": verlust_v,
                        **pair_report(d.ziel[idx_v].astype(np.int8), lg_v)},
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