#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pairs.py -- BAUSTEIN [2]: Paar-Datensatz fuer das Kantenkosten-NN.

Erzeugt Trainingspaare aus gelabelten Runs. Jedes Paar:
  - stammt aus der (alpha, r_max)-Nachbarschaft OHNE k-NN, ueber
    graph.neighborhood_pairs -- DIESELBE Funktion, die spaeter in solve die
    Graphkanten baut. Damit ist Training/Einsatz-Konsistenz strukturell
    erzwungen statt nur konventionell zugesichert.
  - beide Events SAUBER (status == 0), sonst kein verlaessliches Label
  - Label 1 = gleiche track_id (JA), 0 = verschieden (NEIN)

GESAMMELT WIRD IN DEN FENSTERN DES EINSATZES: anstossende Fenster der
Laenge `neighborhood.alpha`, harter Schnitt. Ein Paar ueber eine
Fenstergrenze kann in Stufe 1 nicht entstehen, also darf es hier nicht
gelernt werden.

BEIDE KLASSEN aus DENSELBEN Fenstern, ausschliesslich aus Overlap-Runs --
saubere Runs liefern physikalisch null NEIN. Fenster ohne NEIN gehoeren
dazu: ihre JA-Paare sind Teil der Einsatzverteilung.

RAM: ein Run liefert bei voller Abdeckung ueber eine Milliarde JA-Paare.
Geduennt wird deshalb IM Fenster (`ja_frac`), NEIN vollstaendig. Die ROHEN
Klassenzahlen laufen mit, sonst waere die Einsatzverteilung danach nicht
mehr rekonstruierbar.
"""
from __future__ import annotations

import resource          # Unix-only; SLURM buchhaltet Speicher hier nicht
import time
import json
from pathlib import Path

import numpy as np

from eventpiv import config
from eventpiv.graph import neighborhood_pairs, polarity_mask
from eventpiv.labels import STATUS_CLEAN
from eventpiv.runconfig import PairCfg, identity_meta


FIELDS = ("dt", "dx", "dy", "label", "run")



def collect_pairs(t, x, y, track_id, status, *, alpha, r_max, geom,
                  ja_frac, rng):
    """Schreitet EINEN Run in anstossenden Fenstern der Laenge alpha ab.

    Harter Schnitt, keine Ueberlappung, keine Erweiterung des rechten
    Randes -- genau die Fenster, die Stufe 1 im Einsatz bildet.

    ja_frac : Anteil der je Fenster gezogenen JA-Paare. NEIN bleibt
        vollstaendig, es ist die knappe Klasse (gemessen 263:1 bis 1785:1).
    rng : vom Aufrufer, damit die ganze Sammlung an EINEM Seed haengt.

    Rueckgabe: (data, zaehler). data hat dt, dx, dy, label (ohne 'run');
    zaehler die ROHEN Klassenzahlen vor dem Ziehen.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(t) > 1 and not (np.diff(t) >= 0).all():
        # Die Fenstergrenzen entstehen ueber searchsorted -- bei
        # unsortiertem t schnitte das an beliebigen Stellen, und
        # neighborhood_pairs braucht die Sortierung nicht.
        raise ValueError("collect_pairs erwartet aufsteigend sortierte t")

    acc = {k: [] for k in FIELDS if k != "run"}
    zaehler = {"n_roh_ja": 0, "n_roh_nein": 0, "n_fenster": 0,
               "n_fenster_ohne_nein": 0}
    if len(t) < 2:
        leer = {k: np.empty(0, np.float32) for k in acc}
        leer["label"] = np.empty(0, np.int8)
        return leer, zaehler

    # Anstossende Fenster ab dem ERSTEN Event. searchsorted statt Masken:
    # bei 800 Fenstern und 17 Mio. Events der Unterschied zwischen
    # Sekunden und Minuten.
    grenzen = np.searchsorted(t, np.arange(t[0], t[-1], alpha))
    grenzen = np.append(grenzen, len(t))

    for lo, hi in zip(grenzen[:-1], grenzen[1:]):
        if hi - lo < 2:
            continue
        zaehler["n_fenster"] += 1
        tw, xw, yw = t[lo:hi], x[lo:hi], y[lo:hi]
        tidw, staw = track_id[lo:hi], status[lo:hi]

        i, j = neighborhood_pairs(tw, xw, yw, alpha, r_max, geom=geom)
        if len(i) == 0:
            # Auch das ist ein Fenster ohne NEIN -- sonst zaehlte
            # n_fenster_ohne_nein systematisch zu niedrig.
            zaehler["n_fenster_ohne_nein"] += 1
            continue

        keep = (staw[i] == STATUS_CLEAN) & (staw[j] == STATUS_CLEAN)
        i, j = i[keep], j[keep]
        if len(i) == 0:
            zaehler["n_fenster_ohne_nein"] += 1
            continue

        ja = tidw[i] == tidw[j]
        n_ja, n_nein = int(ja.sum()), int((~ja).sum())
        zaehler["n_roh_ja"] += n_ja
        zaehler["n_roh_nein"] += n_nein
        if n_nein == 0:
            zaehler["n_fenster_ohne_nein"] += 1

        # NEIN vollstaendig, JA anteilig. Bernoulli statt fester Anzahl:
        # die Fenster tragen sehr verschieden viele Paare (Median 2,7 Mio.,
        # Maximum 11,4 Mio.).
        idx_ja = np.flatnonzero(ja)
        gezogen = idx_ja[rng.random(len(idx_ja)) < ja_frac]
        sel = np.concatenate([np.flatnonzero(~ja), gezogen])
        if len(sel) == 0:
            continue
        sel.sort()
        i, j, lab = i[sel], j[sel], ja[sel].astype(np.int8)

        acc["dt"].append((tw[j] - tw[i]).astype(np.float32))   # >= 0
        acc["dx"].append((xw[j] - xw[i]).astype(np.float32))
        acc["dy"].append((yw[j] - yw[i]).astype(np.float32))
        acc["label"].append(lab)

    if not acc["dt"]:
        out = {k: np.empty(0, np.float32) for k in acc}
        out["label"] = np.empty(0, np.int8)
        return out, zaehler
    return {k: np.concatenate(v) for k, v in acc.items()}, zaehler


def cap_per_class(data: dict, max_ja, max_nein, rng) -> dict:
    """Deckelt JE KLASSE, nicht ueber beide zusammen.

    Ein gemeinsamer Deckel zoege blind aus dem Gemisch, und das
    Klassenverhaeltnis waere danach eine Folge des Deckels statt der Daten
    -- unbemerkt.
    """
    lab = data["label"]
    ja = np.flatnonzero(lab == 1)
    nein = np.flatnonzero(lab == 0)
    if max_ja is not None and len(ja) > max_ja:
        ja = rng.choice(ja, max_ja, replace=False)
    if max_nein is not None and len(nein) > max_nein:
        nein = rng.choice(nein, max_nein, replace=False)
    sel = np.concatenate([ja, nein])
    sel.sort()
    return {k: v[sel] for k, v in data.items()}


def build_pairs_for_run(cfg: PairCfg, run_nr, rng):
    """Laedt die Label-Datei eines Runs und sammelt daraus Paare.

    Ein Gang je Run, beide Klassen. Rueckgabe: (data, zaehler).
    """
    nb, ps = cfg.neighborhood, cfg.pairs
    path = (Path(cfg.io.label_dir) / f"run{run_nr}_eventlabels.npz"
            if cfg.io.label_dir else config.label_file(ps.split, run_nr))
    if not path.exists():
        raise FileNotFoundError(
            f"{path} fehlt -- erst scripts/build_event_labels.py "
            f"selection.split={ps.split} selection.runs=[{run_nr}] laufen lassen.")
    with np.load(path) as d:
        # Das Labeling aus der DATEI lesen, nicht aus der Konfiguration:
        # welche Parameter diese Labels erzeugt haben, weiss nur sie.
        meta = json.loads(str(d["meta"]))
        lb = (meta.get("labeling"), meta.get("label_format"))
        # VOR dem float64-Cast in collect_pairs: bei polarity="on" halbiert
        # das nebenbei den festen Speicheraufwand je Run.
        m = polarity_mask(d["p"], nb.polarity)
        daten, zaehler = collect_pairs(
            d["t"][m], d["x"][m], d["y"][m], d["track_id"][m], d["status"][m],
            alpha=nb.alpha, r_max=nb.r_max, geom=nb.geom,
            ja_frac=ps.ja_frac, rng=rng)
    return daten, zaehler, lb


def build_pair_dataset(cfg: PairCfg, verbose=True, on_summary=None):
    """Baut den kompletten Paar-Datensatz und schreibt ihn als .npz.

    on_summary : optional Callable, bekommt EINMAL am Ende die Kennzahlen
        samt Ausbeute je Run. Nur eine Senke, nicht zwei wie im Training:
        die Ausbeuten sind eine TABELLE, keine Reihe -- sie haben keine
        Ordnung in der Zeit, nur je Run.

    Der Pfad wird ABGELEITET, cfg.io.out_path ist der Notausgang.

    EIN rng fuer Ziehen UND Deckeln, Runs in der Reihenfolge von ps.runs
    -> reproduzierbar bei gleichem seed.
    """
    nb, ps = cfg.neighborhood, cfg.pairs

    rng = np.random.default_rng(ps.seed)
    t_start = time.time()
    teile = []
    je_run = []
    roh_ja = roh_nein = n_fenster = 0
    labeling = None

    if verbose:
        print(f"Paare aus {len(ps.runs)} Run(s), Fenster {nb.alpha/1000:.0f}ms, "
              f"ja_frac={ps.ja_frac:g}:")
    for r in ps.runs:
        tic = time.time()
        roh, z, lb = build_pairs_for_run(cfg, r, rng)
        # ALLE Runs muessen dasselbe Labeling tragen. Sonst stuenden in
        # einem Datensatz Paare aus zwei verschiedenen Wahrheiten, und
        # keine Kennzahl darauf waere etwas wert.
        if labeling is None:
            labeling = lb
        elif lb != labeling:
            raise ValueError(
                f"run{r} wurde mit anderen Labeling-Parametern gelabelt als "
                f"die vorigen Runs: {lb} gegen {labeling}. Label-Dateien "
                "einheitlich neu bauen (scripts/build_event_labels.py "
                "io.overwrite=true).")
        # VOR dem Deckel zaehlen: sonst waere ein greifender max_yes nicht
        # von der ja_frac-Duennung zu unterscheiden.
        vor_deckel = roh["label"]
        n_gezogen_ja = int((vor_deckel == 1).sum())
        n_gezogen_nein = int((vor_deckel == 0).sum())

        d = cap_per_class(roh, ps.max_yes, ps.max_no, rng)
        lab = d["label"]
        n_ja, n_nein = int((lab == 1).sum()), int((lab == 0).sum())
        d["run"] = np.full(len(lab), r, np.int16)
        teile.append(d)

        roh_ja += z["n_roh_ja"]
        roh_nein += z["n_roh_nein"]
        n_fenster += z["n_fenster"]
        # Erst die ROHEN Zahlen sagen, welches Klassenverhaeltnis das Netz
        # im EINSATZ vorfindet -- ohne sie ist der Betriebspunkt nicht
        # rekonstruierbar.
        je_run.append({
            "run": int(r), "n_fenster": z["n_fenster"],
            "n_fenster_ohne_nein": z["n_fenster_ohne_nein"],
            "n_roh_ja": z["n_roh_ja"], "n_roh_nein": z["n_roh_nein"],
            "n_gezogen_ja": n_gezogen_ja, "n_gezogen_nein": n_gezogen_nein,
            "n_ja": n_ja, "n_nein": n_nein,
            "gedeckelt_ja": bool(n_ja < n_gezogen_ja),
            "gedeckelt_nein": bool(n_nein < n_gezogen_nein),
            "sec": round(time.time() - tic, 2)})
        z_ = je_run[-1]
        deckel = "".join([
            "  (JA gedeckelt)" if z_["gedeckelt_ja"] else "",
            "  (NEIN gedeckelt)" if z_["gedeckelt_nein"] else ""])
        if verbose:
            print(f"  run{r:<3d}: {n_ja:>9,} JA / {n_nein:>9,} NEIN behalten "
                  f"aus {z['n_roh_ja']:,} / {z['n_roh_nein']:,} roh "
                  f"({z['n_fenster']} Fenster, {z['n_fenster_ohne_nein']} ohne "
                  f"NEIN){deckel}")

    # Feldweise zusammenfuegen und die Teile SOFORT freigeben. Sonst liegen
    # am Ende alle Teile UND das Ergebnis gleichzeitig im RAM (Faktor 2).
    data = {}
    for k in FIELDS:
        data[k] = np.concatenate([p[k] for p in teile])
        for p in teile:
            del p[k]
    teile.clear()

    n_yes = int((data["label"] == 1).sum())
    n_no = int((data["label"] == 0).sum())

    leer = [z["run"] for z in je_run if z["n_ja"] + z["n_nein"] == 0]
    gedeckelt = [z["run"] for z in je_run
                 if z["gedeckelt_ja"] or z["gedeckelt_nein"]]

    if verbose:
        print("=" * 52)

    out_path = (Path(cfg.io.out_path) if cfg.io.out_path
                else config.pairs_file(nb, ps))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **data,
             meta=config.meta_json(stage="pair_dataset",
                                   **identity_meta(neighborhood=nb, pairs=ps),
                                   # Labeling geht in keinen Slug ein: ohne
                                   # dieses Feld liesse sich nicht mehr
                                   # feststellen, welche Wahrheit dieser
                                   # Datensatz gelernt hat.
                                   labeling=labeling[0],
                                   label_format=labeling[1],
                                   n_yes=n_yes, n_no=n_no))
    kennzahlen = {
        "n_total": int(len(data["label"])), "n_ja": n_yes, "n_nein": n_no,
        "ratio_ja_nein": (n_yes / n_no) if n_no else float("nan"),
        "n_runs": len(je_run),
        "n_runs_leer": len(leer),
        "n_runs_gedeckelt": len(gedeckelt),
        "n_roh_ja": roh_ja, "n_roh_nein": roh_nein,
        # DIAGNOSE, kein Sollwert: die Schwelle setzt bereits das
        # Untersampeln, und ein Versatz hierauf machte fast jede Kante
        # positiv.
        "logit_prior_roh": (float(np.log(roh_ja / roh_nein))
                            if roh_ja and roh_nein else float("nan")),
        "n_fenster": n_fenster,
        "covered_ms": round(n_fenster * nb.alpha / 1000.0, 1),
        "out_path": str(out_path),
        "diag": {
            "sec_total": round(time.time() - t_start, 1),
            "max_rss_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        },
        "per_run": je_run,
    }
    if on_summary is not None:
        on_summary(kennzahlen)
    if verbose:
        print(f"-> {out_path}")
    return out_path, data
