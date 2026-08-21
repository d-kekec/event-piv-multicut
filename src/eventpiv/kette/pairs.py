#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pairs.py -- BAUSTEIN [2]: Paar-Datensatz fuer das Kantenkosten-NN.

Erzeugt Trainingspaare aus gelabelten Runs. Jedes Paar:
  - stammt aus der (alpha, r_max)-Nachbarschaft OHNE k-NN, ueber
    graph.neighborhood_pairs -- DIESELBE Funktion, die spaeter in solve die
    Graphkanten baut. Damit ist Training/Einsatz-Konsistenz strukturell
    erzwungen statt nur konventionell zugesichert.
  - kein Endpunkt RAUSCH (status == 2). MEHRDEUTIGE Endpunkte sind
    ZUGELASSEN: ein mehrdeutiges Event sitzt im KERN einer Ueberlappung,
    also genau dort, wo Stufe 1 trennen muss. Seine track_id ist eine
    Schaetzung (naechstes Boxzentrum) -- die beste, die es gibt.
    Rausch bleibt draussen, und der Grund ist nicht Unsicherheit sondern
    ein FALSCHES Label: Rauschevents tragen track_id = -1, und
    tid[i] == tid[j] zaehlte zwei davon als JA.
  - Label 1 = gleiche track_id (JA), 0 = verschieden (NEIN)
  - traegt `jac`, den Anteil gemeinsamer Nachbarn im Fenster. Gerechnet
    auf der UNGEFILTERTEN Kantenmenge, sonst weicht die Nachbarschaft vom
    Einsatz ab.

GESAMMELT WIRD IN DEN FENSTERN DES EINSATZES: anstossende Fenster der
Laenge `neighborhood.alpha`, harter Schnitt. Ein Paar ueber eine
Fenstergrenze kann in Stufe 1 nicht entstehen, also darf es hier nicht
gelernt werden.

BEIDE KLASSEN aus DENSELBEN Fenstern, ausschliesslich aus Overlap-Runs --
saubere Runs liefern physikalisch null NEIN. Fenster ohne NEIN gehoeren
dazu: ihre JA-Paare sind Teil der Einsatzverteilung.

RAM: ein Run liefert bei voller Abdeckung ueber eine Milliarde JA-Paare.
Geduennt wird deshalb IM Fenster (`ja_frac`), NEIN vollstaendig. Am Ende
jedes Runs wird JA auf `ja_vorrat` mal NEIN getrimmt. Die ROHEN
Klassenzahlen laufen mit, sonst waere die Einsatzverteilung danach nicht
mehr rekonstruierbar.
"""
from __future__ import annotations

import resource          # Unix-only; SLURM buchhaltet Speicher hier nicht
import time
import json
from pathlib import Path

import numpy as np

from eventpiv.kern import config
from eventpiv.kern.graph import (jaccard_paare, neighborhood_pairs,
                                 polarity_mask)
from eventpiv.kern.runconfig import PairCfg, identity_meta
from eventpiv.kern.status import STATUS_AMBIG, STATUS_NOISE


FIELDS = ("dt", "dx", "dy", "jac", "label", "run")



def collect_pairs(t, x, y, track_id, status, *, alpha, r_max, geom,
                  ja_frac, rng):
    """Schreitet EINEN Run in anstossenden Fenstern der Laenge alpha ab.

    Harter Schnitt, keine Ueberlappung, keine Erweiterung des rechten
    Randes -- genau die Fenster, die Stufe 1 im Einsatz bildet.

    ja_frac : Anteil der je Fenster gezogenen JA-Paare. NEIN bleibt
        vollstaendig, es ist die knappe Klasse -- mit mehrdeutigen
        Endpunkten gemessen 94:1 (run32) bis 307:1 (run61).
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
    # ambig-Zaehler getrennt: sie beziffern, was dieser Filter
    # hereinlaesst, und ohne sie waere der Zuwachs nach dem Sammeln nicht
    # mehr vom Rest zu unterscheiden.
    zaehler = {"n_roh_ja": 0, "n_roh_nein": 0, "n_fenster": 0,
               "n_fenster_ohne_nein": 0,
               "n_roh_ja_ambig": 0, "n_roh_nein_ambig": 0,
               "n_roh_rausch": 0}
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

        # VOR dem Rauschfilter: solve kennt keinen Status und sieht alle
        # Events als Knoten. Danach gerechnet haette dasselbe Event im
        # Training eine kleinere Nachbarschaft als im Einsatz.
        jac = jaccard_paare(i, j, hi - lo)

        keep = (staw[i] != STATUS_NOISE) & (staw[j] != STATUS_NOISE)
        zaehler["n_roh_rausch"] += int((~keep).sum())
        i, j, jac = i[keep], j[keep], jac[keep]
        if len(i) == 0:
            zaehler["n_fenster_ohne_nein"] += 1
            continue

        # NACH dem Rauschfilter: die Kanten, die es hier noch gibt und die
        # es vorher nicht gab.
        ambig = (staw[i] == STATUS_AMBIG) | (staw[j] == STATUS_AMBIG)

        ja = tidw[i] == tidw[j]
        n_ja, n_nein = int(ja.sum()), int((~ja).sum())
        zaehler["n_roh_ja"] += n_ja
        zaehler["n_roh_nein"] += n_nein
        zaehler["n_roh_ja_ambig"] += int((ja & ambig).sum())
        zaehler["n_roh_nein_ambig"] += int((~ja & ambig).sum())
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
        acc["jac"].append(jac[sel])
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
    roh_ja_ambig = roh_nein_ambig = roh_rausch = 0
    labeling = None

    if verbose:
        print(f"Paare aus {len(ps.runs)} Run(s), Fenster {nb.alpha/1000:.0f}ms, "
              f"ja_frac={ps.ja_frac:g}, ja_vorrat={ps.ja_vorrat:g}:1:")
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
        # VOR dem Deckel zaehlen: sonst waere ein greifender Deckel nicht
        # von der ja_frac-Duennung zu unterscheiden.
        vor_deckel = roh["label"]
        n_gezogen_ja = int((vor_deckel == 1).sum())
        n_gezogen_nein = int((vor_deckel == 0).sum())

        # NEIN zuerst: der JA-Deckel wird daraus abgeleitet.
        n_nein_ziel = (min(n_gezogen_nein, ps.max_no)
                       if ps.max_no is not None else n_gezogen_nein)
        max_ja = int(ps.ja_vorrat * n_nein_ziel)
        d = cap_per_class(roh, max_ja, n_nein_ziel, rng)
        lab = d["label"]
        n_ja, n_nein = int((lab == 1).sum()), int((lab == 0).sum())
        vorrat_erreicht = n_ja >= max_ja
        d["run"] = np.full(len(lab), r, np.int16)
        teile.append(d)

        roh_ja += z["n_roh_ja"]
        roh_nein += z["n_roh_nein"]
        n_fenster += z["n_fenster"]
        roh_ja_ambig += z["n_roh_ja_ambig"]
        roh_nein_ambig += z["n_roh_nein_ambig"]
        roh_rausch += z["n_roh_rausch"]
        # Erst die ROHEN Zahlen sagen, welches Klassenverhaeltnis das Netz
        # im EINSATZ vorfindet -- ohne sie ist der Betriebspunkt nicht
        # rekonstruierbar.
        je_run.append({
            "run": int(r), "n_fenster": z["n_fenster"],
            "n_fenster_ohne_nein": z["n_fenster_ohne_nein"],
            "n_roh_ja": z["n_roh_ja"], "n_roh_nein": z["n_roh_nein"],
            "n_roh_ja_ambig": z["n_roh_ja_ambig"],
            "n_roh_nein_ambig": z["n_roh_nein_ambig"],
            "n_roh_rausch": z["n_roh_rausch"],
            "n_gezogen_ja": n_gezogen_ja, "n_gezogen_nein": n_gezogen_nein,
            "n_ja": n_ja, "n_nein": n_nein,
            "vorrat_ist": (n_ja / n_nein) if n_nein else float("nan"),
            "vorrat_erreicht": bool(vorrat_erreicht),
            "gedeckelt_nein": bool(n_nein < n_gezogen_nein),
            "sec": round(time.time() - tic, 2)})
        z_ = je_run[-1]
        deckel = "".join([
            "" if z_["vorrat_erreicht"] else
            f"  (JA-VORRAT VERFEHLT: {z_['vorrat_ist']:.2f}:1 statt "
            f"{ps.ja_vorrat:g}:1 -- pairs.ja_frac erhoehen)",
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
    gedeckelt = [z["run"] for z in je_run if z["gedeckelt_nein"]]
    vorrat_verfehlt = [z["run"] for z in je_run if not z["vorrat_erreicht"]]

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
        "n_runs_vorrat_verfehlt": len(vorrat_verfehlt),
        "n_roh_ja": roh_ja, "n_roh_nein": roh_nein,
        "n_roh_ja_ambig": roh_ja_ambig,
        "n_roh_nein_ambig": roh_nein_ambig,
        "n_roh_rausch": roh_rausch,
        # Die Kennzahl des Umbaus: welcher Anteil der NEIN-Paare ueberhaupt
        # erst durch mehrdeutige Endpunkte entsteht. Gemessen 0,65 (run32)
        # bis 0,83 (run61).
        "ambig_anteil_nein": (roh_nein_ambig / roh_nein
                              if roh_nein else float("nan")),
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
