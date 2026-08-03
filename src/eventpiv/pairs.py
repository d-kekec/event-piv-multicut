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

STRATEGIE gegen Imbalance schon beim Sammeln:
  JA   aus SAUBEREN, weit getrennten Runs (dort sind ~alle Paare JA)
  NEIN aus OVERLAP-Runs -- die EINZIGE Quelle; weit getrennte Partikel
       erzeugen zwischen sich ueberhaupt keine Paare, also PHYSIKALISCH
       null NEIN.
Gespeichert wird ROH (beide Klassen, ungewichtet) plus die Rohgroessen
(dt, dx, dy, ti, xi, yi). Das Balancing passiert erst im Training, damit der
Datensatz neutral bleibt und beide Strategien ohne Neuerzeugung testbar sind.

RAM: die volle Nachbarschaft ohne k-NN materialisiert zig Millionen Paare
auf einmal -> slab-weise verarbeiten (Scheiben der Breite slab_dt, rechter
Rand um alpha erweitert, Paar der Slab des FRUEHEREN Events zugeordnet ->
kein Doppelzaehlen).
"""
from __future__ import annotations

import resource          # Unix-only; SLURM buchhaltet Speicher hier nicht
import time
from pathlib import Path

import numpy as np

from eventpiv import config
from eventpiv.graph import neighborhood_pairs
from eventpiv.labels import STATUS_CLEAN
from eventpiv.runconfig import PairCfg, identity_meta

# --- Defaults liegen in runconfig.PairSampling / .Neighborhood ---
# Frueher standen sie zusaetzlich hier. Zwei Wahrheiten ueber dieselbe Zahl
# fuehrten dazu, dass train.py den Datensatzpfad ohne geom neu baute und
# still den falschen Datensatz lud.

FIELDS = ("dt", "dx", "dy", "ti", "xi", "yi", "label", "run")



def collect_pairs(t, x, y, track_id, status, *, alpha, r_max,
                  win_offsets, win_dt, slab_dt, want_label, geom):
    """Sammelt Paare aus EINEM Run. Reine Funktion -- Arrays rein, Arrays raus.

    t, x, y      : Eventkoordinaten (float64)
    track_id     : GT je Event, -1 wo nicht sauber
    status       : 0/1/2 (labels.STATUS_*)
    want_label   : 1 = nur JA behalten, 0 = nur NEIN, None = beide

    Rueckgabe: dict mit dt, dx, dy, ti, xi, yi, label (ohne 'run').
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(t) > 1 and not (np.diff(t) >= 0).all():
        # t[0] und die Slab-Kanten ab tw[0] verankern Fenster und Raster --
        # bei unsortiertem t fielen alle Events davor in gar keine Slab.
        # neighborhood_pairs braucht die Sortierung NICHT, deshalb faellt es
        # sonst nirgends auf.
        raise ValueError("collect_pairs erwartet aufsteigend sortierte t")

    offs = sorted(win_offsets)
    if any(b - a < win_dt for a, b in zip(offs, offs[1:])):
        # Ueberlappende Fenster sammelten dieselben Paare zweimal. Die
        # Slab-Zuordnung schuetzt nur INNERHALB eines Fensters.
        raise ValueError(
            f"win_offsets ueberlappen bei win_dt={win_dt}: {offs}. "
            "Ueberlappende Fenster sammeln Paare doppelt.")

    acc = {k: [] for k in FIELDS if k != "run"}
    if len(t) == 0:
        return {k: np.empty(0, np.float32) for k in acc} | {
            "label": np.empty(0, np.int8)}

    t_run0 = t[0]
    for off in win_offsets:
        w0 = t_run0 + off
        win = (t >= w0) & (t < w0 + win_dt)
        if not win.any():
            continue
        tw, xw, yw = t[win], x[win], y[win]
        tidw, staw = track_id[win], status[win]

        # Slab-Raster ab dem ERSTEN Event des Fensters, nicht ab w0: sonst
        # laege die erste Slab teilweise vor allen Daten.
        edges = np.arange(tw[0], tw[0] + win_dt + slab_dt, slab_dt)
        for a in edges:
            if not ((tw >= a) & (tw < a + slab_dt)).any():
                continue
            # Slab + rechter Rand alpha: Partner eines Kern-Events koennen
            # bis zu alpha spaeter liegen.
            ext = np.flatnonzero((tw >= a) & (tw < a + slab_dt + alpha))
            ts, xs, ys = tw[ext], xw[ext], yw[ext]
            tids, stas = tidw[ext], staw[ext]

            i, j = neighborhood_pairs(ts, xs, ys, alpha, r_max, geom=geom)
            if len(i) == 0:
                continue

            # Paar der Slab zuordnen, in der das FRUEHERE Event liegt
            # (graph garantiert t[i] <= t[j]) -> jedes Paar genau einmal.
            keep = (ts[i] >= a) & (ts[i] < a + slab_dt)
            # beide Events sauber -> verlaessliches Label
            keep &= (stas[i] == STATUS_CLEAN) & (stas[j] == STATUS_CLEAN)
            i, j = i[keep], j[keep]
            if len(i) == 0:
                continue

            lab = (tids[i] == tids[j]).astype(np.int8)
            if want_label is not None:
                sel = lab == want_label
                if not sel.any():
                    continue
                i, j, lab = i[sel], j[sel], lab[sel]

            acc["dt"].append((ts[j] - ts[i]).astype(np.float32))   # >= 0
            acc["dx"].append((xs[j] - xs[i]).astype(np.float32))
            acc["dy"].append((ys[j] - ys[i]).astype(np.float32))
            # Absolutkoordinaten des FRUEHEREN Events (fuer raw6-Features);
            # das spaetere ergibt sich als (ti+dt, xi+dx, yi+dy).
            acc["ti"].append(ts[i].astype(np.float32))
            acc["xi"].append(xs[i].astype(np.float32))
            acc["yi"].append(ys[i].astype(np.float32))
            acc["label"].append(lab)

    if not acc["dt"]:
        out = {k: np.empty(0, np.float32) for k in acc}
        out["label"] = np.empty(0, np.int8)
        return out
    return {k: np.concatenate(v) for k, v in acc.items()}


def cap_arrays(data: dict, n_max, rng) -> dict:
    """Zufaellige Teilstichprobe auf n_max Paare (alle Felder gleich indiziert)."""
    n = len(data["label"])
    if n_max is None or n <= n_max:
        return data
    sel = rng.choice(n, n_max, replace=False)
    return {k: v[sel] for k, v in data.items()}


def build_pairs_for_run(cfg: PairCfg, run_nr, want_label, win_dt):
    """Laedt die Label-Datei eines Runs und sammelt daraus Paare.

    win_dt explizit, weil die beiden Klassen verschiedene Sammelfenster
    haben (cfg.pairs.win_dt_yes bzw. .win_dt) -- der Aufrufer entscheidet.
    """
    nb, ps = cfg.neighborhood, cfg.pairs
    path = (Path(cfg.io.label_dir) / f"run{run_nr}_eventlabels.npz"
            if cfg.io.label_dir else config.label_file(ps.split, run_nr))
    if not path.exists():
        raise FileNotFoundError(
            f"{path} fehlt -- erst scripts/build_event_labels.py "
            f"selection.split={ps.split} selection.runs=[{run_nr}] laufen lassen.")
    with np.load(path) as d:
        return collect_pairs(d["t"], d["x"], d["y"], d["track_id"], d["status"],
                             alpha=nb.alpha, r_max=nb.r_max, geom=nb.geom,
                             win_offsets=tuple(ps.win_offsets), win_dt=win_dt,
                             slab_dt=ps.slab_dt, want_label=want_label)


def build_pair_dataset(cfg: PairCfg, verbose=True, on_summary=None):
    """Baut den kompletten Paar-Datensatz und schreibt ihn als .npz.

    on_summary : optional Callable, bekommt EINMAL am Ende die Kennzahlen
        samt Ausbeute je Run. Nur eine Senke, nicht zwei wie im Training:
        die Ausbeuten sind eine TABELLE, keine Reihe -- sie haben keine
        Ordnung in der Zeit, nur je Run.

    Der Pfad wird ABGELEITET, cfg.io.out_path ist der Notausgang. Metadaten
    kommen aus identity_meta statt aus einer handgepflegten Feldliste, die
    schon einmal ein Feld vergessen hat.

    EIN rng fuer alle Deckel, in fester Reihenfolge (erst clean, dann
    overlap) -> reproduzierbar bei gleichem seed.
    """
    nb, ps = cfg.neighborhood, cfg.pairs
    if ps.win_dt is None or ps.win_dt_yes is None:
        raise ValueError(
            "win_dt/win_dt_yes nicht aufgeloest. cfg.resolve_derived() laeuft "
            "normalerweise in scripts/_cli.to_cfg -- bei direktem Aufruf aus "
            "Code oder Test vorher selbst aufrufen.")

    rng = np.random.default_rng(ps.seed)
    t_start = time.time()
    teile = []
    je_run = []

    def sammle(runs, want_label, n_max, kind, wdt):
        print(f"{kind}-Paare aus {len(runs)} Run(s) (win_dt={wdt/1000:.0f}ms):")
        for r in runs:
            tic = time.time()
            roh = build_pairs_for_run(cfg, r, want_label, wdt)
            n_roh = int(len(roh["label"]))
            d = cap_arrays(roh, n_max, rng)
            n_behalten = int(len(d["label"]))
            d["run"] = np.full(n_behalten, r, np.int16)
            teile.append(d)
            # ROH und BEHALTEN getrennt: cap_arrays schneidet still ab. Erst
            # die Differenz sagt, ob das Klassenverhaeltnis eine Eigenschaft
            # der DATEN ist oder eine Folge von max_yes/max_no.
            je_run.append({"run": int(r), "kind": kind, "n_roh": n_roh,
                           "n_behalten": n_behalten,
                           "gedeckelt": bool(n_max is not None
                                             and n_roh > n_max),
                           "sec": round(time.time() - tic, 2)})
            deckel = "  (gedeckelt)" if je_run[-1]["gedeckelt"] else ""
            print(f"  run{r:<3d}: {n_behalten:>10,} {kind}-Paare "
                  f"von {n_roh:,} roh{deckel}")
            
    # Das breite Fenster ist auf die NEIN-Ausbeute geeicht. Saubere Runs
    # liefern physikalisch NULL NEIN-Paare -- dort kostet es nur Zeit.
    sammle(ps.clean_runs, 1, ps.max_yes, "JA", ps.win_dt_yes)
    sammle(ps.overlap_runs, 0, ps.max_no, "NEIN", ps.win_dt)

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

    leer = [z["run"] for z in je_run if z["n_behalten"] == 0]
    gedeckelt = [z["run"] for z in je_run if z["gedeckelt"]]

    print("=" * 52)
    print(f"Datensatz gesamt: {len(data['label']):,} Paare")
    print(f"  JA  : {n_yes:,}")
    print(f"  NEIN: {n_no:,}")
    if n_no:
        print(f"  Verhaeltnis JA:NEIN = {n_yes/n_no:,.1f} : 1")
    if gedeckelt:
        print(f"  GEDECKELT: {len(gedeckelt)} Run(s) {gedeckelt} -- das "
              f"Verhaeltnis folgt hier aus max_yes/max_no, nicht aus den Daten")
    if leer:
        print(f"  OHNE AUSBEUTE: {len(leer)} Run(s) {leer}")
    print(f"  Zeit: {time.time()-t_start:.0f}s")

    out_path = (Path(cfg.io.out_path) if cfg.io.out_path
                else config.pairs_file(nb, ps))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **data,
             meta=config.meta_json(stage="pair_dataset",
                                   **identity_meta(neighborhood=nb, pairs=ps),
                                   n_yes=n_yes, n_no=n_no))
    kennzahlen = {
        "n_total": int(len(data["label"])), "n_ja": n_yes, "n_nein": n_no,
        "ratio_ja_nein": (n_yes / n_no) if n_no else float("nan"),
        "n_runs": len(je_run),
        "n_runs_leer": len(leer),
        "n_runs_gedeckelt": len(gedeckelt),
        "win_dt": ps.win_dt, "win_dt_yes": ps.win_dt_yes,
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
    print(f"-> {out_path}")
    return out_path, data
