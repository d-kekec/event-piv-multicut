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

STRATEGIE gegen Imbalance schon beim Sammeln (BA_NN_Kantenkosten.md §4.1):
  JA   aus SAUBEREN, weit getrennten Runs (dort sind ~alle Paare JA)
  NEIN aus OVERLAP-Runs -- die EINZIGE Quelle; saubere Runs liefern
       physikalisch null NEIN-Paare.
Gespeichert wird ROH (beide Klassen, ungewichtet) plus die Rohgroessen
(dt, dx, dy, ti, xi, yi). Das Balancing passiert erst im Training, damit der
Datensatz neutral bleibt und beide Strategien ohne Neuerzeugung testbar sind.

RAM: volle Nachbarschaft ohne k-NN materialisiert zig Mio Paare auf einmal
-> slab-weise verarbeiten (SLAB_DT-Scheiben, rechter Rand um alpha erweitert,
Paar der Slab des FRUEHEREN Events zugeordnet -> kein Doppelzaehlen).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from eventpiv import config
from eventpiv.graph import neighborhood_pairs
from eventpiv.labels import STATUS_CLEAN

# --- Defaults (Herkunft: BA_NN_Kantenkosten.md §2.1 / §4.1) ---
ALPHA_DEFAULT   = 10_000        # µs
R_MAX_DEFAULT   = 7.06          # px
WIN_DT_DEFAULT  = 40_000        # µs je Sammelfenster
SLAB_DT_DEFAULT = 10_000        # µs Slab-Breite (RAM-Schutz)
WIN_OFFSETS_DEFAULT = (1_500_000, 3_500_000, 5_500_000)   # µs ab Run-Anfang
MAX_PER_RUN_DEFAULT = 2_000_000  # Deckel je Klasse und Run

# Run-Listen: bewusst KEINE Modulkonstanten mehr -- sie gehoeren in die
# Aufrufebene (Skript/Config), nicht ins Rechenmodul. Herkunft der bisherigen
# Listen: identify_overlap_runs.py, BA_Datensatz_und_Pipeline.md §2.2.

FIELDS = ("dt", "dx", "dy", "ti", "xi", "yi", "label", "run")


def collect_pairs(t, x, y, track_id, status, *, alpha, r_max,
                  win_offsets, win_dt, slab_dt, want_label, geom="ellipsoid"):
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

        # Slab-Kanten ab dem ERSTEN Event des Fensters (nicht ab w0) --
        # Verhalten der Vorgaengerversion, bewusst beibehalten.
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


def build_pairs_for_run(split, run_nr, want_label, *, alpha, r_max,
                        win_offsets, win_dt, slab_dt, label_dir=None, geom="ellipsoid"):
    """Laedt die Label-Datei eines Runs und sammelt daraus Paare."""
    path = (config.label_file(split, run_nr) if label_dir is None
            else Path(label_dir) / f"run{run_nr}_eventlabels.npz")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} fehlt -- erst scripts/build_event_labels.py "
            f"--split {split} --runs {run_nr} laufen lassen.")
    d = np.load(path)
    return collect_pairs(d["t"], d["x"], d["y"], d["track_id"], d["status"],
                         alpha=alpha, r_max=r_max, win_offsets=win_offsets,
                         win_dt=win_dt, slab_dt=slab_dt, want_label=want_label, geom=geom)


def build_pair_dataset(clean_runs, overlap_runs, *, split="train",
                       alpha=ALPHA_DEFAULT, r_max=R_MAX_DEFAULT,
                       win_offsets=WIN_OFFSETS_DEFAULT, win_dt_yes = None,
                       win_dt=WIN_DT_DEFAULT, slab_dt=SLAB_DT_DEFAULT,
                       max_yes=MAX_PER_RUN_DEFAULT, max_no=MAX_PER_RUN_DEFAULT,
                       label_dir=None, out_path=None, seed=0, geom="ellipsoid"):
    """Baut den kompletten Paar-Datensatz und schreibt ihn als .npz.

    clean_runs   : JA-Quelle  (weit getrennte Runs)
    overlap_runs : NEIN-Quelle (nur echte Kreuzungen liefern NEIN)

    EIN rng fuer alle Deckel, in fester Reihenfolge (erst clean, dann
    overlap) -> reproduzierbar bei gleichem seed.
    """
    rng = np.random.default_rng(seed)
    t_start = time.time()
    teile = []

    def sammle(runs, want_label, n_max, kind, wdt):
        print(f"{kind}-Paare aus {len(runs)} Run(s) (win_dt={wdt/1000:.0f}ms):")
        for r in runs:
            d = cap_arrays(
                build_pairs_for_run(split, r, want_label, alpha=alpha,
                                    r_max=r_max, win_offsets=win_offsets,
                                    win_dt=wdt, slab_dt=slab_dt, geom=geom,
                                    label_dir=label_dir),
                n_max, rng)
            d["run"] = np.full(len(d["label"]), r, np.int16)
            teile.append(d)
            print(f"  run{r:<3d}: {len(d['label']):>10,} {kind}-Paare")

    # win_dt_for ist auf die NEIN-Ausbeute geeicht. Saubere Runs liefern
    # physikalisch NULL NEIN-Paare -- das breite Fenster kostet dort nur
    # Zeit und RAM, ohne etwas beizutragen.
    sammle(clean_runs, 1, max_yes, "JA", win_dt_yes or WIN_DT_DEFAULT)
    sammle(overlap_runs, 0, max_no, "NEIN", win_dt)

    # Feldweise zusammenfuegen und die Teile SOFORT freigeben. Sonst liegen
    # am Ende alle Teile UND das Ergebnis gleichzeitig im RAM (Faktor 2).
    data = {}
    for k in FIELDS:
        data[k] = np.concatenate([p[k] for p in teile])
        for p in teile:
            del p[k]
    teile.clear()      # Referenzen loesen, Namen behalten (sammle() haelt ihn als Closure)   

    n_yes = int((data["label"] == 1).sum())
    n_no  = int((data["label"] == 0).sum())

    print("=" * 52)
    print(f"Datensatz gesamt: {len(data['label']):,} Paare")
    print(f"  JA  : {n_yes:,}")
    print(f"  NEIN: {n_no:,}")
    if n_no:
        print(f"  Verhaeltnis JA:NEIN = {n_yes/n_no:,.1f} : 1")
    print(f"  Zeit: {time.time()-t_start:.0f}s")

    out_path = pairs_path(geom) if out_path is None else Path(out_path)
        
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, alpha=alpha, r_max=r_max, geom=geom, **data,
             meta=config.meta_json(stage="pair_dataset", split=split,
                                   alpha=alpha, r_max=r_max, geom=geom,
                                   win_dt=win_dt, win_dt_yes=win_dt_yes,
                                   slab_dt=slab_dt,
                                   win_offsets=list(win_offsets),
                                   clean_runs=list(clean_runs),
                                   overlap_runs=list(overlap_runs),
                                   max_yes=max_yes, max_no=max_no, seed=seed,
                                   n_yes=n_yes, n_no=n_no))
    print(f"-> {out_path}")
    return out_path, data

def win_dt_for(alpha, r_max, *, ref_alpha=10_000, ref_r_max=7.06,
               ref_win_dt=40_000, cap=300_000) -> int:
    """Sammelfenster so waehlen, dass die NEIN-Ausbeute konstant bleibt.

    Die Zahl der NEIN-Paare ist proportional zum Nachbarschaftsvolumen,
    also ~ r_max^2 * alpha. Verkleinert man die Nachbarschaft, bricht die
    NEIN-Ausbeute ein -> ueber ein breiteres Sammelfenster kompensieren:

        WIN_DT ~ 1 / (r_max^2 * alpha)

    Geeicht am ELLIPSOID. Zylinder und Hyperboloid haben ein groesseres
    Volumen (Faktor ~1.5 bzw. leicht darueber), liefern also mehr NEIN-Paare
    je Zeit -- das Fenster ist dort konservativ, nicht falsch.

    Nur fuers SAMMELN; das Solve-Fenster bleibt klein.
    """
    v = int(ref_win_dt * (ref_r_max ** 2 * ref_alpha) / (r_max ** 2 * alpha))
    return min(v, cap)

def pairs_path(geom="ellipsoid", out_dir=None):
    """Standardpfad des Paar-Datensatzes -- EINE Namensregel fuer Bauen
    und Laden. Sonst muss man den Datensatz erst erzeugen, um zu wissen,
    wie er heisst."""
    name = ("pairs_dataset.npz" if geom == "ellipsoid"
            else f"pairs_dataset_{geom}.npz")
    return (config.pairs_dir() if out_dir is None else Path(out_dir)) / name