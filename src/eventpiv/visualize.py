#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize.py -- Animationen von Rohevents und Clustern.

Zwei Anwendungsfaelle:

  [A] animate_run()       ein ganzer Run, Events nach Polaritaet eingefaerbt.
                          Zweck: visuell finden, WANN Ueberlappungen und
                          Kreuzungen passieren, um Zeitfenster gezielt zu
                          waehlen. (Portiert aus dem Baseline-Werkzeug.)

  [B] animate_clusters()  ein geloestes Zeitfenster, Panels nebeneinander:
                          Rohevents | RAMA-Cluster | Ground Truth.
                          Zweck: Reinheit und Fragmentierung SEHEN statt nur
                          als Zahl lesen.

FARBWAHL (wichtig fuer die Interpretation!):
  color_by="cluster"  jeder RAMA-Cluster bekommt eine eigene Farbe.
      -> FRAGMENTIERUNG wird sichtbar: ein Partikel in mehreren Farben.
      -> Vergleich mit dem GT-Panel ist muehsam (verschiedene ID-Raeume).
  color_by="gt"       jeder Cluster wird in der Farbe seines MEHRHEITS-
      Partikels gezeichnet.
      -> REINHEITSFEHLER werden sichtbar (zwei GT-Partikel in einer Farbe
         verschmolzen), Fragmentierung dagegen NICHT (die Bruchstuecke
         bekommen dieselbe Farbe).
  Default ist "cluster", weil das die tatsaechliche Partition zeigt.

  Grau = Cluster unterhalb min_size (Rauschen, Singletons) sowie Events ohne
  Clusterzuordnung (RAMA vergibt -1 fuer Knoten ohne Kante) bzw. ohne
  verlaessliches GT-Label (status != 0).

Nur Pillow, kein ffmpeg -- Ausgabe ist ein GIF. Matplotlib wird bewusst
nicht verwendet (WSLg-Probleme, s. BA_Umgebung_Setup.md §3).
"""
from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from eventpiv import config
from eventpiv.data import load_events
from eventpiv.labels import STATUS_CLEAN
from eventpiv.runconfig import AnimClustersCfg, AnimRunCfg, slug

SENSOR_H, SENSOR_W = 720, 1280

BG = np.array([255, 255, 255], np.uint8)      # Hintergrund
GREY = np.array([200, 200, 200], np.uint8)    # Rauschen / keine Zuordnung
ON_COLOR = np.array([220, 40, 40], np.uint8)  # Polaritaet 1
OFF_COLOR = np.array([40, 60, 220], np.uint8)  # Polaritaet 0


# ============================================================
# Farben
# ============================================================

def distinct_colors(n: int) -> np.ndarray:
    """n gut unterscheidbare Farben ueber den Goldenen-Winkel-Trick.

    Aufeinanderfolgende Indizes landen weit auseinander im Farbkreis, so
    dass benachbarte Cluster fast nie aehnliche Farben bekommen.
    """
    out = np.zeros((max(n, 1), 3), np.uint8)
    for k in range(n):
        h = (k * 0.6180339887 + 0.11) % 1.0
        s = 0.55 + 0.30 * ((k % 3) / 2.0)
        v = 0.95 - 0.25 * ((k % 2) / 1.0)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        out[k] = (int(r * 255), int(g * 255), int(b * 255))
    return out


def color_lookup(ids, *, min_size=1, ignore=(-1,)):
    """Baut eine Farbtabelle fuer Label-IDs.

    Gross nach Anzahl sortiert -> die groessten Cluster bekommen die ersten
    (kraeftigsten) Farben. IDs in `ignore` und Gruppen kleiner als min_size
    werden grau.

    Rueckgabe: (uniq, colors) -- Farbe eines Events:
        colors[np.searchsorted(uniq, ids)]
    """
    ids = np.asarray(ids)
    uniq, counts = np.unique(ids, return_counts=True)
    colors = np.tile(GREY, (len(uniq), 1))

    sichtbar = (counts >= min_size) & ~np.isin(uniq, list(ignore))
    idx = np.flatnonzero(sichtbar)
    idx = idx[np.argsort(-counts[idx])]          # groesste zuerst
    colors[idx] = distinct_colors(len(idx))
    return uniq, colors

def majority_gt_colors(labels, track_id, status, *, min_size=1, ignore=(-1,)):
    """Faerbt jeden Cluster in der Farbe seines Mehrheits-GT-Partikels.

    Macht das Cluster-Panel direkt mit dem GT-Panel vergleichbar: gleiche
    Farbe = gleiches Partikel. ACHTUNG: Fragmentierung wird dadurch
    unsichtbar (s. Modul-Docstring).

    Behandelt -1 und kleine Cluster wie color_lookup: Label -1 ist KEIN
    Cluster, sondern das Sammelbecken der kantenlosen Knoten. Ohne diese
    Ausnahme gewinnt dort ein einzelnes sauberes Event die Mehrheits-
    abstimmung und faerbt den gesamten isolierten Rauschanteil ein.
    """
    labels = np.asarray(labels)
    sauber = np.asarray(status) == STATUS_CLEAN

    gt_uniq, gt_colors = color_lookup(np.asarray(track_id)[sauber],
                                      min_size=min_size)
    lab_uniq, lab_counts = np.unique(labels, return_counts=True)
    colors = np.tile(GREY, (len(lab_uniq), 1))

    for k, (c, n) in enumerate(zip(lab_uniq, lab_counts)):
        if c in ignore or n < min_size:
            continue
        m = sauber & (labels == c)
        if not m.any():
            continue
        werte, anzahl = np.unique(np.asarray(track_id)[m], return_counts=True)
        mehrheit = werte[anzahl.argmax()]
        colors[k] = gt_colors[np.searchsorted(gt_uniq, mehrheit)]
    return lab_uniq, colors


# ============================================================
# Zeichnen
# ============================================================

def _paint(canvas, xs, ys, colors, point):
    """Setzt Punkte (optional als point x point Bloecke) auf die Leinwand."""
    h, w = canvas.shape[:2]
    for dy in range(point):
        for dx in range(point):
            yy = np.clip(ys + dy, 0, h - 1)
            xx = np.clip(xs + dx, 0, w - 1)
            canvas[yy, xx] = colors


def render_panel(x, y, colors, mask, *, shape, downscale, point, crop=None):
    """Ein Panel: Events der Maske an ihrer Position, in ihrer Farbe.

    crop: (y0, y1, x0, x1) in BEREITS herunterskalierten Koordinaten.
    """
    h, w = shape
    canvas = np.tile(BG, (h, w, 1))
    if mask.any():
        _paint(canvas, (x[mask] // downscale).astype(np.int64),
               (y[mask] // downscale).astype(np.int64), colors[mask], point)
    if crop is not None:
        y0, y1, x0, x1 = crop
        canvas = canvas[y0:y1, x0:x1]
    return canvas


def compose(panels, titles, stempel=None, *, sep=10, header=16):
    """Panels nebeneinander mit Titelzeile. Rueckgabe: PIL-Image."""
    h = max(p.shape[0] for p in panels)
    w = sum(p.shape[1] for p in panels) + sep * (len(panels) - 1)
    canvas = np.full((h + header, w, 3), 245, np.uint8)

    off = 0
    for p in panels:
        canvas[header:header + p.shape[0], off:off + p.shape[1]] = p
        off += p.shape[1] + sep

    img = Image.fromarray(canvas)
    d = ImageDraw.Draw(img)
    f = ImageFont.load_default()
    off = 0
    for p, t in zip(panels, titles):
        d.text((off + 3, 3), t, fill=(20, 20, 20), font=f)
        off += p.shape[1] + sep
    if stempel:
        d.text((max(w - 78, 4), 3), stempel, fill=(90, 90, 90), font=f)
    return img


def save_gif(frames, path, frame_ms):
    """Speichert die Frames als GIF und meldet Groesse und Pfad."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=max(int(frame_ms), 20), loop=0)
    print(f"-> {path}  ({path.stat().st_size / 1e6:.1f} MB, "
          f"{len(frames)} Frames)")
    return path


def _crop_box(x, y, downscale, shape, rand=12):
    """Zuschnitt auf den tatsaechlich belegten Bildbereich plus Rand."""
    h, w = shape
    xs, ys = x // downscale, y // downscale
    x0 = max(int(xs.min()) - rand, 0)
    x1 = min(int(xs.max()) + rand, w)
    y0 = max(int(ys.min()) - rand, 0)
    y1 = min(int(ys.max()) + rand, h)
    if x1 - x0 < 20 or y1 - y0 < 20:      # zu klein -> lieber ganzes Bild
        return None
    return (y0, y1, x0, x1)


# ============================================================
# [A] ganzer Run
# ============================================================
def animate_run(cfg: AnimRunCfg):
    """Animiert einen ganzen Run (oder ein Teilstueck), Farbe = Polaritaet."""
    sel, win, an = cfg.selection, cfg.window, cfg.anim
    split, run_nr = sel.split, sel.run_nr

    if run_nr is None:
        raise SystemExit("selection.run_nr fehlt -- z.B. selection.run_nr=3")

    t0, dt = win.t0, win.dt
    frame_ms, speedup = an.frame_ms, an.speedup
    downscale, point = an.downscale, an.point

    ev = load_events(split, run_nr, t0=t0, dt=dt)
    if len(ev) == 0:
        raise ValueError("Leeres Zeitfenster.")
    et = ev["t"].astype(np.int64)
    ex, ey, ep = ev["x"], ev["y"], ev["p"]

    t_start = int(et[0]) if t0 is None else int(t0)
    t_ende = int(et[-1]) + 1
    print(f"run{run_nr} ({split}): {len(ev):,} Events in "
          f"[{t_start/1e3:.0f}, {t_ende/1e3:.0f}] ms")

    shape = (SENSOR_H // downscale, SENSOR_W // downscale)
    farben = np.where(ep[:, None] == 1, ON_COLOR, OFF_COLOR).astype(np.uint8)

    schritt = int(frame_ms * 1000 * speedup)
    fenster = int(frame_ms * 1000)
    starts = np.arange(t_start, t_ende, schritt)
    print(f"{len(starts)} Frames à {frame_ms:.1f} ms "
          f"(Speedup {speedup}x, Downscale {downscale}x)")

    frames = []
    for k, s in enumerate(starts):
        m = (et >= s) & (et < s + fenster)
        panel = render_panel(ex, ey, farben, m, shape=shape,
                             downscale=downscale, point=point)
        frames.append(compose([panel], [f"run{run_nr} ({split})  "
                                        f"rot=ON  blau=OFF"],
                              f"{(s - t_start)/1000:.0f} ms"))
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(starts)} Frames")

    out_dir = (config.results_dir() / "animations" if cfg.io.out_path is None
               else Path(cfg.io.out_path))
    name = f"run{run_nr}_{split}_{frame_ms:.0f}ms_{len(frames)}f.gif"
    return save_gif(frames, out_dir / name, frame_ms)


# ============================================================
# [B] geloestes Fenster: Roh | Cluster | GT
# ============================================================

def animate_clusters(cfg: AnimClustersCfg):
    """Loest ein Fenster und animiert Rohevents, Cluster und GT nebeneinander.

    Der Modellpfad wird ABGELEITET wie beim Solve; cfg.io.model_path ist der
    Notausgang. Rueckgabe: (gif_pfad, info) mit den Cluster-Metriken.
    """
    from eventpiv.evaluate import align_gt_window, cluster_metrics
    from eventpiv.model import load_checkpoint
    from eventpiv.solve import check_consistency, solve_window

    sel, win, an = cfg.selection, cfg.window, cfg.anim
    split, run_nr = sel.split, sel.run_nr

    if run_nr is None:
        raise SystemExit("selection.run_nr fehlt -- z.B. selection.run_nr=3")

    if an.color_by not in ("cluster", "gt"):
        raise ValueError("anim.color_by muss 'cluster' oder 'gt' sein")

    if win.t0 is None or win.dt is None:
        raise SystemExit(
            "window.t0/window.dt muessen gesetzt sein -- animate_clusters "
            "LOEST das Fenster. Ganze Runs sind das Ziel (TODO B: k-NN, "
            "ON-only, Tracklets), sprengen aber derzeit den VRAM.")

    mpath = (Path(cfg.io.model_path) if cfg.io.model_path
             else config.model_file(cfg.neighborhood, cfg.pairs, cfg.training,
                                    create=False))
    model, mu, sig, feature_mode, ckpt = load_checkpoint(
        mpath, device=cfg.compute.device)
    check_consistency(ckpt, cfg.neighborhood, strict=cfg.io.strict)

    print(f"run{run_nr} ({split}), Fenster [{win.t0/1000:.0f}ms "
          f"+{win.dt/1000:.1f}ms], Netz {mpath.name}")
    labels, ev, info = solve_window(cfg, model, mu, sig, feature_mode)
    if labels is None:
        raise ValueError("Keine Loesung (leeres Fenster oder keine Kanten).")

    show_gt = an.show_gt

    t = ev["t"]
    x = ev["x"].astype(np.int64)
    y = ev["y"].astype(np.int64)

    # --- GT laden, falls vorhanden ---
    tid = sta = None
    gt_pfad = (config.label_file(split, run_nr) if cfg.io.label_dir is None
               else Path(cfg.io.label_dir) / f"run{run_nr}_eventlabels.npz")
    if gt_pfad.exists():
        gt = np.load(gt_pfad)
        sl = align_gt_window(gt["t"], t)
        tid, sta = gt["track_id"][sl], gt["status"][sl]
        info.update(cluster_metrics(labels, tid, sta))
        print(f"  Reinheit {info['purity_mean']:.3f}  "
              f"Fragmentierung {info['frag_mean']:.2f}  "
              f"({info['n_clusters']} Cluster / {info['n_gt']} GT-Partikel)")
    else:
        print(f"  keine GT-Datei ({gt_pfad.name}) -> nur Roh + Cluster")
        show_gt = False

    # --- Farbtabellen (einmal fuer das ganze Fenster -> stabil ueber Frames) ---
    if an.color_by == "gt" and tid is not None:
        lab_uniq, lab_colors = majority_gt_colors(labels, tid, sta,
                                                  min_size=an.min_size)
    else:
        lab_uniq, lab_colors = color_lookup(labels, min_size=an.min_size)
    cluster_farben = lab_colors[np.searchsorted(lab_uniq, labels)]

    roh_farben = np.tile(np.array([70, 70, 70], np.uint8), (len(t), 1))

    gt_farben = None
    if show_gt and tid is not None:
        gt_uniq, gt_colors = color_lookup(np.where(sta == STATUS_CLEAN,
                                                   tid, -1), min_size=1)
        gt_farben = gt_colors[np.searchsorted(
            gt_uniq, np.where(sta == STATUS_CLEAN, tid, -1))]

    # --- Frames ---
    shape = (SENSOR_H // an.downscale, SENSOR_W // an.downscale)
    zuschnitt = _crop_box(x, y, an.downscale, shape) if an.crop else None

    fenster = an.frame_ms * 1000.0
    starts = np.arange(t[0], t[-1] + fenster, fenster)
    print(f"{len(starts)} Frames à {an.frame_ms:.2f} ms  "
          f"(trail={'an' if an.trail else 'aus'}, Farbe nach "
          f"{'GT-Mehrheit' if an.color_by == 'gt' else 'Cluster-ID'})")

    frames = []
    for s in starts:
        m = (t < s + fenster) if an.trail else ((t >= s) & (t < s + fenster))
        gem = dict(shape=shape, downscale=an.downscale, point=an.point,
                   crop=zuschnitt)
        panels = [render_panel(x, y, roh_farben, m, **gem),
                  render_panel(x, y, cluster_farben, m, **gem)]
        titel = ["Rohevents", f"RAMA-Cluster (grau: < {an.min_size} Events)"]
        if show_gt:
            panels.append(render_panel(x, y, gt_farben, m, **gem))
            titel.append("Ground Truth (grau: kein sauberes Label)")
        frames.append(compose(panels, titel, f"{(s - t[0])/1000:.2f} ms"))

   
    out_dir = (config.results_dir() / "animations" if cfg.io.out_path is None
               else Path(cfg.io.out_path))
    name = (f"cluster_{split}_run{run_nr}_{slug(cfg.neighborhood)}_"
            f"{cfg.training.short()}_{an.color_by}.gif")
    pfad = save_gif(frames, out_dir / name, max(an.frame_ms * 20, 60))
    return pfad, info
