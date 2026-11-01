#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
konvertiere_raw.py -- EVT3-.raw nach Prophesee-DAT.

Kein Weg fuehrt hier an einer eigenen Bit-Packung vorbei: evt3 liest EVT3
samt externem Trigger, schreibt aber nur eigene Formate. Die Packung ist
deshalb DOPPELT im Projekt (auch in tests/mini_dataset.py) und wird hier
gegen PSEELoader zurueckgeprueft -- ein vertauschtes Bit ergaebe sonst
eine lesbare Datei mit falschen Koordinaten.

Geschrieben wird STUECKWEISE: tobytes() ueber den ganzen Strom legte neben
den dekodierten Strom eine Kopie mit 8 Byte je Event.

Die Run-Nummer kommt aus der ersten Ziffernfolge im Dateinamen -- aus
<name>_048ms.raw wird run48.dat.

DAT-FORMAT (psee_io/dat_events_tools.py):
  Kopf      Zeilen "% ...", danach zwei Bytes ev_type=0 und ev_size=8
  Event     4 B Zeitstempel u4 + 4 B Nutzlast i4
  Nutzlast  Bit 0-13 x, Bit 14-27 y, Bit 28 Polaritaet

Aufruf ohne Hydra und ohne EVENTPIV_*. <rohdaten> ist eine .raw-Datei oder
ein Verzeichnis, dessen *.raw alle umgewandelt werden; <ziel> wird
angelegt. rebase=1, die Vorgabe, zieht die Zeit des ersten ausgewaehlten
Events ab und schreibt den Versatz in den Kopf. roi in px schneidet aus,
ohne die Koordinaten zu verschieben. t0 und dt in ms ab dem ersten Event
der Datei; dt wirkt nur zusammen mit t0.
  python tools/konvertiere_raw.py <rohdaten> <ziel> [rebase=0|1]
         [roi=x0,y0,breite,hoehe] [t0=ms] [dt=ms]
  python tools/konvertiere_raw.py <rohdaten> <ziel> t0=1000 dt=500
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import evt3
import numpy as np

from eventpiv.psee_io.psee_loader import PSEELoader

BREITE, HOEHE = 1280, 720
BLOCK = 10_000_000          # Events je Schreibblock
_EV_DTYPE = np.dtype([("t", "<u4"), ("_", "<i4")])


def _kopf(breite: int, hoehe: int, versatz_us: int) -> bytes:
    """Kopfzeilen plus ev_type/ev_size.

    Der Zeitversatz steht als Kommentarzeile darin: nach dem Rebasen ist
    die Kamerazeit sonst verloren, und der Triggerkanal liegt in ihr.
    """
    return (b"% Data file containing Event2D events.\n"
            b"% Version 2\n"
            + f"% Height {hoehe}\n% Width {breite}\n".encode()
            + f"% t_offset_us {versatz_us}\n".encode()
            + bytes([0, 8]))


def schreibe_dat_blockweise(pfad, t, x, y, p, breite, hoehe, versatz_us):
    """Wie tests/mini_dataset.schreibe_dat, aber ohne die Vollkopie.

    Auch die Rundung liegt IM Block: xi und yi als int64 ueber den ganzen
    Strom legten 16 Byte je Event zusaetzlich an.
    """
    with open(pfad, "wb") as f:
        f.write(_kopf(breite, hoehe, versatz_us))
        for a in range(0, len(t), BLOCK):
            b = min(a + BLOCK, len(t))
            xi = np.clip(np.rint(x[a:b]), 0, breite - 1).astype(np.int64)
            yi = np.clip(np.rint(y[a:b]), 0, hoehe - 1).astype(np.int64)
            ev = np.empty(b - a, dtype=_EV_DTYPE)
            ev["t"] = np.asarray(t[a:b], dtype=np.uint32)
            ev["_"] = (xi | (yi << 14)
                       | (np.asarray(p[a:b], np.int64) << 28)).astype(np.int32)
            f.write(ev.tobytes())


def pruefe(pfad, t, x, y, p) -> None:
    """Rundreise durch PSEELoader. Schlaegt sie fehl, ist die Packung falsch.

    Nur die ersten und letzten BLOCK Events, sonst laege die Datei ein
    zweites Mal im Speicher.
    """
    ld = PSEELoader(str(pfad))
    n = ld.event_count()
    if n != len(t):
        raise SystemExit(f"PRUEFUNG: {n} Events gelesen, {len(t)} geschrieben")
    ev = ld.load_n_events(min(BLOCK, n))
    k = len(ev)
    for name, soll in (("t", np.asarray(t[:k], np.int64)),
                       ("x", np.clip(np.rint(x[:k]), 0, BREITE - 1)),
                       ("y", np.clip(np.rint(y[:k]), 0, HOEHE - 1)),
                       ("p", np.asarray(p[:k], np.int64))):
        ist = ev[name].astype(np.int64)
        if not np.array_equal(ist, soll.astype(np.int64)):
            schlecht = int(np.flatnonzero(ist != soll.astype(np.int64))[0])
            raise SystemExit(
                f"PRUEFUNG fehlgeschlagen bei Feld {name!r}, Index "
                f"{schlecht}: gelesen {ist[schlecht]}, "
                f"geschrieben {soll[schlecht]}")
    print(f"    Rundreise geprueft ueber {k:,} Events")


def konvertiere(quelle: Path, ziel: Path, rebase: bool, roi, t0_ms, dt_ms):
    nr = re.search(r"(\d+)", quelle.stem)
    if nr is None:
        raise SystemExit(f"keine Ziffern im Namen: {quelle.name}")
    aus = ziel / f"run{int(nr.group(1))}.dat"

    ev, _ = evt3.decode_file_with_triggers(str(quelle))
    d = ev.to_dict()
    t = np.asarray(d["timestamp"])
    x = np.asarray(d["x"])
    y = np.asarray(d["y"])
    p = np.asarray(d["polarity"])

    m = np.ones(t.size, bool)
    if t0_ms is not None:
        lo = int(t[0] + t0_ms * 1000)
        m &= (t >= lo) & (t < lo + int((dt_ms or 0) * 1000)) if dt_ms else \
            (t >= lo)
    if roi is not None:
        x0, y0, b, h = roi
        m &= (x >= x0) & (x < x0 + b) & (y >= y0) & (y < y0 + h)
    if not m.all():                   # sonst kopiert die Maske alles zweimal
        t, x, y, p = t[m], x[m], y[m], p[m]
    if t.size == 0:
        raise SystemExit(f"{quelle.name}: Auswahl ist leer")

    versatz = int(t[0]) if rebase else 0
    t = t.copy() if m.all() else t     # nie in das evt3-Array schreiben
    t -= versatz                       # in place: keine Kopie der Zeitstempel
    if t[-1] > np.iinfo(np.uint32).max:
        raise SystemExit(f"{quelle.name}: t={t[-1]} sprengt uint32")

    print(f"  {quelle.name} -> {aus.name}: {t.size:,} Events, "
          f"{t[-1] / 1000:.1f} ms, Versatz {versatz} us")
    schreibe_dat_blockweise(aus, t, x, y, p, BREITE, HOEHE, versatz)
    pruefe(aus, t, x, y, p)


def main() -> None:
    quelle = Path(sys.argv[1]).expanduser()
    ziel = Path(sys.argv[2]).expanduser()
    opt = dict(kv.split("=", 1) for kv in sys.argv[3:] if "=" in kv)
    roi = [int(v) for v in opt["roi"].split(",")] if "roi" in opt else None
    t0 = float(opt["t0"]) if "t0" in opt else None
    dt = float(opt["dt"]) if "dt" in opt else None
    rebase = opt.get("rebase", "1") == "1"

    ziel.mkdir(parents=True, exist_ok=True)
    dateien = sorted(quelle.glob("*.raw")) if quelle.is_dir() else [quelle]
    print(f"{len(dateien)} Datei(en) -> {ziel}  (rebase={rebase}, roi={roi})")
    for f in dateien:
        konvertiere(f, ziel, rebase, roi, t0, dt)


if __name__ == "__main__":
    main()