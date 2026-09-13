#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py -- zentrale Aufloesung von Pfaden und Metadaten.

Einzige Stelle im Projekt, die weiss, WO etwas liegt. Kein anderes Modul
enthaelt Hartpfade.

Drei getrennte Wurzeln:
  PROJECT_ROOT  Code/Configs/Docs   -- im Git-Repo
  DATA_ROOT     Rohdaten            -- ausserhalb, read-only, ueber $EVENTPIV_DATA
  OUT_ROOT      erzeugte Artefakte  -- ausserhalb, ueber $EVENTPIV_OUT

Auf dem Cluster werden nur die beiden Umgebungsvariablen anders gesetzt;
der Code bleibt unveraendert.
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from eventpiv.kern.runconfig import slug

# ============================================================
# Wurzeln
# ============================================================


# config.py liegt irgendwo unter src/ im Repo. Aufwaerts nach der Marke
# suchen, statt Ebenen zu zaehlen: parents[N] zeigt nach jedem Umzug einer
# Datei still ins Falsche, und ein falscher PROJECT_ROOT sieht aus wie ein
# leeres Verzeichnis, nicht wie ein Fehler.
def _find_project_root(start: Path) -> Path | None:
    """Repo-Wurzel, oder None ausserhalb des Repos.

    None statt einer Ausnahme, weil PROJECT_ROOT nur zwei Verbraucher hat --
    git_commit() und describe() --, und git_commit laut seinem Docstring nie
    zum Abbruch fuehren darf. Bei einer regulaeren Installation in
    site-packages liegt keine pyproject.toml oberhalb; dann meldet
    git_commit 'unknown', und das ist die richtige Antwort.
    """
    for kandidat in (start, *start.parents):
        if (kandidat / "pyproject.toml").is_file():
            return kandidat
    return None

# Nur fuer configs/ und docs/ gedacht; Daten/Ausgaben NIE hierueber aufloesen.
PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)


def _env_path(var_name: str) -> Path:
    """Pfad aus Umgebungsvariable. '~' wird aufgeloest.

    Kein Fallback aufs Home: der liefert plausibel aussehende, aber falsche
    Pfade (~/train, ~/labels/), und der Lauf tut dann still nichts oder
    schreibt an die falsche Stelle.
    """
    value = os.environ.get(var_name)
    if not value:
        raise RuntimeError(
            f"${var_name} ist nicht gesetzt -- in ~/.bashrc eintragen: "
            f"export {var_name}=...")
    return Path(value).expanduser().resolve()

# Eingabe/Ausgabe von Dateien wird hier angesteuert.
DATA_ROOT = _env_path("EVENTPIV_DATA")   # Ordner MIT train/val/test
OUT_ROOT  = _env_path("EVENTPIV_OUT")    # Ausgabewurzel


VALID_SPLITS = ("train", "val", "test")


def _check_split(split: str) -> str:
    if split not in VALID_SPLITS:
        raise ValueError(
            f"Unbekannter Split {split!r}. Erlaubt: {VALID_SPLITS}. "
            "Run-Nummern kollidieren zwischen Splits -- der Split MUSS stimmen."
        )
    return split


# ============================================================
# Eingaben (DATA_ROOT, read-only)
# ============================================================

def split_dir(split: str) -> Path:
    """Rohdaten eines Splits. DATA_ROOT zeigt auf das Verzeichnis, das
    train/, val/ und test/ enthaelt."""
    return DATA_ROOT / _check_split(split)


def run_files(split: str, run_nr: int, must_exist: bool = True,
              need_bbox: bool = True) -> tuple[Path, Path]:
    """Gibt (events.dat, bbox.npy) fuer einen Run zurueck.

    need_bbox=False fuer Aufrufer, die nur Events brauchen. Echte Messdaten
    haben keine Bounding Boxes; der Solve braucht sie auch nicht, nur die
    Label-Erzeugung und die Bewertung.
    """
    base = split_dir(split) / f"run{run_nr}"
    dat  = base.with_suffix(".dat")
    bbox = base.parent / f"run{run_nr}_bbox.npy"
    if must_exist:
        for p in ((dat, bbox) if need_bbox else (dat,)):
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} fehlt. DATA_ROOT={DATA_ROOT} -- stimmt $EVENTPIV_DATA?"
                )
    return dat, bbox


def available_runs(split: str) -> list[int]:
    """Alle Run-Nummern eines Splits, aufsteigend. Gelesen wird ueber die
    _bbox.npy-Dateien."""
    pattern = re.compile(r"^run(\d+)_bbox\.npy$")
    hits = (pattern.match(p.name) for p in split_dir(split).glob("run*_bbox.npy"))
    return sorted(int(m.group(1)) for m in hits if m)


# ============================================================
# Ausgaben (OUT_ROOT, beschreibbar)
# ============================================================

def _ensure(path: Path, create: bool) -> Path:
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def label_dir(split: str, create: bool = False) -> Path:
    """Event-Labels, split-getrennt (Run-Nummern kollidieren!)."""
    return _ensure(OUT_ROOT / "labels" / _check_split(split), create)


def label_file(split: str, run_nr: int) -> Path:
    return label_dir(split, create=False) / f"run{run_nr}_eventlabels.npz"


def results_dir(create: bool = False) -> Path:
    return _ensure(OUT_ROOT / "results", create)

# ============================================================
# Abgeleitete Artefaktpfade
# ============================================================
# Der Baum IST die Abhaengigkeitskette: eine Ebene je Stufe. Ein Artefakt
# haengt sichtbar unter allem, wovon es abhaengt, und zwei Laeufe mit
# verschiedenen Parametern koennen sich nicht mehr still ueberschreiben.
#
# Labels liegen bewusst DANEBEN, nicht unter wp/: sie haengen von keinem
# dieser Parameter ab.



# create=False ist Default: diese Funktionen BERECHNEN einen Pfad. Sonst
# erzeugte schon das Fragen danach den ganzen Ast, und ein Lauf mit falschen
# Overrides fand ein leeres, selbst angelegtes Verzeichnis vor -- ein
# Konfigurationsfehler, der wie eine Messung aussieht. 
# Nur Schreibstellen setzen create=True.
def wp_dir(nb, create: bool = False) -> Path:
    """Arbeitspunkt = Nachbarschaft. Wurzel aller abgeleiteten Artefakte."""
    return _ensure(OUT_ROOT / "wp" / slug(nb), create)


def pairs_file(nb, ps, create: bool = False) -> Path:
    return _ensure(wp_dir(nb, create) / "pairs" / slug(ps), create) / "pairs.npz"


def model_file(nb, ps, tr, create: bool = False) -> Path:
    d = _ensure(wp_dir(nb, create) / "pairs" / slug(ps) / slug(tr), create)
    return d / "model.pt"


def solve_file(nb, ps, tr, sel, win, sv, create: bool = False) -> Path:
    """Ergebnis EINES Solves. Dateiname aus Laufauswahl, Fenster und Solver --
    die drei Dinge, die NICHT zur Identitaet des Netzes gehoeren."""
    d = _ensure(wp_dir(nb, create) / "pairs" / slug(ps) / slug(tr) / "solves",
                create)
    return d / f"{sel.split}_run{sel.run_nr}_{win.short()}_{sv.short()}.json"


def tracklet_file(nb, ps, tr, sel, sv, create: bool = False) -> Path:
    """Tracklets EINES Runs. Granularitaet Run, nicht Fenster.

    Bei rund 800 Fenstern je Run und 100 Runs waeren fensterweise Artefakte
    80 000 Dateien und ebenso viele W&B-Laeufe. Ein Fenster ist stattdessen
    eine Zeile im Run-Artefakt.

    Keine eigene Slug-Ebene: was zwei Tracklet-Artefakte desselben Netzes
    unterscheidet, ist der SOLVER, und der steht wie bei solve_file im
    Dateinamen. Das Fenster steht NICHT darin -- der Treiber schreitet den
    ganzen Run ab.
    """
    d = _ensure(wp_dir(nb, create) / "pairs" / slug(ps) / slug(tr)
                / "tracklets", create)
    return d / f"{sel.split}_run{sel.run_nr}_{sv.short()}.npz"


def kandidaten_file(nb, ps, tr, sel, sv, kd, create: bool = False) -> Path:
    """Kandidatenpaare und Knotentensor EINES Runs [6a].

    Eigene Slug-Ebene: Reichweite, Mindestgroesse und Stichprobengroesse
    unterscheiden zwei Kandidatensaetze desselben Solves. Der Solver steht
    im DATEINAMEN -- er waehlt die Quelle, nicht die Rechnung.
    """
    d = _ensure(wp_dir(nb, create) / "pairs" / slug(ps) / slug(tr)
                / "tracklets" / "stufe2" / slug(kd), create)
    return d / f"{sel.split}_run{sel.run_nr}_{sv.short()}.npz"


def transformer_file(nb, ps, tr, sv, kd, tf, create: bool = False) -> Path:
    """Das Stufe-2-Netz [6b]. Kein Run im Pfad.

    Ein Netz ueber alle Runs des Kandidatensatzes -- dieselbe Granularitaet
    wie model_file eine Ebene tiefer.

    Der Solver steht im Dateinamen statt in einer Slug-Ebene: zwei Netze
    desselben Kandidatensatzes, eines aus RAMA- und eines aus
    CC-Tracklets, sollen nebeneinander liegen koennen.
    """
    d = _ensure(wp_dir(nb, create) / "pairs" / slug(ps) / slug(tr)
                / "tracklets" / "stufe2" / slug(kd) / slug(tf), create)
    return d / f"model_{sv.short()}.pt"


def solve2_file(nb, ps, tr, sv, kd, tf, sel, sv2, create: bool = False) -> Path:
    """Kennzahlen EINES Stufe-2-Solves [6c] als JSON. Die Partition liegt
    unter demselben Stamm als npz, siehe trajektorien_file.

    Unter dem Netz, das die Kosten geliefert hat -- der Baum IST die
    Abhaengigkeitskette. Laufauswahl und Stufe-2-Solver stehen im
    Dateinamen, wie bei solve_file: sie gehoeren nicht zur Identitaet des
    Netzes.

    BEIDE Solver stehen im Dateinamen. Der von Stufe 1 waehlt die QUELLE
    (Tracklets und Kandidaten tragen ihn ebenfalls im Namen), der von
    Stufe 2 die Rechnung. Sie stehen NEBENEINANDER unter demselben Netz --
    solves/ liegt neben model_<solver>.pt, nicht darunter, und ohne beide
    ueberschrieben sich zwei Solves desselben Runs still.
    """
    d = _ensure(wp_dir(nb, create) / "pairs" / slug(ps) / slug(tr)
                / "tracklets" / "stufe2" / slug(kd) / slug(tf) / "solves",
                create)
    return d / f"{sel.split}_run{sel.run_nr}_{sv.short()}_{sv2.short()}.json"


def trajektorien_file(nb, ps, tr, sv, kd, tf, sel, sv2,
                      create: bool = False) -> Path:
    """Trajektorien-ID je ON-Event [6c]. Neben der Solve-JSON, gleicher Stamm.

    Abgeleitet statt getippt: die beiden Dateien eines Solves duerfen nicht
    auseinanderlaufen koennen.
    """
    return solve2_file(nb, ps, tr, sv, kd, tf, sel, sv2,
                       create).with_suffix(".npz")


def fit_file(nb, ps, tr, sv, kd, tf, sel, sv2, ft,
             create: bool = False) -> Path:
    """Gefittete Bahnen EINES Runs [7]. Neben dem Solve, mit fit_-Praefix.

    Der Fit-Slug steht im Dateinamen und nicht als Ebene: zwei
    Arbeitspunkte des Fits sollen nebeneinander liegen, ohne den Baum zu
    vertiefen -- dieselbe Ueberlegung wie beim Solver.
    """
    p = solve2_file(nb, ps, tr, sv, kd, tf, sel, sv2, create)
    return p.with_name(f"fit_{p.stem}_{slug(ft)}.npz")


# ============================================================
# Metadaten (Code-Prinzip: jede erzeugte Datei traegt ihre Herkunft)
# ============================================================

def git_commit() -> str:
    """Kurzer Commit-Hash des Repos, '+dirty' bei uncommitteten Aenderungen.

    Gibt 'unknown' zurueck, wenn kein Git verfuegbar ist -- darf nie zum
    Abbruch fuehren, Metadaten sind Beiwerk.
    """
    def _git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), *args],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    try:
        commit = _git("rev-parse", "--short", "HEAD")
        if _git("status", "--porcelain"):
            commit += "+dirty"
        return commit
    except Exception:
        return "unknown"


def build_meta(**extra) -> dict:
    """Standard-Metadatenblock; zusaetzliche Felder als Keyword-Argumente.

    Beantwortet: WOHER kommt diese Datei -- wann, aus welchem Commit, auf
    welcher Maschine, in welchem Job. NICHT: wie lief der Lauf; Laufzeit und
    Speicher sind Messungen und gehoeren zu den Kennzahlen der Stufe.

    host und slurm_job_id machen ein Artefakt seinem Log zuordenbar
    (logs/<jobname>-<jobid>.out), sind aber KEINE Identitaet: derselbe Solve
    liefert auf allen drei Maschinen exakt dieselben Zahlen. Wer sie in die
    Identitaet zoege, bekaeme fuer einen Lauf zwei Artefakte.

    slurm_job_id ist None ausserhalb von SLURM -- das ist die Information,
    nicht ein fehlender Wert.
    """
    from eventpiv import __version__
    meta = {
        "created":      datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit":   git_commit(),
        "pkg_version":  __version__,
        "host":         socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    meta.update(extra)
    return meta


def meta_json(**extra) -> str:
    """Metadaten als JSON-String -- so in np.savez(..., meta=meta_json(...)) ablegen.

    Beim Lesen:  json.loads(str(np.load(f)["meta"]))
    (np.load gibt ein 0-dimensionales Array zurueck, daher der str()-Umweg.)
    """
    return json.dumps(build_meta(**extra), ensure_ascii=False)


# ============================================================
# Diagnose
# ============================================================

def describe() -> None:
    """Zeigt die aufgeloesten Pfade -- erster Befehl bei Pfadproblemen,
    besonders auf SLURM."""
    print(f"host         : {socket.gethostname()}"
          f"{'  (SLURM Job ' + os.environ['SLURM_JOB_ID'] + ')' if os.environ.get('SLURM_JOB_ID') else '  (kein SLURM)'}")
    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"DATA_ROOT    : {DATA_ROOT}   (${{EVENTPIV_DATA}}, exists={DATA_ROOT.exists()})")
    print(f"OUT_ROOT     : {OUT_ROOT}   (${{EVENTPIV_OUT}}, exists={OUT_ROOT.exists()})")
    print(f"git_commit   : {git_commit()}")
    for s in VALID_SPLITS:
        d = split_dir(s)
        n = len(available_runs(s)) if d.exists() else 0
        print(f"  split {s:<5}: {n:3d} Runs  ({d})")


if __name__ == "__main__":
    describe()