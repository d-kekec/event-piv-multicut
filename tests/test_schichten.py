#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Die Schichtung ist eine PRUEFBARE Aussage, kein Kommentar.

Erlaubte Richtung:

    kern       -> kern
    bewerten   -> bewerten, kern
    kette      -> kette, bewerten, kern
    zeigen     -> alles

Warum ueberhaupt ein Test: die Ordner allein halten nichts. Ein
`from eventpiv.kette.labels import ...` in bewerten/ laeuft anstandslos, und
erst Monate spaeter faellt auf, dass die Bewertung von dem abhaengt, was sie
bewertet. Genau diese Kante gab es (evaluate -> labels.STATUS_*), und sie
ist nicht durch Aufmerksamkeit aufgefallen, sondern beim Vermessen der
Importkanten.

GELESEN WIRD DER QUELLTEXT, nicht der Importgraph zur Laufzeit: der
Syntaxbaum erfasst auch die Importe in Funktionsruempfen, von denen es hier
etliche gibt (visualize, tracklets, solve), und er braucht weder torch noch
RAMA noch einen Datensatz.

Die Zuordnung Modul -> Schicht kommt aus dem VERZEICHNISBAUM. Ein neues
Modul in kette/ kostet damit keine Testaenderung -- nur eine echte
Verletzung schlaegt an.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from eventpiv.kern import config

PAKET = config.PROJECT_ROOT / "src" / "eventpiv"

# Was eine Schicht importieren darf. psee_io ist Fremdcode und steht
# ausserhalb der Ordnung: jede Schicht darf ihn benutzen, er selbst
# importiert nichts aus eventpiv.
# Jede Schicht steht in ihrer eigenen Menge: INNERHALB einer Schicht ist
# jede Kante erlaubt. Eingeschraenkt wird die Richtung ZWISCHEN Schichten --
# solve -> fenster oder train -> model sind normale Kanten der Kette.
ERLAUBT = {
    "kern":     {"kern", "psee_io"},
    "bewerten": {"bewerten", "kern", "psee_io"},
    "kette":    {"kette", "bewerten", "kern", "psee_io"},
    "zeigen":   {"zeigen", "kette", "bewerten", "kern", "psee_io"},
}


def _schicht(pfad: Path) -> str | None:
    """Erste Ebene unter src/eventpiv/, oder None fuer __init__.py."""
    rel = pfad.relative_to(PAKET).parts
    return rel[0] if len(rel) > 1 else None


def _importierte_schichten(pfad: Path):
    """(Schicht, Zeilennummer, Modulpfad) je eventpiv-Import in der Datei.

    Erfasst beide Formen -- `import eventpiv.kern.config` und
    `from eventpiv.kern import config` -- und beide Orte, Modulebene wie
    Funktionsrumpf, weil ast.walk den ganzen Baum abschreitet.

    `from eventpiv import __version__` traegt keine Schicht und faellt
    heraus: das Wurzelpaket gehoert zu keiner.
    """
    baum = ast.parse(pfad.read_text(encoding="utf-8"))
    for knoten in ast.walk(baum):
        if isinstance(knoten, ast.Import):
            namen = [a.name for a in knoten.names]
        elif isinstance(knoten, ast.ImportFrom) and knoten.level == 0:
            namen = [knoten.module or ""]
        else:
            continue
        for name in namen:
            teile = name.split(".")
            if teile[0] != "eventpiv" or len(teile) < 2:
                continue
            yield teile[1], knoten.lineno, name


def _module():
    """Alle Module, die einer Schicht angehoeren.

    psee_io faellt heraus: Fremdcode, keine Schicht, und ohne diese
    Ausnahme liefe der Zugriff auf ERLAUBT in einen KeyError statt in eine
    lesbare Meldung.
    """
    return sorted(p for p in PAKET.rglob("*.py")
                  if _schicht(p) in ERLAUBT
                  and "__pycache__" not in p.parts)


def test_paket_ist_geschichtet():
    """Die vier Schichten existieren und sind Pakete."""
    for name in ERLAUBT:
        assert (PAKET / name / "__init__.py").is_file(), \
            f"{name}/ fehlt oder hat keine __init__.py -- find_packages " \
            "wuerde die Module beim Installieren uebergehen"


def test_keine_module_direkt_im_paket():
    """src/eventpiv/ selbst traegt keine Module mehr, nur __init__.py.

    Ein Modul ohne Schicht waere nicht pruefbar: die Regeln unten greifen
    ueber das Verzeichnis.
    """
    lose = [p.name for p in PAKET.glob("*.py") if p.name != "__init__.py"]
    assert lose == [], f"ohne Schicht: {lose}"


@pytest.mark.parametrize("pfad", _module(), ids=lambda p: str(p.name))
def test_import_verletzt_die_schichtung_nicht(pfad):
    verletzungen = [
        f"{pfad.relative_to(PAKET)}:{zeile} -> {modul}"
        for ziel, zeile, modul in _importierte_schichten(pfad)
        if ziel not in ERLAUBT[_schicht(pfad)]
    ]
    assert not verletzungen, (
        f"Schicht '{_schicht(pfad)}' darf nur aus "
        f"{sorted(ERLAUBT[_schicht(pfad)])} importieren:\n  "
        + "\n  ".join(verletzungen))