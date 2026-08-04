# event-piv-multicut

Event-basierte Particle Image Velocimetry: aus den Rohdaten einer
Event-Kamera Partikel-Trajektorien rekonstruieren, formuliert als
**Multicut-Problem** und gelöst mit [RAMA](https://github.com/pawelswoboda/RAMA).

Jedes Event ist ein Knoten. Ein kleines neuronales Netz bewertet für jedes
benachbarte Eventpaar, ob beide zum selben Partikel gehören, und gibt einen
Logit aus — dieser Logit geht **unverändert** als Kantenkosten an den Solver.
Die Cluster der resultierenden Partition sind die Partikel.

> Bachelorarbeit Informatik, Heinrich-Heine-Universität Düsseldorf, betreut
> von Prof. Dr. Paul Swoboda.
> Der Datensatz ist nicht öffentlich und nicht Teil dieses Repositoriums.

---

## Pipeline
dat + _bbox.npy
│
[1] build_event_labels Event → track_id + Qualitätsstatus
│ (Trainingswahrheit, nie NN-Eingabe)
[2] build_pair_dataset Eventpaare aus der (α, r_max)-Nachbarschaft
│ JA aus sauberen Runs, NEIN aus Überlappungen
[3] train_edge_nn Mini-MLP, BCE → Logit = Kantenkosten
│
[4] solve_with_nn Graph → Kosten (GPU) → RAMA → Cluster → Bewertung
│
report_solves mittelt über mehrere Runs

**Es gibt keinen Orchestrator.** Jede Stufe leitet ihren Artefaktpfad aus der
Konfiguration ab — Stufe [3] findet den Datensatz von [2], [4] das Netz von
[3]. Dieselben Overrides an alle Aufrufe genügen.

---

## Installation

Kurzfassung; die vollständige Einrichtung samt RAMA-Build steht in
[`docs/Umgebung_Setup.md`](docs/Umgebung_Setup.md).

```bash
# Python 3.12 in einer venv, PyTorch VOR dem Projekt installieren
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[dev]" --no-build-isolation

# zwei Umgebungsvariablen, ohne die nichts läuft
export EVENTPIV_DATA=/pfad/zum/ordner/mit/train_val_test
export EVENTPIV_OUT=/pfad/fuer/erzeugte/artefakte

python -m eventpiv.config     # zeigt alle aufgelösten Pfade
pytest -q                     # läuft ohne Datensatz und ohne GPU
```

`torch` und `rama_py` stehen bewusst **nicht** in den `dependencies` — beide
brauchen eine bestimmte CUDA-Variante, die pip sonst überschreiben würde.

---

## Verwendung

Es gibt keine `--flags`. Überschrieben wird mit `gruppe.feld=wert`:

```bash
python scripts/build_event_labels.py selection.split=train
python scripts/build_pair_dataset.py
python scripts/train_edge_nn.py
python scripts/solve_with_nn.py selection.split=train selection.run_nr=3
python scripts/report_solves.py report.split=val
```

Ohne Overrides läuft alles am dokumentierten Arbeitspunkt. Vollständige
Parameterliste → [`docs/CLI_Referenz.md`](docs/CLI_Referenz.md).

**Verifikationsanker.** Nach jeder Einrichtung und nach jeder Änderung, die
das Verhalten nicht ändern soll:

```bash
python scripts/solve_with_nn.py selection.split=train selection.run_nr=3 \
    window.t0=1700000 window.dt=20000
```

Erwartet: **12.495.728 Kanten · Reinheit 1.000 · Fragmentierung 1.03**.

Dies ist provisorisch und deckt sehr wenig ab,
gute Integrationstests sind alsbaldig zu entwickeln.
---

## Dokumentation

Die Dateien in `docs/` teilen sich nach ihrer **Lebensdauer**:
Zustandsdokumente beschreiben, wie es *jetzt* ist — datumsfrei, und Falsches
wird ersetzt statt annotiert. Das Journal beschreibt, *wie es dazu kam*, und
wird nie überschrieben.

| Datei | Frage, die sie beantwortet |
|---|---|
| **`Code_Guide.md`** | Wie funktioniert der Code? Mechanik und tragende Entscheidungen, entlang der Pipeline. **Der Einstieg.** |
| **`Code_Karte.md`** | Wo liegt was? Module, Verzeichnisbaum, Aufbau der Slugs und Dateinamen. |
| **`CLI_Referenz.md`** | Welche Skripte und Parameter gibt es? Nachschlagewerk samt Stolperfallen. |
| **`Experimente.md`** | Welche Läufe lohnen sich? Sieben Versuche mit Frage, Erwartung und Aussage. |
| **`Datensatz.md`** | Woraus bestehen die Daten, und welche Messungen begründen die Parameter? |
| **`NN_Kantenkosten.md`** | Warum diese Nachbarschaft, diese Merkmale, dieser Arbeitspunkt? |
| **`RAMA_Anbindung.md`** | Was ist Multicut, wie löst RAMA es, welche Modi gibt es? |
| **`Feldliste_Logging.md`** | Was misst jedes protokollierte Feld, und warum heißt es so? |
| **`Umgebung_Setup.md`** | Wie richtet man Cluster, Python und den RAMA-Build ein? |
| **`Journal.md`** | Datierte Befunde, widerlegte Hypothesen, verworfene Ansätze. |
| **`TODO.md`** | Was steht an, in welcher Reihenfolge, und warum? |

**Für den ersten Überblick:** `Code_Guide.md`, dann `TODO.md`.
**Für eine konkrete Frage:** `CLI_Referenz.md`.
**Für ein „warum eigentlich?":** `Journal.md`.

---

## Referenz

Abbas & Swoboda, *RAMA: A Rapid Multicut Algorithm on GPU*, CVPR 2022.
Der Eventloader `psee_io/` stammt aus der Prophesee-Toolbox (gepatcht für
NumPy 2).