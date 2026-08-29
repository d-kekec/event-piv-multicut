# event-piv-multicut

Event-basierte Particle Image Velocimetry: aus den Rohdaten einer
Event-Kamera Partikel-Trajektorien rekonstruieren, formuliert als
**Multicut-Problem** und gelöst mit [RAMA](https://github.com/pawelswoboda/RAMA).

**Zwei Stufen, beide Multicut.** In Stufe 1 ist jedes Event ein Knoten: ein
kleines neuronales Netz bewertet benachbarte Eventpaare und gibt einen Logit
aus, der **unverändert** als Kantenkosten an den Solver geht. Gelöst wird
fensterweise, das Ergebnis sind kurze Bahnstücke — Tracklets.

In Stufe 2 ist jedes Tracklet ein Knoten. Ein Paar-Transformer bewertet, ob
zwei Tracklets zum selben Partikel gehören, und gelöst wird **einmal über den
ganzen Run**. Die Cluster dieser Partition sind die Trajektorien.

> Bachelorarbeit Informatik, Heinrich-Heine-Universität Düsseldorf, betreut
> von Prof. Dr. Paul Swoboda.
> Der Datensatz ist nicht öffentlich und nicht Teil dieses Repositoriums.

---

## Pipeline
dat + _bbox.npy
│
[1] build_event_labels   Event → track_id + Qualitätsstatus
│                        (Trainingswahrheit, nie NN-Eingabe)
[2] build_pair_dataset   Eventpaare aus der (α, r_max)-Nachbarschaft
│                        beide Klassen aus denselben Fenstern
[3] train_edge_nn        Mini-MLP, BCE → Logit = Kantenkosten
│
├─ [4]  solve_with_nn    EIN Fenster: Graph → Kosten (GPU) → RAMA → Bewertung
│
└─ [5a] build_tracklets  ganzer Run in Fenstern der Länge α → Artefakt (A)
   │
   ├─ [5b] build_features    ein Tracklet, eine Zeile → Artefakt (B)
   │
   └─ [6a] build_kandidaten  Tracklets werden Knoten: Filter, Zentroid,
      │                      Eventstichprobe, Kandidatenpaare
      [6b] train_stufe2      Paar-Transformer → Logit = Kantenkosten
      │
      [6c] solve_stufe2      EIN Multicut über den ganzen Run → Trajektorien

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

python -m eventpiv.kern.config     # zeigt alle aufgelösten Pfade
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
python scripts/build_tracklets.py selection.split=train selection.runs=[32]
python scripts/build_features.py  selection.split=train selection.runs=[32]
python scripts/build_kandidaten.py selection.split=train selection.runs=[32]
python scripts/train_stufe2.py
python scripts/solve_stufe2.py    selection.split=val   selection.runs=[4]
```

Ohne Overrides läuft alles am dokumentierten Arbeitspunkt. Vollständige
Parameterliste → [`docs/CLI_Referenz.md`](docs/CLI_Referenz.md).

**Verifikationsanker.** Nach jeder Einrichtung und nach jeder Änderung, die
das Verhalten nicht ändern soll:

```bash
pytest -m anker
```

Der Anker braucht den Datensatz, gebaute Label-Dateien und ein trainiertes
Netz und läuft deshalb nicht in der normalen Suite. Er ist ein
Reproduzierbarkeitsanker, keine Zielvorgabe.
---

## Dokumentation

Die Dateien in `docs/` teilen sich nach ihrer **Lebensdauer**:
Zustandsdokumente beschreiben, wie es *jetzt* ist — datumsfrei, und Falsches
wird ersetzt statt annotiert. Das Journal beschreibt, *wie es dazu kam*, und
wird nie überschrieben.

| Datei | Frage, die sie beantwortet |
|---|---|
| **`Code_Guide.md`** | Wie funktioniert der Code? Mechanik und tragende Entscheidungen, entlang der Pipeline. **Der Einstieg.** |
| **`Code_Karte.md`** | Erster Überblick. Wo liegt was? Module, Verzeichnisbaum, Aufbau der Slugs und Dateinamen. |
| **`CLI_Referenz.md`** | Welche Skripte und Parameter gibt es? Nachschlagewerk samt Stolperfallen. |
| **`Experimente.md`** | Welche Läufe lohnen sich? Sieben Versuche mit Frage, Erwartung und Aussage. |
| **`Datensatz.md`** | Woraus bestehen die Daten, und welche Messungen begründen die Parameter? |
| **`Kantenkosten_NN.md`** | Warum diese Nachbarschaft, diese Merkmale, dieser Arbeitspunkt? |
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