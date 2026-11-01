# event-piv-multicut

Event-basierte Particle Image Velocimetry: aus den Rohdaten einer
Event-Kamera Partikel-Trajektorien rekonstruieren, formuliert als
**Multicut-Problem** und gelöst mit [RAMA](https://github.com/pawelswoboda/RAMA).

> Bachelorarbeit Informatik, Heinrich-Heine-Universität Düsseldorf, betreut
> von Prof. Dr. Paul Swoboda.

---

## Worum es geht

**Eingabe** ist der Eventstrom eines Runs, für Training und Bewertung dazu
die Ground Truth als Bounding Boxes je Partikel. **Ausgabe** ist für jedes
Event (an den Vorgaben jedes ON-Event) die Nummer seiner Trajektorie oder
`-1`, wenn es keiner zugeordnet wird. Dazu kommen für jede Trajektorie, die
[7] fitten kann, Ort und Geschwindigkeit über der Zeit.

**Zwei Stufen, beide Multicut.** In Stufe 1 ist jedes Event ein Knoten: ein
kleines neuronales Netz bewertet benachbarte Eventpaare, und sein Logit geht,
um einen einstellbaren Versatz verschoben, als Kantenkosten an den Solver.
Gelöst wird fensterweise, das Ergebnis sind kurze Bahnstücke — Tracklets.

In Stufe 2 ist jedes Tracklet ein Knoten. Ein Paar-Transformer bewertet, ob
zwei Tracklets zum selben Partikel gehören, und sein Logit geht, um einen
einstellbaren Versatz verschoben, als Kantenkosten an den Solver. Gelöst wird
**einmal über den ganzen Run**. Die Cluster dieser Partition sind die
Trajektorien.

Zuletzt ([7]) schätzen je Trajektorie ein Kalman-Filter mit RTS-Glätter, eine
Positionskorrektur und ein Glättungsspline Ort und Geschwindigkeit auf einem
frei wählbaren Zeitgitter.

Die Begriffe — Tracklet, Trajektorie, Slug, Arbeitspunkt, die Stufen [1] bis
[7] und die Darstellungen [A] bis [D] — erklärt das Glossar im Docstring von
`src/eventpiv/__init__.py`.

---

## Aufbau des Repos

| Ordner | Inhalt |
|---|---|
| `src/eventpiv/kern/` | Pfade (`config.py`), Konfiguration und Vorgaben (`runconfig.py`), Laden der Rohdaten |
| `src/eventpiv/kette/` | die Stufen [1] bis [7] |
| `src/eventpiv/bewerten/` | Kennzahlen gegen die Ground Truth |
| `src/eventpiv/zeigen/` | die Darstellungen [A] bis [D], als GIF |
| `src/eventpiv/psee_io/` | Eventloader aus der Prophesee-Toolbox, Fremdcode |
| `scripts/` | je Stufe und Darstellung ein Aufrufskript mit Hydra; die Bedienung erklärt `scripts/_cli.py` |
| `configs/` | Hydra: je Stufe eine Treiberdatei, in Unterordnern die Gruppendateien je Datensatz |
| `slurm/` | sbatch-Skripte für die Stufen und die Anker |
| `tests/` | pytest-Suite auf einem synthetischen Mini-Datensatz, dazu die Anker |
| `tools/` | Werkzeuge außerhalb der Kette: Konverter, Bewertung, Parameterschätzer |

---

## Anforderungen

- **GPU:** eine NVIDIA-Karte mit Treiber für CUDA 13; für ihre Architektur
  wird RAMA gebaut (siehe Installation). Auf der Karte rechnen [2], [4],
  [5a], [6b] und [6c]; [1], [3], [6a] und [7] laufen nur auf der CPU.
- **Grafikspeicher:** Das Training braucht eine Karte mit 24 GB: [6b]
  belegt an den Vorgaben um 20 GB, [2] auf dichten Runs knapp 15 GB. Die
  Inferenz lief auf Karten mit 24 und 32 GB; mit weniger ist sie nicht
  geprüft.
- **Arbeitsspeicher:** wächst mit der Eventzahl eines Runs. Auf den
  dichtesten STELLA-Runs brauchen [1], [3], [6a] und [7] je 15 bis 30 GB,
  [6c] mit Bewertung bis knapp 60 GB. Auf einem mittleren Run wie val-Run 10
  bleiben [5a], [6a] und [7] unter 10 GB.
- **Zum Ausprobieren** eignet sich deshalb ein mittlerer oder kleiner Run wie
  val-Run 10; die größten Runs brauchen ein Vielfaches an Speicher und Zeit.
- **Geprüft** mit Python 3.12, torch 2.13 und 2.14 (cu130) und RAMA v0.0.8.

---

## Installation

Reihenfolge: Repo, Python-Umgebung, PyTorch, RAMA, dann das Paket. `torch`
und `rama_py` stehen bewusst nicht in den `dependencies`: beide brauchen eine
bestimmte CUDA-Variante, und pip installierte sonst eine falsche nach. Mit
welchen Versionen das Repo betrieben wird, steht in
`requirements-cluster.txt`; ein Installationsweg ist die Datei nicht.

```bash
git clone https://github.com/d-kekec/event-piv-multicut.git ~/event-piv-multicut
python3.12 -m venv ~/eventpiv-env
source ~/eventpiv-env/bin/activate
```

Python 3.12; `pyproject.toml` verlangt mindestens 3.10. Die beiden Pfade
`~/event-piv-multicut` und `~/eventpiv-env` erwarten auch die sbatch-Skripte.

Fehlen `python3.12 -m venv` oder die C-Header von Python und gibt es kein
sudo, legt uv die Umgebung an:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
~/.local/bin/uv python install 3.12
~/.local/bin/uv venv ~/eventpiv-env --python 3.12 --seed
source ~/eventpiv-env/bin/activate
```

`--seed` installiert nur pip.

### PyTorch

Vor allem anderen, aus dem Index seiner CUDA-Variante:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

Der Grafiktreiber muss die CUDA-Version des Wheels unterstützen, für
`cu130` also CUDA 13.0 oder neuer.

### RAMA

`rama_py`, die Python-Anbindung von RAMA, wird aus dem Quellcode gebaut, an
Commit `ee459d6` (v0.0.8), ausgecheckt im Detached-HEAD-Zustand.
Voraussetzungen:

- ein CUDA-Toolkit mit `nvcc`. Die CUDA-Pakete, die pip mit torch
  installiert, enthalten keinen Compiler. Den Pfad `/usr/local/cuda` an das
  eigene Toolkit anpassen:

  ```bash
  export PATH=/usr/local/cuda/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
  ```

- `cmake`, `ninja`, `make` und ein C++-Compiler (gebaut wurde mit GCC 13);
- die C-Header von Python, bei System-Python etwa das Paket
  `python3.12-dev`.

```bash
git clone https://github.com/pawelswoboda/RAMA.git ~/RAMA
cd ~/RAMA
git checkout ee459d6
pip install "setuptools<81"
export CUDAFLAGS="-gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120"
printf '%s\n' "$CUDAFLAGS" | cat -A   # genau eine Zeile, die auf $ endet
rm -rf build
pip install --no-build-isolation .
```

RAMAs `setup.py` importiert `pkg_resources`, das neuere setuptools-Versionen
nicht mehr enthalten. Ohne `--no-build-isolation` baut pip mit dem neuesten
setuptools und bricht ab; deshalb `setuptools<81` in der venv und der Build
ohne Abschottung.

Je Zielarchitektur ein `-gencode`; das Beispiel deckt sm_89 (etwa RTX 4090)
und sm_120 (etwa RTX 5090) ab. Fallstricke:

- `-DCMAKE_CUDA_ARCHITECTURES` auf der Kommandozeile verdeckt still die
  Angaben in RAMAs `CMakeLists.txt`, deshalb `CUDAFLAGS`.
- `CUDAFLAGS` wird nur beim ersten Konfigurieren eines Build-Verzeichnisses
  gelesen. Deshalb `rm -rf build` vor jedem Versuch, auch nach einem
  gescheiterten, in derselben Shell wie `pip install`. Sonst nimmt CMake die
  Flags aus seinem Zwischenspeicher.
- Ein Zeilenumbruch in `CUDAFLAGS`, etwa beim Kopieren der langen Zeile,
  schneidet die Flags ab; CMake warnt dann mit „contained a newline“, und
  nvcc bricht ab. Die `printf`-Zeile oben zeigt ihn.
- Eingebettet wird nur Maschinencode für die genannten Architekturen, kein
  PTX. Auf einer Karte anderer Architektur läuft `rama_py` nicht.

eventpiv importiert `rama_py` erst beim Lösen; die Testsuite läuft ohne RAMA.

### Das Paket

```bash
cd ~/event-piv-multicut
pip install -e ".[dev]"
```

Das bringt die übrigen Abhängigkeiten mit, darunter Hydra, OmegaConf und
W&B; `[dev]` fügt pytest hinzu.

### Umgebungsvariablen

Ohne sie bricht schon der Import von `eventpiv.kern.config` ab:

```bash
export EVENTPIV_DATA=/pfad/zu/den/daten   # das Verzeichnis mit train/, val/, test/
export EVENTPIV_OUT=/pfad/fuer/ausgaben   # Wurzel aller erzeugten Dateien
```

Je Datensatz ein eigenes Paar: Label-Dateien tragen keinen Slug und lägen
sonst für zwei Datensätze am selben Ort.

### Prüfen

```bash
python -m eventpiv.kern.config   # aufgelöste Pfade und Runs je Split
pytest -q                        # Mini-Datensatz, ohne Daten und ohne GPU
python - <<'EOF'
import torch, rama_py
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
opts = rama_py.multicut_solver_options("PD"); opts.sanitize_graph = True
print(rama_py.rama_cuda([0, 1, 2], [1, 2, 0], [1.1, -2, 3], opts))
EOF
```

Der letzte Aufruf löst einen Multicut auf der GPU und prüft damit auch, ob
`rama_py` für die Architektur der Karte gebaut ist. Erwartet: `CUDA: True`,
der Name der Karte, und RAMA gibt ein Tupel zurück.

---

## Daten

Die Datensätze sind nicht Teil des Repos. `EVENTPIV_DATA` zeigt auf ein
Verzeichnis dieser Form:

```
$EVENTPIV_DATA/
├── train/
│   ├── run1.dat
│   ├── run1_bbox.npy
│   └── …
├── val/
└── test/
```

- **Splits** heißen `train`, `val` und `test`; andere Namen weist
  `kern/config.py` ab.
- **`run<N>.dat`**: die Events im Prophesee-DAT-Format, Kopfzeilen mit `%`
  und zwei Bytes für Typ und Größe, danach je Event 4 Byte Zeitstempel in µs
  und 4 Byte mit x (Bit 0–13), y (Bit 14–27) und Polarität (Bit 28).
- **`run<N>_bbox.npy`**: die Ground Truth, ein strukturiertes Array mit den
  Feldern `t` (µs), `x`, `y`, `w`, `h` (px), `class_id`, `track_id` und
  `class_confidence`. `(x, y)` ist die Ecke oben links, `(w, h)` sind die
  Kantenlängen der Box.
- **Runs werden über die `_bbox.npy` gefunden.** Ein Run ohne Box-Datei
  fehlt in der Run-Liste seines Splits und läuft nur, wenn `selection.runs`
  oder `selection.run_nr` ihn nennt. Seine Events lädt der Code trotzdem:
  die Boxen brauchen nur [1] und die Bewertung.

### STELLA

Der öffentliche Datensatz der Arbeit STELLA (Sachs et al., 2026):
synthetisch, polydispers, Ground Truth auf 1-kHz-Raster. Die Daten liegen
auf Zenodo als `SynthData.7z`; herunterladen und entpacken:

```bash
curl -L -o SynthData.7z "https://zenodo.org/records/21531421/files/SynthData.7z?download=1"
7z x SynthData.7z
```

`EVENTPIV_DATA` zeigt auf den entpackten Ordner, der `train/`, `val/` und
`test/` enthält; die Runs liegen dort schon im Format oben vor. Die Vorgaben
in `kern/runconfig.py` sind der Arbeitspunkt von STELLA.

### batch1

Ein synthetischer Datensatz mit gleich großen GT-Boxen und Ground Truth auf
60-Hz-Raster. Er ist nicht öffentlich und kann nicht bereitgestellt werden.
Seine Werte stehen in den batch1-Gruppen unter `configs/`; wie man sie
zuschaltet, steht unter „Kette ausführen“.

### Eigene Aufnahmen

Eigene Aufnahmen bekommen ein eigenes `EVENTPIV_DATA` und `EVENTPIV_OUT`.
Eine Aufnahme ohne Ground Truth braucht nur ihre `.dat`, abgelegt als
`run<N>.dat` in einem der drei Split-Ordner. EVT3-Rohdaten (`.raw`) wandelt
`tools/konvertiere_raw.py` um; die Run-Nummer nimmt es aus der ersten
Ziffernfolge im Dateinamen. Der Konverter braucht zusätzlich das Paket
`evt3`:

```bash
pip install evt3==0.4.0
python tools/konvertiere_raw.py <aufnahme.raw> $EVENTPIV_DATA/test
```

Der Konverter und die Darstellungen gehen von einem Sensor mit
1280 × 720 px aus. Wie man darauf mit fertigen Netzen rechnet, steht unter
„Inferenz mit fertigen Netzen“.

---

## Kette ausführen

### Die Stufen

```
run<N>.dat + run<N>_bbox.npy
│
[1] build_event_labels   Event → track_id + Status aus den GT-Boxen
│                        (Trainingswahrheit, nie NN-Eingabe)
[2] build_pair_dataset   Eventpaare aus der Nachbarschaft (alpha1, r_max),
│                        beide Klassen aus denselben Fenstern
[3] train_edge_nn        Mini-MLP, BCE → Logit + Versatz = Kantenkosten
│
├─ [4]  solve_with_nn    EIN Ausschnitt: Graph → Kosten (GPU) → RAMA → Bewertung
│
└─ [5a] build_tracklets  ganzer Run in Fenstern der Länge alpha1 → Artefakt (A)
   │
   └─ [6a] build_kandidaten  Tracklets werden Knoten: Filter, Zentroid,
      │                      Eventstichprobe, Kandidatenpaare
      [6b] train_stufe2      Paar-Transformer → Logit + Versatz = Kantenkosten
      │
      [6c] solve_stufe2      EIN Multicut über den ganzen Run → Trajektorien
      │
      [7]  fit_trajektorien  Kalman, RTS, Positionskorrektur und Spline je
                             Trajektorie → Ort und Geschwindigkeit
```

**Es gibt keinen Orchestrator.** Jede Stufe leitet ihren Artefaktpfad aus der
Konfiguration ab — Stufe [3] findet den Datensatz von [2], [4] das Netz von
[3]. Dieselben Overrides an alle Aufrufe genügen.

### Bedienung

Jedes Skript unter `scripts/` ist eine Stufe oder Darstellung. Es gibt keine
`--flags`: überschrieben wird mit `gruppe.feld=wert`, und `gruppe=name`
schaltet die Gruppendatei `configs/<gruppe>/<name>.yaml` zu.

```bash
python scripts/solve_with_nn.py selection.run_nr=8 window.t0=1000000 window.dt=500
python scripts/build_tracklets.py selection.split=val 'selection.runs=[<run>]'
python scripts/build_tracklets.py datensatz=batch1 'selection.runs=[32]'
python scripts/train_edge_nn.py --cfg job --resolve   # nur anzeigen, nichts rechnen
```

Listen stehen ohne Leerzeichen in Anführungszeichen. Unbekannte Schlüssel
brechen ab, auch mit `+`. Welche Gruppen ein Skript kennt, steht in seiner
Treiberdatei `configs/<stufe>.yaml`, was ihm eigen ist, in seinem Docstring.
Das Allgemeine, dazu eine Übersicht Stufe → Skript → sbatch-Datei, steht in
`scripts/_cli.py`. Hydra legt je Aufruf die zusammengesetzte Konfiguration
unter `$EVENTPIV_OUT/hydra/` ab.

Wie die Gruppen zusammenhängen, welche Felder es gibt und welche
Gruppendateien eine Stufe annimmt, beschreibt Teil A von
[CLI_und_Logging.md](CLI_und_Logging.md).

### Training und Lauf am Arbeitspunkt

Ohne Gruppen gelten die Vorgaben aus `kern/runconfig.py`, der Arbeitspunkt
von STELLA. Die Netze lernen auf `train`, bewertet wird auf `val`:

```bash
# Stufe 1: Labels, Paar-Datensatz, Netz
python scripts/build_event_labels.py
python scripts/build_pair_dataset.py
python scripts/train_edge_nn.py
# Stufe 2: Tracklets und Kandidaten auf train, dann das Netz
python scripts/build_tracklets.py
python scripts/build_kandidaten.py
python scripts/train_stufe2.py
# auf val; die Labels liefern dort nur die Kennzahlen
python scripts/build_event_labels.py selection.split=val
python scripts/build_tracklets.py selection.split=val
python scripts/build_kandidaten.py selection.split=val
python scripts/solve_stufe2.py selection.split=val
python scripts/fit_trajektorien.py selection.split=val
```

- Ohne `selection.runs` laufen [1], [5a] und [6a] über alle Runs des Splits.
  Gebraucht werden auf `train` nur die Runs aus `pairs.runs` für [2] und aus
  `kandidaten.runs` für [6b]; wer Rechenzeit sparen will, nennt nur diese.
- Für einen anderen Split nur `selection.split` ändern. `pairs.split` und
  `kandidaten.split` bleiben `train`: sie legen fest, woraus die Netze aus
  [3] und [6b] gelernt haben, und stehen in deren Pfad.
- [4] löst einen Ausschnitt zum Ansehen und Prüfen; die Kette braucht ihn
  nicht.
- Wie eine Stufe mit vorhandenen Artefakten umgeht, nennt ihr Docstring.

### batch1

Auf batch1 braucht jede Stufe ihre Gruppen:

| Stufe | Gruppen |
|---|---|
| [1] | `labeling=batch1` |
| [2] | `neighborhood=batch1 pairsampling=batch1` |
| [3], [4], [5a] | `datensatz=batch1` |
| [6a] | `datensatz=batch1 kandidatensatz=batch1` |
| [6b], [6c], [D] | `datensatz=batch1 kandidatensatz=batch1 transformersatz=batch1` |
| [7], [D] mit Fit-Panel | wie [6c], dazu `fitsatz=batch1` |

`datensatz=batch1` setzt neighborhood, pairsampling und training zusammen;
[1] und [2] kennen es nicht. [A] braucht keine Gruppe. [B] und [C] schalten
keine Gruppen zu und laufen nur mit den Vorgaben. Fehlt eine Gruppe, rechnet
eine Stufe meist still mit den Werten von STELLA; die Falle jedes Skripts
nennt sein Docstring.

### W&B

Die meisten Stufen protokollieren nach Weights & Biases, Projekt
`event-piv-multicut`. Ohne Konto vor dem Aufruf `WANDB_MODE=disabled`
setzen, das schaltet W&B ab, oder `WANDB_MODE=offline`, das schreibt nur
lokal. Ohne W&B laufen ohnehin `build_event_labels.py`,
`identify_overlap_runs.py` und die `animate_*`-Skripte.

Was die Stufen protokollieren und wo die Zahlen ohne W&B stehen, beschreibt
Teil B von [CLI_und_Logging.md](CLI_und_Logging.md).

### Ergebnisse

Alles Erzeugte liegt unter `$EVENTPIV_OUT`; die schreibenden Skripte melden
ihre Ausgabe meist hinter `->`. Der Baum ist die Abhängigkeitskette, eine
Ebene je Stufe:

```
$EVENTPIV_OUT/
├── labels/<split>/run<N>_eventlabels.npz                        [1]
├── wp/<nb>/pairs/<ps>/
│   ├── pairs.npz                                                [2]
│   └── <tr>/
│       ├── model.pt, model.metrics.json                         [3]
│       ├── solves/<split>_run<N>_<fenster>_<s1>.json            [4]
│       └── tracklets/
│           ├── <split>_run<N>_<s1>.npz                          [5a]
│           └── stufe2/<kd>/
│               ├── <split>_run<N>_<s1>.npz                      [6a]
│               └── <tf>/
│                   ├── model_<s1>.pt                            [6b]
│                   └── solves/
│                       ├── <split>_run<N>_<s1>_<s2>.json, .npz  [6c]
│                       └── fit_<split>_run<N>_<s1>_<s2>_<fit>.npz  [7]
├── results/animations/                                          [A] bis [D]
└── hydra/<name>/<zeitstempel>/
```

`<nb>`, `<ps>`, `<tr>`, `<kd>`, `<tf>` und `<fit>` sind die Slugs der Gruppen
neighborhood, pairs, training, kandidaten, transformer und fit, `<s1>` und
`<s2>` die Kurznamen von solver und solver2. Die Pfade baut `kern/config.py`.

**Ansehen.** Die Darstellungen schreiben GIFs nach
`$EVENTPIV_OUT/results/animations/`, mit `io.out_path` anderswohin:

```bash
python scripts/animate_run.py selection.run_nr=8          # [A] Rohevents
python scripts/animate_clusters.py selection.run_nr=8     # [B] Cluster eines Ausschnitts
python scripts/animate_tracklets.py selection.run_nr=8    # [C] Tracklets über Fenstergrenzen
python scripts/animate_trajektorien.py selection.run_nr=8 # [D] Tracklets, Trajektorien, GT
```

[B] und [C] lösen selbst und brauchen das Netz aus [3]; [D] liest die
Artefakte aus [5a] und [6c], mit Fit-Panel auch aus [7].

**Bewerten.** Mit Label-Dateien messen die Stufen gegen die Ground Truth und
legen die Kennzahlen in W&B ab, dazu in JSON neben den Artefakten:
`model.metrics.json` aus [3] und die JSON-Dateien aus [4] und [6c]. Zwei
Werkzeuge bewerten nachträglich:

- `tools/stella_bewertung.py` bewertet die Fits aus [7] mit den Kennzahlen
  von STELLA (Abschnitt 4.1 des Papiers). `<solves>` ist das Verzeichnis,
  das `fit_trajektorien.py` hinter `->` meldet.
- `scripts/evaluate_pairs.py` bewertet das Netz aus [3] auf einem
  Paar-Datensatz. Das Netz wählen die Gruppen wie beim Training, den
  Datensatz `io.data_path`, ohne ihn der eigene aus [2]; den Split der
  Kennzahlen liest es aus dem Datensatz. Einen Datensatz aus val-Runs baut
  [2] mit `pairs.split=val`; beim Bewerten bleibt `pairs.split` dagegen
  `train`, sonst sucht es ein anderes Netz.

```bash
python tools/stella_bewertung.py <solves> val <runs>
python scripts/build_pair_dataset.py pairs.split=val 'pairs.runs=[<runs>]'
python scripts/evaluate_pairs.py io.data_path=<pfad>/pairs.npz
```

### Inferenz mit fertigen Netzen

Mit den Netzen aus [3] und [6b] rechnet die Kette ohne Training:
[5a] → [6a] → [6c] → [7], nach Bedarf dazu [4] und die Darstellungen.
[1], [2], [3] und [6b] entfallen.

**Wo die Netze liegen.** Unter `$EVENTPIV_OUT`, an den Pfaden aus
`config.model_file` und `config.transformer_file`:

```
wp/<nb>/pairs/<ps>/<tr>/model.pt                                   [3]
wp/<nb>/pairs/<ps>/<tr>/tracklets/stufe2/<kd>/<tf>/model_<s1>.pt   [6b]
```

An den Vorgaben, dem Arbeitspunkt von STELLA, lauten die Slugs so, wie
`tests/test_anker.py` sie festhält:

```
wp/a500_rmax5.1_hyperboloid_on-cf244e/pairs/split_train_rn30-89b883/dr_sample-7e042d/model.pt
wp/a500_rmax5.1_hyperboloid_on-cf244e/pairs/split_train_rn30-89b883/dr_sample-7e042d/tracklets/stufe2/split_train_rn30_a7500_rmax8_n32-010e37/d64_l2_cls_sample_mehrheit-e71ec8/model_P.pt
```

**Die Konfiguration muss die des Trainings sein.** Jeder Slug enthält einen
Hash über alle Felder seiner Gruppe, auch über `pairs.runs` und
`kandidaten.runs`; zusammen mit solver bestimmen sie den Pfad zum Netz.
Weicht ein Wert ab, sucht die Stufe ein anderes Netz und bricht mit dem
erwarteten Pfad ab. Für die Netze von batch1 also dieselben Gruppen wie beim
Training. `io.model_path` lädt ein Netz von einem anderen Ort ([4], [5a],
[6c]); die Artefakte landen trotzdem unter den Slugs der Konfiguration.

**Ohne Ground Truth.** Ohne Boxen entfällt [1]. [4], [5a], [6a], [6c] und
[7] laufen trotzdem und schreiben ihre Artefakte; es fehlen nur die
Kennzahlen gegen die Ground Truth. [D] lässt das GT-Panel weg, [B] und [C]
zeigen keine GT. Weil ein Run ohne `_bbox.npy` nicht gefunden wird, nennt
jeder Aufruf ihn ausdrücklich:

```bash
export EVENTPIV_DATA=/pfad/zu/aufnahmen    # mit test/run1.dat
export EVENTPIV_OUT=/pfad/fuer/ausgaben    # die Netze an den Pfaden oben
python scripts/build_tracklets.py selection.split=test 'selection.runs=[1]'
python scripts/build_kandidaten.py selection.split=test 'selection.runs=[1]'
python scripts/solve_stufe2.py selection.split=test 'selection.runs=[1]'
python scripts/fit_trajektorien.py selection.split=test 'selection.runs=[1]'
python scripts/animate_trajektorien.py selection.split=test selection.run_nr=1
```

Die Netze werden an dieselben Pfade unter dem `EVENTPIV_OUT` der Aufnahmen
kopiert. Das eigene `EVENTPIV_OUT` verhindert, dass ein Run auf die
Artefakte eines anderen Datensatzes mit gleichem Split und gleicher Nummer
trifft; [6a], [6c] und [7] übersprängen ihn sonst als schon gerechnet.

---

## Auf einem Slurm-Cluster

Für die meisten Stufen liegt unter `slurm/` ein sbatch-Skript; welches zu
welcher Stufe gehört, zeigt die Übersicht in `scripts/_cli.py`. Die
Variablen stehen direkt vor `sbatch`, etwa
`RUNS="8 13" SPLIT=train sbatch slurm/tracklets.sbatch`; was ein Skript
braucht und welche Variablen es kennt, steht in seinem Kopf, das Allgemeine
im slurm-Abschnitt von `scripts/_cli.py`. Anzupassen sind die
`#SBATCH`-Zeilen und die Pfade zu venv und Repo, und `logs/` muss im
Absendeverzeichnis existieren.

Jobs ohne `--gres` können auf einem Knoten mit wenig Arbeitsspeicher
landen; für [1], [6a] und [7] auf dichten Runs einen Knoten mit genug
Arbeitsspeicher wählen. Die Skripte fangen einen Fehlschlag ab und enden
trotzdem mit Exit-Code 0: Mit `--dependency=afterok` verkettete Jobs
laufen also weiter, auch wenn ein Schritt gescheitert ist. Die Logs zeigen
das mit „FEHLGESCHLAGEN“.

---

## Anker

Die Regressionsanker in `tests/test_anker.py` halten fest, was der
Arbeitspunkt eines Datensatzes auf echten Daten liefert. Sie fangen
Regressionen und sind keine Zielvorgabe; ändern sich die Vorgaben gewollt,
werden ihre Tabellen im selben Commit nachgezogen.

```bash
EVENTPIV_NAME=stella pytest -m anker    # Stufe 1
EVENTPIV_NAME=stella pytest -m anker2   # Stufe 2
```

- **`-m anker`** löst je Eintrag ein Stufe-1-Fenster eines train-Runs und
  vergleicht Kantenzahl, Anteil negativer Kanten, Reinheit, Fragmentierung
  und Verschmelzung mit festen Werten. Dazu prüft es, ob die Gruppendateien
  des Datensatzes die festgehaltenen Slugs ergeben.
- **`-m anker2`** löst Stufe 2 auf einem ganzen train-Run und vergleicht
  Knoten-, Kanten- und Clusterzahl, Anteil negativer Kanten, Verschmelzung
  und HOTA.

Beide brauchen `EVENTPIV_NAME` (`stella` oder `batch1`) mit dazu passenden
`EVENTPIV_DATA` und `EVENTPIV_OUT`, eine GPU mit RAMA und die Label-Dateien
der Anker-Runs; welche Runs, steht in den Tabellen des Tests. `anker`
braucht das Netz aus [3], `anker2` das Netz aus [6b] und für seinen Run die
Artefakte aus [5a] und [6a]. Fehlt ein Netz oder fehlen die Kandidaten, wird
übersprungen; den Grund zeigt `pytest -rs`. Ohne `-m` sind beide abgewählt.
Braucht ein Datensatz mehr Arbeitsspeicher, laufen sie als Job:
`sbatch slurm/anker.sbatch`, `MARKER` wählt die Stufe.

---

## Fremdcode und Referenzen

**psee_io.** `src/eventpiv/psee_io/` stammt aus der Prophesee-Toolbox und
steht unter der Apache License 2.0; der Lizenztext liegt in
`src/eventpiv/psee_io/LICENSE.txt`. Für eventpiv geändert ist
`psee_loader.py`: `_ev_size` und `_start` werden mit `int()` umgewandelt,
für NumPy 2. Der Kopf der Datei vermerkt die Änderung.

**RAMA.** Abbas und Swoboda, *RAMA: A Rapid Multicut Algorithm on GPU*,
CVPR 2022. Code: https://github.com/pawelswoboda/RAMA, gebaut am Commit aus
dem Abschnitt Installation.

**STELLA.** Sachs, Jung, Kahl, Keuper, Willert und Cierpka, *STELLA: a
modular framework for SpatioTemporal Event-based Lagrangian particLe
trAcking*, Experiments in Fluids 67, 110 (2026). Code auf GitHub:
https://github.com/sesa1504/STELLA, Daten auf Zenodo:
https://zenodo.org/records/21531421