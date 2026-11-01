# CLI und Logging

Was sich an den Skripten einstellen lässt (Teil A) und was sie protokollieren
(Teil B). Wie man die Kette installiert, aufruft und auf einen anderen
Datensatz umstellt, beschreibt das [README](README.md). Begriffe wie
Tracklet, Slug oder Arbeitspunkt erklärt das Glossar im Docstring von
`src/eventpiv/__init__.py`. Maßgeblich sind `src/eventpiv/kern/runconfig.py`
für Felder und Vorgaben und `configs/` für Treiber- und Gruppendateien.

---

## Teil A · Konfiguration

### A1 · Wie die Gruppen funktionieren

**Gruppen und Stufen.** Eine Gruppe ist eine Dataclass in `runconfig.py`, die
zusammengehörige Felder bündelt, etwa neighborhood mit `alpha` und `r_max`.
Die Konfiguration einer Stufe setzt sich aus Gruppen zusammen; welche, zeigt
A2. Ein Skript nimmt jedes Feld der Gruppen seiner Stufe an, die Felder aller
Gruppen stehen in A3. Ohne Angabe gelten die Vorgaben aus `runconfig.py`, der
Arbeitspunkt von STELLA.

**Setzen.** `gruppe.feld=wert` setzt ein Feld für einen Aufruf. Eine
Gruppendatei `configs/<gruppe>/<name>.yaml` setzt mehrere Felder einer Gruppe
zugleich und kommt mit `gruppe=name` dazu. Sie darf jedes Feld ihrer Gruppe
enthalten; was sie nicht nennt, behält die Vorgabe. Welche Gruppendateien
eine Stufe annimmt, legt ihre Treiberdatei `configs/<stufe>.yaml` fest (A4).
Ein Wert auf der Kommandozeile gewinnt gegen die Gruppendatei, die
Gruppendatei gegen die Vorgabe. Unbekannte Gruppen und Felder brechen ab.

**Prüfen.** Erlaubte Werte und Wertebereiche prüft `runconfig.py`, sobald der
Lauf die Konfiguration baut, also bevor ein Pfad entsteht.
`--cfg job --resolve` zeigt die zusammengesetzte Konfiguration, ohne diese
Prüfungen auszuführen.

**Pfad und Identität.** neighborhood, pairs, training, kandidaten und
transformer bilden je eine Verzeichnisebene unter `$EVENTPIV_OUT`, benannt
nach ihrem Slug: ein lesbarer Kern und ein Hash über alle Felder der Gruppe,
etwa `a500_rmax5.1_hyperboloid_on-cf244e`. fit steht mit seinem Slug im
Dateinamen, solver und solver2 mit einem Kurznamen. Ein anderer Wert führt so
auf einen anderen Pfad, und vorhandene Ergebnisse bleiben liegen. compute und
io gehen in keinen Pfad ein. Wer eine Vorgabe in `runconfig.py` ändert,
verschiebt den Arbeitspunkt; der Test `test_vorgaben_sind_ein_arbeitspunkt`
hält die Slugs der Vorgaben fest und schlägt dann an.

### A2 · Stufen und ihre Abhängigkeiten

Jede Stufe rechnet mit einigen Gruppen und findet ihre Eingabe über die
Gruppen der Stufen davor.

| Stufe | Skript | rechnet mit | findet ihre Eingabe über |
|---|---|---|---|
| [1] | `build_event_labels.py` | labeling · selection | – |
| [2] | `build_pair_dataset.py` | neighborhood · pairs | – |
| [3] | `train_edge_nn.py` | training | neighborhood · pairs |
| [4], [5a] | `solve_with_nn.py`, `build_tracklets.py` | neighborhood · solver · selection · window | neighborhood · pairs · training |
| [6a] | `build_kandidaten.py` | kandidaten · selection | neighborhood · pairs · training · solver |
| [6b] | `train_stufe2.py` | transformer | neighborhood · pairs · training · solver · kandidaten |
| [6c] | `solve_stufe2.py` | solver2 · selection | neighborhood · pairs · training · solver · kandidaten · transformer |
| [7] | `fit_trajektorien.py` | fit · selection | neighborhood · pairs · training · solver · kandidaten · transformer · solver2 |
| [A] | `animate_run.py` | selection · window · anim | – |
| [B], [C] | `animate_clusters.py`, `animate_tracklets.py` | wie [4], dazu anim | wie [4] |
| [D] | `animate_trajektorien.py` | selection · window · anim | wie [7], für das Fit-Panel dazu fit |
| – | `evaluate_pairs.py` | – | neighborhood · pairs · training |
| – | `identify_overlap_runs.py` | selection · overlap | – |

Dazu hat jedes Skript io, und alle bis auf [1], [7], [A] und
`identify_overlap_runs.py` haben compute.

Die Gruppen der rechten Spalte setzt ein Aufruf so wie die Stufe, die seine
Eingabe geschrieben hat: [3] wie [2], [4] und [5a] wie [3], [6a] wie [5a] und
so fort. Für Gruppendateien gilt dasselbe, etwa `datensatz=batch1` an jede
Stufe ab [3]. Weicht ein Wert ab, sucht die Stufe unter einem anderen Pfad.
Liegt dort keine Datei, bricht sie mit dem erwarteten Pfad ab; liegt dort
eine, etwa die der Vorgaben, rechnet sie mit dieser. Wer eine Gruppe ändert,
rechnet deshalb ab der Stufe neu, die mit ihr rechnet, und startet alle
späteren mit demselben Wert.

Geprüft wird zusätzlich neighborhood: Das Training vergleicht die Gruppe mit
dem Paar-Datensatz, das Lösen mit dem Netz, und eine Abweichung bricht ab.
[6a] prüft, dass `kandidaten.v_max` gleich `neighborhood.v_max` ist.

selection wählt die Runs, für die gerechnet wird; für jeden muss die Eingabe
aus der Stufe davor existieren. Ein anderer Split braucht nur
`selection.split`, denn `pairs.split` und `kandidaten.split` benennen die
Trainingsdaten der Netze.

### A3 · Die Gruppen

In der Reihenfolge der Kette, je Gruppe Feld, Vorgabe und Bedeutung.

#### `labeling` · [1]

| Feld | Vorgabe | Bedeutung |
|---|---|---|
| `r_assign` | `5.5` | px. Radius um das nächste Boxzentrum; Events weiter außen werden Rauschen (Status 2). |
| `r_ambig` | `2.0` | px. Mindestvorsprung des nächsten Zentrums vor dem zweitnächsten; darunter wird das Event mehrdeutig (Status 1). |

Labels gelten immer für den ganzen Run. Label-Dateien liegen ohne Slug je
Split und Run und tragen ihre Parameter. Passt eine vorhandene Datei nicht
zur Konfiguration, bricht [1] mit `LabelingMismatch` ab; `io.overwrite=true`
baut sie neu.

#### `neighborhood` · [2] bis [7], Slug `<nb>`

| Feld | Vorgabe | Bedeutung |
|---|---|---|
| `alpha` | `500` | µs. Zeitliche Reichweite einer Kante, zugleich die Länge eines Tracklet-Fensters. |
| `r_max` | `5.1` | px. Räumliche Reichweite einer Kante. |
| `geom` | `hyperboloid` | Form der Reichweite über Δt: `ellipsoid` schrumpft, `cylinder` bleibt gleich, `hyperboloid` weitet sich mit der Drift. |
| `v_max` | `1.79e-3` | px/µs. Schnellste Partikelgeschwindigkeit; wirkt bei `hyperboloid`. |
| `polarity` | `on` | Knotenmenge: `on` nur positive Events, `both` alle. |

neighborhood legt fest, welche Eventpaare Kanten bilden, im Paarbau wie im
Einsatz; das Netz aus [3] kennt genau diese Kanten. `v_max` steht auch bei
`ellipsoid` und `cylinder` im Hash. Ein kürzeres `alpha` schneidet Kanten
weg, die die Geometrie erlaubt; ein längeres lässt zwei Partikel öfter in
einem Tracklet verschmelzen, und Stufe 2 verbindet Tracklets, trennt sie aber
nicht. Werte für einen neuen Datensatz, auch `labeling.r_assign`, schätzt
`tools/schaetzer/stufe1_parameter_schaetzer.py`.

#### `pairs` · [2], Slug `<ps>`

| Feld | Vorgabe | Bedeutung |
|---|---|---|
| `split` | `train` | Split, aus dem die Paare stammen. |
| `runs` | 30 train-Runs | Quell-Runs, über die Eventrate geschichtet, aufsteigend; die Reihenfolge geht in den Hash ein. |
| `ja_frac` | `0.02` | Anteil der JA-Paare, der je Fenster gezogen wird; NEIN wird vollständig gesammelt. |
| `max_no` | `1000000` | Deckel für NEIN je Run. |
| `fenster_schritt` | `20` | Abgeschritten wird jedes n-te Fenster, dieses vollständig. |
| `ja_vorrat` | `2.0` | JA wird je Run auf `ja_vorrat` × NEIN gekürzt. |
| `seed` | `0` | Ziehung und Deckel. |

Gesammelt wird in anstoßenden Fenstern der Länge `neighborhood.alpha`, den
Fenstern von Stufe 1 im Einsatz; beide Klassen kommen aus denselben Fenstern.
`ja_frac` hält den Speicherbedarf in Grenzen, denn ein ganzer Run liefert weit
mehr JA-Paare, als in den Arbeitsspeicher passen. `ja_vorrat` liegt über
`training.sample_ratio`: [2] legt einen Vorrat an, [3] setzt den
Betriebspunkt. Weil der Trim je Run läuft, trägt jede Auswahl von Runs
dasselbe Verhältnis, auch der Holdout in [3].

Den Spitzenbedarf an Arbeitsspeicher bestimmen `ja_frac` und die rohe
JA-Menge je Run; `max_no` und der Trim greifen nach dem Sammeln und begrenzen
Dateigröße und Klassenverhältnis. Der Paarbau rechnet den Jaccard in jedem
Fenster, auch bei `training.feature_mode=dr`. `pairs.npz` trägt ihn deshalb
immer, und ein Wechsel des Merkmals braucht keinen neuen Paarbau.

#### `training` · [3], Slug `<tr>`

| Feld | Vorgabe | Bedeutung |
|---|---|---|
| `feature_mode` | `dr` | Merkmale je Kante: `dr` = (Δt, Δr), `drjac` = (Δt, Δr, J) mit J als Jaccard-Index der Nachbarschaften beider Events im Fenster. |
| `imbalance` | `sample` | JA wird bis `sample_ratio` untersampelt; der einzige Modus. |
| `sample_ratio` | `1.0` | Ziel JA:NEIN im Training. |
| `hidden` | `32` | Breite der versteckten Schicht des MLP. |
| `epochs` / `batch` / `lr` | `100` / `65536` / `3e-2` | Ein zu großes `lr` zeigt sich als zackige Kurve. |
| `val_frac` | `0.2` | Anteil der Runs des Paar-Datensatzes im Holdout. |
| `seed` | `0` | Initialisierung und Holdout-Wahl. |

`sample_ratio` verschiebt den Achsenabschnitt des Logits um log(s) und lässt
die Rangfolge der Kanten unverändert. Bei 1:1 liegt die Nullstelle mitten in
der Verteilung, TPR und TNR bleiben im Training ablesbar. `drjac` ist für
Daten, auf denen die Kosten mit `dr` allein nie negativ werden. Nur mit
`drjac` rechnet auch das Lösen den Jaccard; es legt dafür die
Nachbarschaftsmatrix eines Fensters dicht als float32 an, n² · 4 Byte bei
n Events, und das begrenzt auf dichten Daten den Grafikspeicher.

#### `solver` · [4], [5a], Kurzname `<s1>`

| Feld | Vorgabe | Bedeutung |
|---|---|---|
| `backend` | `rama` | `rama`: Multicut mit RAMA. `cc`: Zusammenhangskomponenten nach Löschen der negativen Kanten, auf der CPU. |
| `mode` | `P` | RAMA-Modus `P`, `PD`, `PD+` oder `D`; wirkt bei `rama`. |
| `cost_offset` | `0.0` | Wird auf jede Kante addiert. |
| `use_knn` | `false` | Kanten je Knoten auf die `k_max` nächsten Nachbarn kappen. |
| `k_max` | `24` | Wirkt mit `use_knn`. |

`cost_offset` legt den Betriebspunkt fest. Er verschiebt die Schwelle
zwischen Schneiden und Verbinden bei gleicher Rangfolge der Kanten und
tauscht so Verschmelzung gegen Fragmentierung; bei 0 entscheidet allein das
Netz. `training.sample_ratio` wirkt ebenso, verlangt aber ein neues Training.

`cc` dient als Vergleich: CC verbindet zwei Wolken, sobald eine einzige Kante
zwischen ihnen positiv ist, der Multicut wägt alle Kanten gegeneinander ab.
`P` ist rein primal und schafft größere Instanzen, `PD+` liefert die beste
Qualität und braucht länger; `D` bricht mit dem verwendeten RAMA-Stand ab.
`use_knn` ändert das Modell: An einer Kreuzung sind die nächsten Nachbarn
eines Events oft die des anderen Partikels.

Der Kurzname ist der Modus, ergänzt um das, was von der Vorgabe abweicht:
`P`, `P_cc`, `P_o0.4`, `P_knn24`.

#### `selection` · `window`

| Feld | Vorgabe | Bedeutung |
|---|---|---|
| `selection.split` | `train` | Split, für den gerechnet wird: `train`, `val` oder `test`. |
| `selection.run_nr` | `null` | Ein einzelner Run, für [4] und die Darstellungen. |
| `selection.runs` | `[]` | Mehrere Runs; leer heißt alle Runs des Splits. |
| `window.t0` | `1700000` | µs ab Run-Anfang: Beginn des Ausschnitts. |
| `window.dt` | `20000` | µs: Länge des Ausschnitts. |

Run-Nummern gelten je Split: run10 aus `val` ist ein anderer Run als run10
aus `train`. `window` schneidet den betrachteten Ausschnitt aus; die Länge
eines Tracklet-Fensters ist `neighborhood.alpha`. `null` für beide heißt
ganzer Run. Die Vorgabe ist ein kurzes Fenster für einzelne Solves und
Tests; `configs/tracklets.yaml` setzt für [5a] beide auf `null`, [A] und [D]
haben `null` als Vorgabe. [4] braucht einen Ausschnitt, ohne ihn würde der
ganze Run ein einziger Graph. In [D] schneidet `window` die Anzeige zu, die
Farben gelten für den ganzen Run.

#### `kandidaten` · [6a], Slug `<kd>`

| Feld | Vorgabe | Bedeutung |
|---|---|---|
| `split` | `train` | Split der Trainingsdaten von [6b]. |
| `runs` | wie `pairs.runs` | Runs der Trainingsdaten von [6b], aufsteigend; die Reihenfolge geht in den Hash ein. |
| `alpha` | `7500` | µs. Reichweite zwischen den Startzeiten `t_min` zweier Knoten. |
| `r_max` | `8.0` | px. Taille: Zentroidabstand zweier Knoten derselben Bahn bei kleinem Δt. |
| `geom` | `hyperboloid` | Wie bei neighborhood; die Aufweitung trägt die Drift `v_max` · Δt. |
| `v_max` | `1.79e-3` | px/µs. Gleich `neighborhood.v_max`. |
| `min_events` | `1` | Kleinste Knotengröße in Events; bei 1 wird jedes Tracklet ein Knoten. |
| `n_events` | `32` | Event-Tokens je Knoten; kleinere Knoten werden maskiert aufgefüllt. |
| `seed` | `0` | Ziehung der Event-Stichprobe je Knoten. |

Ein Knoten ist ein Tracklet aus (A). Die Gruppe misst den zeitlichen Abstand
zwischen Startzeiten und den räumlichen zwischen Zentroiden; sie hat dafür
eigene Felder, ohne `polarity`. Gerechnet wird für die Runs aus selection;
`split` und `runs` benennen die Trainingsdaten und bleiben auch für `val`
gleich. Mehr `alpha` tauscht Brüche gegen Verschmelzung. Jedes `min_events`
über 1 verwirft mit den Bruchstücken auch Brücken über Stellen, an denen ein
Partikel nur wenige Events liefert. Eine bewusste Abweichung der beiden
`v_max` erlaubt `io.strict=false`.

#### `transformer` · [6b], Slug `<tf>`

| Feld | Vorgabe | Bedeutung |
|---|---|---|
| `d_model` | `64` | Breite der Tokens; durch `n_heads` teilbar. |
| `n_layers` / `n_heads` / `d_ff` | `2` / `4` / `256` | Größe des Encoders; mit `d_model` zusammen rund 105 000 Parameter. |
| `dropout` | `0.0` | Kein Dropout; je Epoche kommen rund 38 frisch gezogene Paare auf einen Parameter. |
| `pooling` | `cls` | Eingang des Kopfes: `cls` das CLS-Token, `mean` das Mittel der echten Tokens. Der Unterschied ist nicht gemessen. |
| `label_modus` | `mehrheit` | `streng`: Knoten mit genau einem GT-Partikel. `mehrheit`: dazu verschmolzene, nach ihrer Mehrheit. |
| `gt_min_anteil` | `0.9` | Mindestanteil der Mehrheit bei `mehrheit`. |
| `imbalance` / `sample_ratio` | `sample` / `3.0` | Ziel JA:NEIN je Epoche; setzt den Betriebspunkt von Stufe 2 mit. |
| `train_max` | `4000000` | Trainingspaare je Epoche, klassenneutral gedeckelt; `null` heißt kein Deckel. |
| `epochs` / `batch` / `lr` | `100` / `1024` / `1e-3` | `lr` ist kleiner als in Stufe 1, das Netz ist um Größenordnungen größer. |
| `val_frac` / `seed` | `0.2` / `0` | Holdout, nach Runs getrennt. |

Die Trainingsmenge wird je Epoche neu gezogen: NEIN und JA im Verhältnis
1 : `sample_ratio`, zusammen höchstens `train_max`, ohne Zurücklegen. Mit
Deckel sieht das Netz über den Lauf ein Vielfaches des Datensatzes; die
Ziehung hängt an `seed` und ist reproduzierbar. Den Deckel braucht es, weil
die Kandidatenzahl mit `kandidaten.alpha` wächst, die nötige Zahl gesehener
Kanten aber nicht.

`label_modus` entscheidet, welche Paare Gradienten bekommen. Ziel und Maske
entstehen erst im Training aus `gt_track_id`, `gt_anteil` und `n_gt`, die das
Artefakt aus [6a] je Knoten trägt; ein Wechsel braucht deshalb keinen neuen
Kandidatensatz.

#### `solver2` · [6c], Kurzname `<s2>`

| Feld | Vorgabe | Bedeutung |
|---|---|---|
| `backend` | `rama` | Wie bei solver. |
| `mode` | `P` | Wie bei solver. |
| `cost_offset` | `1.0` | Wird auf jede Kante addiert; in Stufe 2 positiv. |

Ohne Versatz zerfallen in Stufe 2 mehr Cluster, als es Partikel gibt; der
Versatz ist deshalb positiv. Mit `cc` verschmelzen zwei Trajektorien, sobald
eine einzige Kante zwischen ihnen positiv ist; der Vergleich mit `rama` zeigt
den Beitrag des Multicut. An den Vorgaben lautet der Kurzname `ramaP_o1`.

solver wählt die Quelle von Stufe 2, also welche Tracklets und Kandidaten
gelesen werden, und steht im Dateinamen von (A) und [6a]. solver2 wählt die
Rechnung von Stufe 2: `solver2.mode=PD` rechnet Stufe 2 mit `PD`,
`solver.mode=PD` sucht Stufe-1-Artefakte dieses Modus.

#### `fit` · [7], Slug `<fit>` im Dateinamen

| Feld | Vorgabe | Bedeutung |
|---|---|---|
| `r_mess` | `3.0` | px². Varianz eines Einzelevents um die Bahn. |
| `q_c` | `1e7` | px²/s⁵. Intensität des Prozessrauschens (weißer Ruck), gewichtet mit der verstrichenen Zeit. |
| `a_unbekannt` | `1e4` | px/s². Streuung der Beschleunigung beim Start. |
| `init_us` | `5000.0` | µs. Fenster der Startgeraden. |
| `min_dauer_us` | `10000.0` | µs. Kürzere Bahnen werden nicht gefittet. |
| `glaetten` | `true` | RTS-Rückwärtslauf; `false` liefert den kausalen Filter. |
| `c0` | `0.06` | px. Positionskorrektur, fester Anteil. |
| `c1` | `0.0` | px je px/ms. Positionskorrektur je Geschwindigkeit. |
| `beta` | `1.31` | Positionskorrektur je Querstreuung σ_quer der Bahn. |
| `spline_eps` | `0.02` | px. Ziel-RMSE des Glättungssplines; `0` schaltet ihn ab. |
| `dt_aus` | `1000.0` | µs. Gitterweite der Ausgabe. |

**Filter.** Ein Kalman-Filter (Kalman 1960) mit RTS-Glätter (Rauch, Tung und
Striebel 1965). Auf das Ergebnis wirkt bis auf den Start nur `q_c` / `r_mess`:
Mit demselben Faktor skaliert, wachsen Q, R und die Startvarianzen gemeinsam,
nur die der Beschleunigung nicht. Abgestimmt wird deshalb `q_c`; `r_mess`
bleibt die Varianz der Daten, so sagt die Kovarianz des Filters etwas über
die Wirklichkeit. Bei `a_unbekannt` ist Unterschätzen der teure Fehler, ein
zu sicherer Start wird nur langsam korrigiert. Die Startgerade misst die
mittlere Geschwindigkeit ihres Fensters; `init_us` ist deshalb kurz gegen die
Zeit, in der sich die Geschwindigkeit merklich ändert.

**Korrektur.** Sie hebt den Vorwärtsversatz der Eventwolke auf, Betrag
c0 + c1 · |u| + beta · σ_quer, mit σ_quer je Bahn aus ihren Events quer zum
Fit. Hängt der Versatz an der Partikelgröße, trägt `beta`, so die Form aus
STELLA (Sachs et al. 2026, Gl. 1); hängt er an der Geschwindigkeit, trägt
`c1`. Alle drei null schalten die Korrektur ab. Mit `beta` > 0 fallen Bahnen
ohne bestimmbares σ_quer weg.

**Spline und Ausgabe.** Nach der Korrektur legt [7] je Bahn einen kubischen
Glättungsspline über den Ort: Ort aus dem Spline, Geschwindigkeit als seine
Ableitung, wie der Hybrid (X_KS, u_KS) aus STELLA. `dt_aus` ist für den
Filter frei. Die Bewertung interpoliert linear auf die GT-Zeiten, und ein
gröberes Gitter als das GT-Raster kostet dort Ortsfehler; im Arbeitspunkt
steht deshalb das Raster des Datensatzes.

Der Slug steht im Dateinamen, so liegen zwei Arbeitspunkte nebeneinander; an
den Vorgaben lautet er `q1e+07_r3_rts_b1.31_s0.02-679f2f`. `_rts`, `_b` und
`_s` erscheinen, wenn Glätter, `beta` und Spline wirken; `c0` und `c1` stehen
im Hash.

#### `compute` · `io`

| Feld | Vorgabe | Bedeutung |
|---|---|---|
| `compute.device` / `device_index` | `cuda` / `0` | Gerät und Karte für die Rechnungen auf der GPU. |
| `compute.cost_batch` | `4000000` | Kanten je Stapel bei der Kostenberechnung. |
| `io.strict` | `true` | Prüfungen an; `false` übergeht sie, etwa den `v_max`-Abgleich in [6a]. |
| `io.overwrite` | `false` | `true` rechnet vorhandene Ergebnisse neu und überschreibt sie. |
| `io.out_path` | `null` | Ausgabe an diesen Ort statt an den abgeleiteten Pfad. |
| `io.json_out` | `null` | [4]: Ziel der Ergebnis-JSON statt des abgeleiteten Pfads, Pflicht mit `io.model_path`. `identify_overlap_runs.py`: Ziel für Kennzahlen und Run-Listen als JSON. |
| `io.model_path` | `null` | Netz von einem anderen Ort laden. |
| `io.data_path` | `null` | Paar-Datensatz von einem anderen Ort, etwa für `evaluate_pairs.py`. |
| `io.label_dir` | `null` | Label-Dateien aus einem anderen Verzeichnis lesen. |
| `io.list_runs` | `false` | [A]: Runs nur auflisten, nichts rechnen. |
| `io.train_fest` | `[]` | Runs, die [6b] nicht in den Holdout ziehen darf. |

compute und io gehören nie zur Identität, auch wo ein Feld das Ergebnis
ändert. `train_fest` ist so ein Fall: Die Größe des Holdouts bleibt
`val_frac`, eingeschränkt wird nur, woraus gezogen wird. Welche Runs Holdout
waren, steht als `val_runs` im Checkpoint, und zwei Läufe mit verschiedenem
`train_fest` teilen sich einen Pfad. `model_path`, `data_path` und
`label_dir` laden Dateien, deren Parameter nicht aus der Konfiguration
folgen. Mit `model_path` landen die Artefakte trotzdem unter den Slugs der
Konfiguration; [4] verlangt dann `io.json_out`, damit der Ergebnispfad keine
falsche Herkunft trägt. Felder von io wirken in den Skripten, die sie lesen.
`io.overwrite=true` gehört in den einzelnen Aufruf; in einem sbatch-Skript
wäre der Schutz dauerhaft aus. `out_path` und `json_out` erweitern `~` nicht,
also absolute Pfade angeben. Ein Ausschnitt in [5a] braucht `out_path`, weil
der Artefaktname das Fenster nicht enthält.

#### `overlap` · `anim`

| Feld | Vorgabe | Bedeutung |
|---|---|---|
| `overlap.box_w` | `11.0` | px. Abstand, unter dem zwei Partikel als nah gelten. |
| `overlap.cross_thresh` | `1.4` | Mittlere Zahl naher Paare je Zeitpunkt, ab der ein Run als kreuzend gilt. |
| `overlap.mean_clean_max` | `0.05` | Darunter gilt ein Run als praktisch sauber. |

`identify_overlap_runs.py` ordnet die Runs dünn besetzter Aufnahmen nach
Überlappung und gibt Run-Listen zum Kopieren aus, mit `io.json_out` auch als
Datei. Die drei Schwellen bestimmen die Listen.

| Feld | [A] | [B], [C] | [D] | Bedeutung |
|---|---|---|---|---|
| `frame_ms` | `20.0` | `0.25` | `20.0` | Abgetastete Zeit je Frame, ms. |
| `speedup` | `1.0` | – | – | Über 1 werden Frames übersprungen. |
| `trail` | – | `true` | – | Bahnen wachsen auf; `false` zeigt Momentaufnahmen. |
| `trail_ms` | – | – | `20.0` | Gleitendes Nachleuchten, ms; `0` zeigt Momentaufnahmen. |
| `hold_ms` | – | – | `0.0` | Anzeigedauer je Frame, ms; `0` leitet sie aus `frame_ms` ab. |
| `color_by` | – | `cluster` | – | `cluster` zeigt Fragmentierung, `gt` Reinheitsfehler. |
| `min_size` | – | `5` | `15` | Cluster darunter werden grau. |
| `panels` | – | `roh,tracklets,gt` | `tracklets,trajektorien,gt` | Gezeichnete Panels, kommagetrennt. |
| `show_gt` | – | `true` | – | GT-Panel zeigen. |
| `crop` | – | `true` | `true` | Auf den belegten Bildbereich zuschneiden. |
| `downscale` / `point` | `1` / `1` | `1` / `2` | `1` / `2` | Bildverkleinerung und Punktgröße. |

[A] färbt nach Polarität. `color_by=gt` zeigt Reinheitsfehler, verbirgt aber
Fragmentierung. In [B] und [C] setzt der Trail an jeder Fenstergrenze neu an,
so bleibt die Grenze sichtbar; in [D] gleitet er, und sie verschwindet. Für
[D] stehen `roh`, `tracklets`, `trajektorien`, `fit` und `gt` zur Wahl,
`min_size` wirkt dort auf Tracklets und Trajektorien zugleich. [D] liest (A),
die Partition aus [6c] und für `fit` die Bahnen aus [7]; fehlt eine Datei,
bricht es mit deren Namen ab. Listen auf der Kommandozeile quoten:
`"anim.panels='fit,gt'"`. Die Framezahl ist die Dauer durch `frame_ms`, in
[A] zusätzlich durch `speedup`; über etwa hundert Frames wird ein GIF
unhandlich.

### A4 · Gruppendateien

Eine Gruppendatei setzt die Werte eines anderen Datensatzes oder
Arbeitspunkts. Sie darf jedes Feld ihrer Gruppe enthalten; die vorhandenen
setzen diese:

| Gruppe | Dateien | setzt |
|---|---|---|
| `labeling` | `batch1`, `stella` | `r_assign` |
| `neighborhood` | `batch1`, `stella` | `alpha`, `r_max`, `geom`, `v_max` |
| `pairsampling` | `batch1`, `stella` | `pairs.runs`, `pairs.fenster_schritt` |
| `training` | `batch1`, `stella` | `feature_mode` |
| `datensatz` | `batch1`, `stella` | neighborhood, pairsampling und training zusammen |
| `kandidatensatz` | `batch1`, `stella`, `stella_a2_15ms` | `kandidaten.runs`, `alpha`, `r_max`, `v_max`, `min_events` |
| `transformersatz` | `batch1`, `stella`, `stella_a2_15ms` | `transformer.epochs` |
| `fitsatz` | `batch1`, `stella` | alle Felder von fit außer `glaetten` |

Zuschalten lässt sich eine Gruppendatei in den Stufen, deren Treiberdatei die
Gruppe in ihrer Defaults-Liste führt:

| Stufe | zuschaltbare Gruppen |
|---|---|
| [1] | labeling |
| [2] | neighborhood · pairsampling |
| [3], [4], [5a], `evaluate_pairs.py` | neighborhood · pairsampling · training · datensatz |
| [6a] | neighborhood · pairsampling · training · datensatz · kandidatensatz |
| [6b], [6c] | neighborhood · pairsampling · training · datensatz · kandidatensatz · transformersatz |
| [7], [D] | neighborhood · pairsampling · training · datensatz · kandidatensatz · transformersatz · fitsatz |
| [A], [B], [C], `identify_overlap_runs.py` | – |

- `stella` wiederholt die Vorgaben; so ist STELLA wie batch1 ausdrücklich
  wählbar.
- `stella_a2_15ms` ist ein zweiter Arbeitspunkt auf STELLA: `kandidaten.alpha`
  15 ms, `min_events` 2 und 50 Epochen. In [6a] `kandidatensatz=stella_a2_15ms`,
  ab [6b] dazu `transformersatz=stella_a2_15ms`.
- batch1 ist nicht öffentlich; seine Werte stehen in diesen Dateien.
- Die erste Zeile einer Datei aus pairsampling, kandidatensatz,
  transformersatz oder fitsatz, etwa `# @package pairs`, lenkt ihren Inhalt in
  das Feld der Gruppe.
- `runs` stehen aufsteigend, weil die Reihenfolge in den Hash eingeht.

---

## Teil B · Logging

Was die Stufen messen und wo es steht. Die Feldnamen sind die Schlüssel, wie
W&B und die Dateien sie tragen.

### B1 · Lesehilfe

**Paare.** Auf Paarebene lautet die Frage, ob zwei Events, in Stufe 2 zwei
Tracklets, zum selben Partikel gehören: JA oder NEIN. Das Netz gibt einen
Logit aus, dessen Vorzeichen die Entscheidung ist. RAMA liest genau dieses
Vorzeichen, Schwelle 0 ist deshalb der Arbeitspunkt des Solvers. Ein FP, also
ein NEIN-Paar als JA, erzeugt Reinheitsfehler, ein FN Fragmentierung. TPR und
TNR stehen getrennt, weil im Einsatz fast alle Kanten zu einem Partikel
gehören: Dort erreicht schon „immer JA“ eine hohe Accuracy, die TNR zeigt den
Unterschied.

**Mengen.** `train/` sind die Runs, auf denen gelernt wird, `holdout/` ein
nach Runs getrennter Teil des train-Splits, auf dem nur gemessen wird. `val`
und `test` sind die Splits des Datensatzes; `test` dient der abschließenden
Bewertung.

**Präfixe.** `data/` der Datensatz, `final/` die Schlusswerte eines
Trainings, `diag/` Laufzeit und Hardware, `pairs/` der Paarbau, `solve/` ein
Fenster, `gt/` Verteilungen über die GT-Partikel, `lauf/` ein ganzer Run.
Verschachteltes steht mit Schrägstrich, etwa `lauf/stufe2/hota`.

**W&B.** Ein Lauf hält die Konfiguration (`config`, genau `asdict(cfg)`), den
Verlauf je Schritt und die Summary mit einem Endwert je Schlüssel. Die
Runs-Tabelle zeigt die Summary, und für einen Schlüssel aus dem Verlauf ist
das der zuletzt geloggte Wert.

**Speicher.** `max_rss_mb` ist der Höchststand des Prozesses. Rechnet ein
Prozess mehrere Runs, wie [6a], [6c] und [7], trägt jeder Run den Höchststand
bis einschließlich seiner selbst.

Ohne Label-Datei, etwa bei echten Aufnahmen, entfallen alle Kennzahlen gegen
die Ground Truth.

### B2 · Wo die Zahlen stehen

| Stufe | W&B | Dateien |
|---|---|---|
| [1] | – | Label-Datei: `track_id` und Status je Event, dazu die Parameter |
| [2] | `pairs/`, Tabelle `pairs/per_run` | `pairs.npz`: die Paare und ihre Klassenzahlen |
| [3] | Verlauf je Epoche, Endwerte | `model.metrics.json`: Endwerte und Verlauf; `model.pt`: Endwerte |
| [4] | ein Lauf je Ausschnitt | JSON je Ausschnitt |
| [5a] | `solve/`, `gt/`, `diag/` je Fenster; `lauf/` | (A): die Zahlen unter `lauf/` |
| [6a] | `lauf/`, Tabelle `lauf/bilanz` | Artefakt: die Zahlen unter `lauf/` |
| [6b] | Verlauf je Epoche, Endwerte | Checkpoint: die Zahlen unter `data/` und `final/diag/` |
| [6c] | `lauf/` | JSON: alle Zahlen unter `lauf/`; npz: die Partition |
| [7] | `lauf/` | npz: die Bahnen und ihre Zahl |
| STELLA 4.1 | – | JSON mit `--json` |

Die Dateien tragen außerdem die Gruppen ihrer Konfiguration, die meisten auch
ihre Herkunft: Zeitpunkt, Git-Stand, Paketversion, Rechner und Job. Die
wichtigsten Zahlen jeder Stufe stehen zudem in der Konsole. Ohne W&B bleiben
Dateien und Konsole.

### B3 · Kennzahlen je Stufe

#### [2] Paarbau

| Feld | misst |
|---|---|
| `pairs/n_total`, `n_ja`, `n_nein`, `ratio_ja_nein` | Größe und Klassenverhältnis des Datensatzes; das Verhältnis setzen die Deckel. |
| `pairs/n_roh_ja`, `n_roh_nein` | Klassenzahlen vor Dünnen und Deckeln; sie zeigen die Verteilung im Einsatz. |
| `pairs/n_roh_ja_ambig`, `n_roh_nein_ambig` | davon Paare mit mindestens einem mehrdeutigen Endpunkt. |
| `pairs/ambig_anteil_nein` | `n_roh_nein_ambig` / `n_roh_nein`. Mehrdeutige Events sitzen dort, wo Partikel sich nahe kommen. |
| `pairs/n_roh_rausch` | Kanten mit einem Rausch-Endpunkt; verworfen, weil zwei Rauschevents sonst als JA zählten. |
| `pairs/logit_prior_roh` | Logit der rohen Klassenverteilung; zeigt, wie weit der Betriebspunkt vom kalibrierten liegt. |
| `pairs/n_runs`, `n_runs_leer`, `n_runs_gedeckelt` | Runs insgesamt, ohne Ausbeute, am `max_no`-Deckel. |
| `pairs/n_runs_vorrat_verfehlt` | Runs, die `ja_vorrat` nicht erreichen. Soll 0 sein: Ein solcher Run trägt das Verhältnis, das `ja_frac` hergibt. |
| `pairs/n_fenster`, `covered_ms` | abgeschrittene Fenster und ihre Dauer. |
| `pairs/per_run` | Tabelle je Run: Fenster, Klassenzahlen roh und behalten, Deckel, Vorrat, Laufzeit. |
| `pairs/out_path`, `pairs/diag/…` | Ausgabepfad; Laufzeit und Spitzenspeicher. |

#### [3] Training Stufe 1

Je Epoche, für `train/` und `holdout/` gleich aufgebaut:

| Feld | misst |
|---|---|
| `epoch` | Nummer der Epoche, x-Achse aller Kurven. |
| `train/loss_running` | laufendes Mittel des Verlusts über die Minibatches, bei wechselnden Gewichten. |
| `…/loss` | Verlust mit den Gewichten am Ende der Epoche; `train/loss` und `holdout/loss` sind vergleichbar. |
| `…/tpr`, `…/tnr` | Anteil erkannter JA- bzw. NEIN-Paare. |
| `…/accuracy`, `…/balanced_accuracy` | Anteil richtig entschiedener Paare; Mittel aus TPR und TNR. |
| `…/precision`, `…/f1` | Anteil echter JA unter den als JA bewerteten; harmonisches Mittel aus Precision und TPR. |
| `…/auc` | Wahrscheinlichkeit, dass ein zufälliges JA-Paar einen höheren Logit bekommt als ein zufälliges NEIN-Paar. |
| `…/pred_neg_frac` | Anteil negativer Logits, das Gegenstück zu `frac_neg` im Einsatz. |
| `…/cm/tp`, `fp`, `fn`, `tn` | die vier Zählungen bei Schwelle 0. |
| `…/confusion` | dieselbe Matrix als Tabelle, einmal am Ende. |
| `diag/sec_epoch` | Sekunden je Epoche, beide Auswertungen eingeschlossen. |

Einmalig:

| Feld | misst |
|---|---|
| `data/raw/`, `data/train_split/`, `data/holdout/`, `data/train_used/` | Klassen an vier Stellen: ganzer Datensatz, Trainingsseite, Holdout, tatsächlich trainierte Menge; je `n`, `n_ja`, `n_nein`, `ja_frac`, `ratio_ja_nein`. |
| `data/sample_ratio_erreicht` | ob `sample_ratio` erreichbar war. Muss `true` sein, sonst setzt der Datensatz den Betriebspunkt. |
| `data/holdout_runs`, `data/holdout_hinweis` | Runs im Holdout, sie hängen an `training.seed`; ein Klartext zum Holdout. |
| `data/mu/…`, `data/sig/…` | Mittel und Streuung je Merkmal vor der Standardisierung, auf der untersampelten Menge. Bleibt `data/mu/dt` nach einer Änderung von `alpha` gleich, wurde der falsche Datensatz geladen. |
| `data/dataset_neighborhood/…`, `data/dataset_labeling/…`, `data/dataset_label_format` | Nachbarschaft, Labeling und Labelformat, gelesen aus dem Paar-Datensatz und gegen die Konfiguration geprüft. |
| `data/pairs_path`, `data/imbalance` | Paar-Datensatz und Modus. |
| `final/train/…`, `final/holdout/…` | Kennzahlen der letzten Epoche, gegliedert wie der Verlauf. |
| `final/n_params`, `final/val_runs` | Zahl der Gewichte; Holdout-Runs. |
| `final/diag/…` | Gerät, Torch-Version, Laufzeiten, Spitzenspeicher. |
| `model_path` | Pfad des Netzes. |

`evaluate_pairs.py` schreibt dieselben Kennzahlen wie `holdout/` für einen
ganzen Paar-Datensatz, unter dem Split, aus dem er stammt, also etwa `val/`.

#### [4] und [5a] Solve Stufe 1

Je Fenster: in [4] ein W&B-Lauf je Ausschnitt, in [5a] ein Eintrag im Verlauf.

| Feld | misst |
|---|---|
| `solve/t0`, `dt_ms`, `n_edges`, `events_per_ms` | Beginn und Dauer des Fensters, Kanten, Eventdichte. |
| `solve/n_events`, `n_gt` | saubere Events und GT-Partikel im Fenster; auf ihnen rechnen Reinheit, Fragmentierung und `n_clusters`. |
| `solve/frac_neg_roh`, `cost_offset`, `frac_neg` | Anteil negativer Kanten vor dem Versatz, der Versatz, der Anteil danach; 0 zählt als nicht negativ. |
| `solve/mode`, `backend` | Rechnung des Solvers. |
| `solve/dt_mean`, `dr_mean` | Merkmalsmittel aller Kanten, das Gegenstück zu `data/mu` aus [3]. |
| `solve/edge_n_bewertbar`, `edge_bewertbar_frac` | Kanten mit zwei sauberen Endpunkten, absolut und als Anteil. |
| `solve/edge_ja_frac` | unter ihnen der Anteil der Kanten innerhalb eines Partikels. |
| `solve/edge_tp`, `fp`, `fn`, `tn`, `n_nein` | Konfusion der Kanten; Zahl der NEIN-Kanten. |
| `solve/n_ambig`, `ambig_frac` | mehrdeutige Events, absolut und als Anteil. |
| `solve/n_clusters_rama` | Cluster, die der Solver bildet, also Labels ≥ 0, mit beiden Backends. |
| `solve/n_clusters` | davon die Cluster mit mindestens einem sauberen Event. |
| `solve/n_nodes_unassigned` | Knoten ohne Kante. |
| `solve/purity_cluster_mean`, `purity_event_mean` | größter GT-Anteil je Cluster, ungewichtet bzw. nach Clustergröße gewichtet. |
| `solve/purity_lt90`, `purity_lt90_frac` | Cluster unter 0,9 Reinheit, absolut und als Anteil. |
| `solve/frag_mean`, `frag_max` | Cluster je GT-Partikel, Mittel und Maximum. |
| `solve/frag_groesster_anteil_median` | Anteil der Events eines Partikels in seinem größten Fragment, Median über die Partikel. |
| `solve/cluster_size_min`, `cluster_size_median`, `cluster_size_max`, `n_singletons` | Größenverteilung der Cluster; Cluster aus genau einem Event. |
| `solve/merge_n_events`, `merge_n_clusters`, `merge_n_gt` | saubere und mehrdeutige Events, ihre Cluster und ihre GT-Partikel: die Menge, über die die Verschmelzung rechnet. |
| `solve/merge_frac_strikt`, `merge_frac_10` | Anteil dieser Cluster mit mindestens zwei GT-Partikeln bzw. mit mindestens 10 % Minderheit. |
| `solve/merge_event_frac`, `merge_n_minderheit` | Events, die in ihrem Cluster nicht zur Mehrheit gehören, als Anteil und absolut. |
| `solve/merge_n_gt_max` | höchste Zahl von GT-Partikeln in einem Cluster. |
| `solve/nah_n_events`, `fern_n_events`, `nah_n_minderheit`, `fern_n_minderheit` | Events und Minderheits-Events für Partikel mit bzw. ohne nahen Nachbarn; absolut, damit sie sich über Fenster summieren lassen. |
| `gt/events_per_particle_median`, `gt/visible_ms_median` | Events je GT-Partikel und Sichtbarkeitsdauer, Median. |
| `diag/t_…`, `vram_peak_gb`, `max_rss_mb`, `gpu_name` | Laufzeit je Abschnitt, Spitzen von Grafik- und Arbeitsspeicher, Karte. |
| `fenster` | Nummer des Fensters, in [5a]. |

Reinheit, Fragmentierung und `n_clusters` rechnen über die sauberen Events,
die Verschmelzung über saubere und mehrdeutige; `merge_event_frac` und
`purity_event_mean` beziehen sich deshalb auf verschiedene Mengen. Liegt
`purity_event_mean` unter `purity_cluster_mean`, sind die unreinen Cluster
die großen. `frag_groesster_anteil_median` unterscheidet drei gleiche Teile
von einem großen Stück mit Krümeln.

Je Run, in [5a]:

| Feld | misst |
|---|---|
| `lauf/n_events`, `n_fenster`, `n_tracklets` | ON-Events, Fenster und Tracklets des Runs. |
| `lauf/events_je_tracklet`, `tracklets_je_fenster` | daraus abgeleitet. |
| `lauf/n_events_unassigned` | Events ohne Tracklet. |
| `lauf/n_fenster_ohne_nein` | Fenster ohne NEIN-Kante, also ohne Begegnung; −1 ohne GT. |
| `lauf/merge_event_frac` | Anteil verschmolzener Events über den Run. |
| `lauf/nah_merge_event_frac` | dieselbe Rate für Partikel mit nahem Nachbarn: die Kennzahl von Stufe 1. |
| `lauf/fern_merge_event_frac` | dieselbe für Partikel ohne nahen Nachbarn: die Kontrolle, sie muss 0 sein. |
| `lauf/nah_event_anteil`, `nah_n_events`, `fern_n_events`, `nah_schwelle_px` | Größe der beiden Gruppen und die Schwelle; der Anteil ordnet die NAH-Rate ein. |
| `lauf/frac_neg` | Anteil schneidender Kanten über den Run, nach Kanten gewichtet: der Betriebspunkt, wie der Solver ihn sieht. |
| `lauf/frag_mean` | Fragmente je GT-Partikel: Summe der Fragmente durch Summe der Partikelvorkommen. |
| `lauf/frag_groesster_anteil_mittel` | Mittel der Fenster-Mediane, nach `n_gt` gewichtet. |
| `lauf/merge_frac_strikt`, `merge_frac_10` | Mittel der Fensterwerte, nach `merge_n_clusters` gewichtet. |
| `lauf/edge_tp`, `fp`, `fn`, `tn`, `edge_tpr`, `edge_tnr` | Konfusion der Kanten über den Run und die Raten daraus. |
| `lauf/kanten_konfusion` | dieselben Zahlen als Tabelle mit Randsummen. |
| `lauf/diag/sec_total`, `sec_laden`, `max_rss_mb` | Laufzeit, davon Laden, Spitzenspeicher. |
| `lauf/run`, `split`, `tracklet_path` | welcher Run; Pfad von (A). |

In der Summary eines [5a]-Laufs steht unter `solve/` das letzte Fenster;
Werte des ganzen Runs tragen `lauf/`. Die Konfusion der Kanten folgt dem
Vorzeichen der Kosten vor dem Solver, denn RAMA verändert die Kosten beim
Lösen. Zwei Netze vergleicht man bei gleichem `lauf/frac_neg`: Das
Klassenverhältnis im Training verschiebt jeden Logit, derselbe `cost_offset`
ergibt deshalb je Netz einen anderen Betriebspunkt.

#### [6a] Kandidaten

| Feld | misst |
|---|---|
| `lauf/n_tracklets_roh`, `n_knoten`, `knoten_frac` | Tracklets aus (A); davon Knoten über `min_events`, absolut und als Anteil. |
| `lauf/knoten_event_frac`, `n_events_run` | Anteil der zugeordneten Events in behaltenen Knoten; Events des Runs. |
| `lauf/n_on_median` | Events je Knoten, Median. |
| `lauf/n_kandidaten`, `kandidaten_je_knoten` | Kandidatenpaare; 2 · Paare / Knoten. |
| `lauf/grad_median`, `grad_max` | Kandidaten je Knoten. |
| `lauf/n_knoten_ohne_kandidat`, `knoten_ohne_kandidat_frac` | Knoten ohne Kandidatenpaar, absolut und als Anteil; Stufe 2 kann sie mit nichts verbinden. |
| `lauf/paar_dt_median_ms`, `paar_dr_median_px` | zeitlicher Abstand und Zentroidabstand der Paare, Median. |
| `lauf/paar_frac_dt_unter_alpha1` | Anteil der Paare mit Δt unter `neighborhood.alpha`, Näherung für Paare aus demselben Stufe-1-Fenster. |
| `lauf/n_gueltig`, `gueltig_frac`, `ja_frac`, `ratio_ja_nein` | Paare mit Label, fest nach genau einem GT-Partikel auf beiden Seiten, und ihr Klassenverhältnis. |
| `lauf/knoten_rausch_frac`, `knoten_merge_frac` | Knoten ohne GT-Partikel; verschmolzene Knoten mit mindestens zweien. |
| `lauf/gt_anteil_lt90_frac` | Knoten, deren Mehrheit unter 0,9 liegt; zur Diagnose. |
| `lauf/gt_n_partikel`, `gt_tracklets_je_partikel_median` | GT-Partikel mit mindestens einem Knoten; Tracklets je Partikel. |
| `lauf/gt_frac_zusammenhaengend`, `gt_n_zersplittert` | Partikel, deren Tracklets im Kandidatengraphen zusammenhängen, als Anteil; die übrigen absolut. |
| `lauf/gt_komponenten_median`, `_mittel`, `_p95`, `_max` | Komponenten je Partikel. |
| `lauf/bilanz` | Tabelle: Tracklets, Knoten, ohne Kandidat, Paare, mit Label. |
| `lauf/diag/max_rss_mb_laden`, `max_rss_mb_knoten`, `max_rss_mb`, `sec_laden`, `sec_total` | Spitzenspeicher nach dem Laden, nach dem Knotenbau und am Ende; Laufzeiten. |
| `lauf/run`, `split`, `kandidaten_path` | welcher Run; Pfad des Artefakts. |

`gt_frac_zusammenhaengend` ist die Obergrenze von Stufe 2: Was im
Kandidatengraphen nicht zusammenhängt, kann kein Solver verbinden. Die
Erreichbarkeit rechnet über Knoten mit mindestens einem GT-Partikel, die
Label-Maske über Knoten mit genau einem; ein verschmolzenes Tracklet dient so
als Brücke, ohne ein eindeutiges Label zu haben. Das Mittel der Komponenten
tragen wenige Partikel, der Median zeigt den typischen Fall.

#### [6b] Training Stufe 2

Wie [3], mit drei Unterschieden: Die Zählungen stehen flach (`holdout/tp`
statt `holdout/cm/tp`), je Epoche stehen `sec` und `sec_train` statt
`diag/sec_epoch`, und die Mengen unter `data/` heißen anders.

| Feld | misst |
|---|---|
| `sec`, `sec_train` | Sekunden der Epoche gesamt und für den lernenden Durchlauf; die Differenz sind die beiden Auswertungen. |
| `data/roh/`, `data/train_vor/`, `data/val/`, `data/train_je_epoche/` | Klassen an vier Stellen: alle Paare mit Label, Trainingsseite vor der Ziehung, Holdout, je Epoche trainierte Menge. `data/val/` bezeichnet den Holdout. |
| `data/sample_ratio_erreicht` | ob `sample_ratio` einzuhalten war; muss `true` sein. |
| `data/n_knoten`, `data/n_paare` | Knoten und Kandidatenpaare über alle Runs. |
| `data/n_gueltig`, `data/gueltig_frac` | Paare mit Label nach `transformer.label_modus`; bei `mehrheit` mehr als `lauf/n_gueltig` in [6a]. |
| `data/val_runs` | Holdout-Runs; zusammen mit der Run-Liste ist der Split damit reproduzierbar. |
| `data/val_gestutzt_auf`, `data/train_eval_gestutzt_auf` | Zahl der Paare, auf die Holdout und Trainingsmenge für die Auswertung gestutzt werden. |
| `data/n_parameter` | trainierbare Gewichte des Netzes. |
| `final/…` | Kennzahlen der letzten Epoche wie im Verlauf; `final/daten/` wiederholt `data/`. |
| `final/diag/gpu_peak_mb` | Spitze des Grafikspeichers, gemessen nach dem Laden der Daten. |

Überanpassung zeigt der Abstand zwischen `train/` und `holdout/`. Die beiden
Mengen tragen verschiedene Klassenverhältnisse, weil nur die Trainingsseite
untersampelt wird, und `train/` rechnet auf einer Stichprobe; aussagekräftig
ist deshalb die Entwicklung des Abstands über die Epochen. `loss_running`
liegt früh über `loss`, das ist der Fortschritt innerhalb der Epoche.

#### [6c] Solve Stufe 2

Kanten, entnommen vor dem Solver:

| Feld | misst |
|---|---|
| `lauf/n_kanten`, `frac_neg` | Kandidatenpaare, also Kanten; Anteil negativer Kosten. |
| `lauf/kosten_min`, `kosten_median`, `kosten_max` | Verteilung der Kosten. |
| `lauf/edge_n_bewertbar`, `edge_bewertbar_frac` | Kanten, deren beide Knoten genau ein GT-Partikel tragen. |
| `lauf/edge_tp`, `fp`, `fn`, `tn` | Konfusion bei Schwelle 0. |
| `lauf/edge_accuracy`, `edge_precision`, `edge_tpr`, `edge_tnr`, `edge_f1`, `edge_balanced_accuracy`, `edge_pred_neg_frac`, `edge_auc` | daraus abgeleitet, wie in [3]. |

Cluster:

| Feld | misst |
|---|---|
| `lauf/n_knoten`, `n_cluster`, `knoten_je_cluster` | Knoten der Instanz, Cluster mit Label ≥ 0, Tracklets je Trajektorie. |
| `lauf/label_max_roh` | größtes Label des Solvers vor dem Umnummerieren; zeigt den Wertebereich, den die Partition braucht. |
| `lauf/n_events`, `event_ohne_cluster_frac` | ON-Events des Runs; Anteil der Events ohne Cluster (−1). |

`edge_…` ist die Paarebene, zwei Tracklets; die Zählungen der Bewertung über
Events heißen `event_…`.

**Zwei Kennzahlensätze.** `lauf/stufe1/` und `lauf/stufe2/` tragen dieselben
Felder auf derselben Knotenmenge; `stufe1` setzt jedes Tracklet in einen
eigenen Cluster, der Unterschied beider ist das Ergebnis von Stufe 2. Die
Werte unter `stufe1` dienen als Bezug: Weil jedes Tracklet ein eigener Cluster
ist, fallen `ass_re`, `idf1`, `idsw` und `lhn_frag` dort per Definition
schlecht aus. Jeder Satz enthält die Cluster- und Verschmelzungsfelder aus
[4] (`purity_…`, `frag_…`, `cluster_size_…`, `n_singletons`, `n_clusters`,
`n_gt`, `n_events`, `merge_…`, `gt_events_per_particle_median`,
`gt_visible_ms_median`) und die folgenden.

HOTA (Luiten et al. 2021), gemessen über den vollen ON-Strom des Runs:

| Feld | misst |
|---|---|
| `event_tp` | Events mit GT-Partikel in einem Cluster. |
| `event_fn` | Events mit GT-Partikel, die in keinem Cluster liegen. |
| `event_fp` | Rauschevents in einer Trajektorie; mehrdeutige Events gelten als GT-Detektionen. |
| `det_re`, `det_pr`, `det_a` | Detektion: Recall, Precision und Jaccard. |
| `ass_re`, `ass_pr`, `ass_a` | Assoziation: `ass_re` fällt, wenn ein Partikel in mehrere Cluster zerfällt, `ass_pr`, wenn ein Cluster mehrere Partikel umfasst; `ass_a` ist ihr Jaccard, gemittelt über die Detektionen. |
| `hota` | Wurzel aus `det_a` · `ass_a`. |
| `ass_a_min`, `ass_a_p10`, `ass_a_lt90`, `ass_a_lt90_frac` | schlechtestes Partikel, 10. Perzentil, Partikel unter 0,9 absolut und als Anteil. |
| `ass_pr_min`, `ass_pr_lt90` | dieselbe Sicht auf die Verschmelzungsseite. |
| `n_cluster_eff_max` | schlimmstes Partikel: effektive Zahl seiner Cluster, Kehrwert des Simpson-Index. |

IDF1 (Ristani et al. 2016):

| Feld | misst |
|---|---|
| `idtp`, `idfn`, `idfp` | Events in der eineindeutig zugeordneten Überlappung von GT und Clustern; die übrigen GT- bzw. Cluster-Events. |
| `idr`, `idp`, `idf1` | Recall, Precision und ihr harmonisches Mittel. |

Die Überlappungsmatrix ist dicht, GT-Partikel mal Cluster, und bestimmt den
Spitzenspeicher von [6c].

IDSW (Bernardin und Stiefelhagen 2008) und Frag (Li, Huang und Nevatia 2009):

| Feld | misst |
|---|---|
| `delta_us` | Rasterweite in µs, abgeleitet aus dem GT-Raster des Datensatzes. |
| `n_metric_zeiteinheiten` | bewertete Zeiteinheiten. |
| `idsw`, `idsw_max`, `idsw_ptkl_betroffen` | Identitätswechsel: gesamt, schlimmstes Partikel, betroffene Partikel. |
| `lhn_frag`, `lhn_frag_max`, `lhn_frag_ptkl_betroffen` | Unterbrechungen der Verfolgung ab einer ganzen Rastereinheit, ebenso aufgeteilt. |

`idsw` zählt Wechsel der Zuordnung über die Zeit. Liegen zwei Partikel über
den ganzen Run in einem Cluster, wechselt nichts, und `idsw` bleibt 0;
Verschmelzung zeigen `merge_event_frac`, `ass_pr` und `n_clusters` gegen
`n_gt`.

FA-HOTA (Luiten et al. 2021, Gl. 32 bis 34), gerastert auf `delta_us`:

| Feld | misst |
|---|---|
| `fa_hota` | HOTA, jeder TP mit der Wurzel aus A · F statt mit A gewichtet. |
| `hota_einheiten` | HOTA auf demselben gerasterten Strom, der Bezugswert. |
| `hota_frag_a` | mittleres F: Anteil der Assoziation im zusammenhängenden Stück. |
| `hota_n_fragmente`, `hota_fragment_len_median`, `hota_fragment_len_max` | ununterbrochene Läufe über alle Paare aus Partikel und Cluster; ihre Länge in Zeiteinheiten. |

Den Abzug für Fragmentierung zeigt `fa_hota` / `hota_einheiten`; mit `hota`
ist `fa_hota` nicht vergleichbar, weil dort Events die Detektionen sind und
hier Zeiteinheiten. Der Median der Fragmentlängen ist nach Anzahl gewichtet,
viele kurze Stücke drücken ihn auf 1. `fa_hota` liest sich deshalb zusammen
mit `hota`, `idsw_ptkl_betroffen` und `merge_event_frac`.

Dazu `lauf/run` und `split`, unter `lauf/diag/` die Laufzeit bis zu den
Kosten (`sec_kosten`), die Gesamtzeit und den Spitzenspeicher, sowie die
Pfade `solve2_path` und `trajektorien_path`.

#### [7] Fit

| Feld | misst |
|---|---|
| `lauf/n_bahnen_partition`, `n_bahnen_gefittet` | Bahnen in der Partition aus [6c]; davon gefittet. Der Rest war zu kurz, hatte zu wenige Events oder, mit `beta` > 0, keine bestimmbare Querstreuung. |
| `lauf/n_punkte` | Stützpunkte im Artefakt. |
| `lauf/u_median_px_ms` | Median des Geschwindigkeitsbetrags über alle Stützpunkte; ohne GT die einzige Zahl zum Ergebnis. |
| `lauf/gt/n_bahnen_bewertet` | Bahnen mit GT-Partikel und genug Rasterpunkten. |
| `lauf/gt/n_bahnen`, `n_bahnen_rein` | Bahnen in den Werten ohne bzw. mit Suffix `_rein`. |
| `lauf/gt/medae_xy` | Ortsfehler gegen die Boxmitte, px. |
| `lauf/gt/medae_u`, `medae_u_rel` | Geschwindigkeitsfehler, px/ms, und relativ zu `neighborhood.v_max`. |
| `lauf/gt/du_signiert` | Betrag der Fit-Geschwindigkeit minus Betrag der GT-Geschwindigkeit, px/ms, mit Vorzeichen. |
| `lauf/gt/du_neg_frac` | Anteil zu langsamer Bahnen; bei reinem Rauschen 0,5. |
| `lauf/gt/v_gt_median` | Median der GT-Geschwindigkeit, die Bezugsgröße. |
| `lauf/diag/sec_total`, `max_rss_mb`, `fit_path` | Laufzeit, Spitzenspeicher, Pfad des Artefakts. |

Jeder Wert unter `gt/` ist ein Median über Bahn-Mediane, einmal über alle
zugeordneten Bahnen und einmal mit `_rein` über die unverschmolzenen, deren
Mehrheit mindestens 0,99 der sauberen Events stellt. `medae_u` wächst mit der
Geschwindigkeit eines Runs; Runs lassen sich deshalb nur mit `v_gt_median`
daneben vergleichen. Die mit STELLA vergleichbaren Werte liefert die
Bewertung im nächsten Abschnitt.

#### Bewertung nach STELLA

`tools/stella_bewertung.py` bewertet die Fit-Dateien aus [7] nach
Abschnitt 4.1 von STELLA, ohne W&B. Die Werte stehen je Run unter
`je_run/<N>/` und gepoolt unter `gesamt/`, daneben `solves`, `split` und
`stamm`. Definition und Auslegung stehen in `stella_fehler` in
`src/eventpiv/bewerten/evaluate_trajektorien.py`.

| Feld | misst |
|---|---|
| `medae_xy`, `p25_xy`, `p75_xy` | Median und Quartile des Ortsfehlers über alle zugeordneten Punkte, px. |
| `medae_u`, `p25_u`, `p75_u` | dasselbe für die Geschwindigkeit, px/ms, über die Punkte, deren GT eine Geschwindigkeit trägt. |
| `medae_u_rel` | `medae_u` relativ zur festen Geschwindigkeitsnormierung des Papiers, `U_NORM_STELLA`. |
| `medae_xy_achse`, `p25_xy_achse`, `p75_xy_achse`, `medae_u_achse`, `p25_u_achse`, `p75_u_achse`, `medae_u_achse_rel` | dieselben Fehler je Achse gepoolt. |
| `medae_u_betrag`, `p25_u_betrag`, `p75_u_betrag`, `medae_u_betrag_rel` | der Fehler im Betrag der Geschwindigkeit. |
| `laengs_median` | Ortsfehler mit Vorzeichen in GT-Bewegungsrichtung, px; positiv heißt voraus. Diagnose zur Positionskorrektur. |
| `t_covered` | Anteil der GT-Punkte, die eine Bahn abdeckt. |
| `n_tracks`, `n_gt`, `n_tracks_je_gt` | behaltene Bahnen, GT-Partikel und ihr Verhältnis. |
| `verworfen_frac` | Anteil der Bahnen, den die Regel aus STELLA, Abschnitt 4.1, verwirft. |
| `zugeordnet_frac`, `n_punkte_zugeordnet` | Anteil der Bahnpunkte mit zugeordnetem GT-Partikel; Zahl der Punkte in den Medianen. |
| `box_w_median` | je Run: Median der Boxbreite, px. Diagnose zur Positionskorrektur. |

Maßgeblich ist die Norm des Fehlers, `medae_xy` und `medae_u`; `_achse` und
`_betrag` sind zwei weitere Lesarten. Gepoolt heißt über Punkte: `gesamt` ist
der Median über alle Punkte aller Runs.

### B4 · W&B

Projekt `event-piv-multicut`. Ein Lauf entspricht in [2] dem Paarbau, in [3]
und [6b] einem Training, in [4] einem Ausschnitt und ab [5a] einem Run;
`group` ist der Slug von neighborhood. `WANDB_MODE=disabled` schaltet W&B
ab, `offline` schreibt nur lokal. `WANDB_NOTES` beschriftet jeden Lauf,
`WANDB_SILENT=true` nimmt die Meldungen von W&B aus der Konsole. Nach W&B
gehen Kennzahlen und Konfigurationen, keine Rohdaten.

---

## Literatur

- Bernardin, K. und Stiefelhagen, R.: *Evaluating Multiple Object Tracking
  Performance: The CLEAR MOT Metrics.* EURASIP Journal on Image and Video
  Processing, 2008.
- Kalman, R. E.: *A New Approach to Linear Filtering and Prediction
  Problems.* Journal of Basic Engineering 82, 1960.
- Li, Y., Huang, C. und Nevatia, R.: *Learning to Associate: HybridBoosted
  Multi-Target Tracker for Crowded Scene.* CVPR 2009.
- Luiten, J. et al.: *HOTA: A Higher Order Metric for Evaluating Multi-Object
  Tracking.* International Journal of Computer Vision 129, 2021.
- Rauch, H. E., Tung, F. und Striebel, C. T.: *Maximum Likelihood Estimates of
  Linear Dynamic Systems.* AIAA Journal 3, 1965.
- Ristani, E. et al.: *Performance Measures and a Data Set for Multi-Target,
  Multi-Camera Tracking.* ECCV Workshops 2016.
- STELLA (Sachs et al. 2026) und RAMA (Abbas und Swoboda 2022): siehe README,
  Fremdcode und Referenzen.
