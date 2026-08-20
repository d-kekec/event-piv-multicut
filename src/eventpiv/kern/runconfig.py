#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
runconfig.py -- Laufparameter als Gruppen-Dataclasses.

PARAMETER, nicht Pfade. config.py bleibt die einzige Stelle mit Pfadwissen:
hier steht, WAS gerechnet wird, dort, WOHIN es geschrieben wird.

Der Gruppenschnitt folgt dem UEBEREINSTIMMUNGS-Kriterium: eine Gruppe ist,
was zwischen zwei Stufen gleich sein MUSS. Nur `Neighborhood` muss das, und
nur sie wird verglichen.

KEIN frozen=True -- OmegaConf macht den Container dann read-only und jeder
Hydra-Override scheitert. GEMESSEN, bei einem OmegaConf-Wechsel neu
pruefen.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields

BOX_W = 11.0                    # px, Kantenlaenge der quadratischen
                                # GT-Bounding-Box. Konstante des Datensatzes.

_CAST = {"int": int, "float": float, "str": str, "bool": bool}


class _Typed:
    """Feldwerte auf ihren annotierten Typ bringen. 

    Ohne diesen Cast bekaeme derselbe Arbeitspunkt zwei Verzeichnisse:
    r_max=3 und r_max=3.0 ergeben verschiedene kanonische Formen. Ueber
    Hydra faellt das nie auf (OmegaConf castet gegen das Schema), bei
    direkter Konstruktion in Tests und Auswertungsskripten schon.

    `from __future__ import annotations` macht f.type zu einem STRING;
    "int | None" wird am Trenner zerlegt. Listen (list[int]) stehen nicht in
    _CAST und bleiben unangetastet.
    """

    def __post_init__(self) -> None:
        for f in fields(self):
            v = getattr(self, f.name)
            if v is None:
                continue
            cast = _CAST.get(str(f.type).split("|")[0].strip())
            if cast is None or isinstance(v, cast):
                continue
            if cast is bool and isinstance(v, str):
                # bool("false") ist True -- der Cast wuerde die Bedeutung
                # still umdrehen.
                raise TypeError(
                    f"{type(self).__name__}.{f.name}={v!r}: Wahrheitswert als "
                    "Zeichenkette. bool(str) ist immer True -- echten "
                    "Wahrheitswert uebergeben.")
            setattr(self, f.name, cast(v))


# ============================================================
# Gruppen
# ============================================================

# Erlaubte Werte, HIER und nicht in graph.py / model.py / train.py:
# runconfig darf weder scipy noch torch importieren, und die Pruefung muss
# greifen, BEVOR der Slug existiert.
NEIGHBORHOOD_GEOMS = ("ellipsoid", "cylinder", "hyperboloid")
POLARITIES = ("on", "both")
FEATURE_MODES = ("dr", "absxy")
IMBALANCE_MODES = ("sample", "weight")
SOLVER_BACKENDS = ("rama", "cc")
SOLVER_MODES = ("P", "PD", "PD+", "D")
POOLINGS = ("cls", "mean")
LABEL_MODI = ("streng", "mehrheit", "weich")

@dataclass
class Neighborhood(_Typed):
    """DER VERTRAG -- und zugleich die Identitaet des Arbeitspunkts.

    v_max wirkt nur bei geom="hyperboloid", gehoert aber hierher: er
    entscheidet dort mit, welche Kanten existieren. Ueberkonservativ ist
    zulaessig, eine bedingte Identitaetsregel waere eine spaetere Falle.

    polarity waehlt die KNOTENMENGE: "on" laesst nur positive Events zu,
    "both" alle. Es gehoert aus demselben Grund hierher wie v_max, nur
    zwingender -- der Filter wirkt in Stufe [2] und in Stufe [4], und
    stimmen sie nicht ueberein, sieht das Netz im Einsatz eine andere
    Kantenpopulation als im Training. Bei r_max = 3 sind 13,8 % der Kanten
    gemischtpolar; ohne dieses Feld wuerde ein mit beiden Polaritaeten
    gebauter Datensatz still fuer einen ON-only-Einsatz weiterverwendet.
    """
    # Arbeitspunkt von Stufe 1. alpha ist zugleich die Laenge eines
    # Tracklet-Fensters -- ein Regler fuer Kantenreichweite UND
    # Tracklet-Laenge, siehe Code_Guide §1.
    alpha: int = 10_000
    r_max: float = 5.0
    geom: str = "cylinder"
    v_max: float = 1.02e-4
    polarity: str = "on"

    def short(self) -> str:
        # polarity immer im Kern: das Feld aendert den Hash ohnehin, ein
        # unterdruecktes Suffix erspart also keinen Umzug.
        return (f"a{self.alpha}_rmax{_num(self.r_max)}_{self.geom}"
                f"_{self.polarity}")

    def __post_init__(self) -> None:
        super().__post_init__()          # Typcast aus _Typed zuerst
        # Hier und nicht erst in graph.neighborhood_pairs: dort ist der Slug
        # laengst berechnet und ein Verzeichnis danach benannt.
        if self.geom not in NEIGHBORHOOD_GEOMS:
            raise ValueError(
                f"neighborhood.geom={self.geom!r} unbekannt. "
                f"Erlaubt: {NEIGHBORHOOD_GEOMS}")
        if self.polarity not in POLARITIES:
            raise ValueError(
                f"neighborhood.polarity={self.polarity!r} unbekannt. "
                f"Erlaubt: {POLARITIES}")

@dataclass
class PairSampling(_Typed):
    """Identitaet des Paar-Datensatzes. `split` ist hier die QUELLE der Paare
    -- nicht zu verwechseln mit Selection.split (was geloest wird).

    `runs` sind AUSSCHLIESSLICH Overlap-Runs. Saubere Runs liefern
    physikalisch null NEIN, und die Overlap-Runs tragen selbst reichlich JA
    (gemessen 94:1 bis 307:1). Beide Klassen stammen damit aus DENSELBEN
    Fenstern -- nur so sieht das Netz eine Eingangsverteilung und nicht zwei.

    `ja_frac` ist keine Feinabstimmung, sondern die Speicherbedingung: ein
    Run liefert bei voller Abdeckung ueber eine Milliarde JA-Paare. NEIN
    wird vollstaendig behalten (bis `max_no`), es ist die knappe Klasse.

    Das Klassenverhaeltnis setzt `ja_vorrat`, nicht ein JA-Deckel. Eine
    obere Schranke fuer JA folgt daraus von selbst: n_nein <= max_no, also
    max_ja <= ja_vorrat * max_no.

    KEIN Sammelfenster mehr: gesammelt wird in anstossenden Fenstern der
    Laenge `neighborhood.alpha` ueber den ganzen Run -- genau die Fenster,
    die Stufe 1 im Einsatz bildet. Und keine Schrittweite: die NEIN-Paare
    sitzen in wenigen Fenstern (Median 29 je Fenster gegen Maximum 91 172
    auf run61), ein Auslassen waere nicht Stichprobe, sondern Verzerrung.
    """
    split: str = "train"
    runs: list[int] = field(
        default_factory=lambda: [2, 11, 19, 30, 32, 35, 44, 59, 61, 76])
    ja_frac: float = 1e-2
    max_no: int = 1_000_000
    # Groesser als training.sample_ratio: [2] legt Vorrat an, [3] setzt den
    # Arbeitspunkt.
    ja_vorrat: float = 4.0
    seed: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.ja_vorrat <= 0:
            raise ValueError(f"pairs.ja_vorrat={self.ja_vorrat} muss > 0 sein.")
        if not 0.0 <= self.ja_frac <= 1.0:
            raise ValueError(
                f"pairs.ja_frac={self.ja_frac} muss in [0, 1] liegen.")

    def short(self) -> str:
        # "rn" statt "r" fuer die ANZAHL Runs: "r" bezeichnet im Projekt
        # sonst einen Radius (r_max) oder einen einzelnen Run (run32).
        # "split_" ausgeschrieben, damit "train" nicht wie ein Name aussieht.
        # Der Hash bleibt unberuehrt -- short() ist eine Methode und steht
        # nicht in asdict(), also nicht in canonical().
        return f"split_{self.split}_rn{len(self.runs)}"


@dataclass
class Training(_Typed):
    feature_mode: str = "dr"
    imbalance: str = "sample"
    sample_ratio: float = 3.0
    hidden: int = 8
    epochs: int = 30
    batch: int = 65_536
    lr: float = 1e-2
    val_frac: float = 0.2
    seed: int = 0

    def short(self) -> str:
        return f"{self.feature_mode}_{self.imbalance}"
    
    def __post_init__(self) -> None:
        super().__post_init__()          # Typcast aus _Typed zuerst
        # Wie bei Neighborhood.geom: bis train_edge_nn es merkt, steht der
        # Slug schon und ein Verzeichnis heisst danach.
        if self.feature_mode not in FEATURE_MODES:
            raise ValueError(
                f"training.feature_mode={self.feature_mode!r} unbekannt. "
                f"Erlaubt: {FEATURE_MODES}")
        if self.imbalance not in IMBALANCE_MODES:
            raise ValueError(
                f"training.imbalance={self.imbalance!r} unbekannt. "
                f"Erlaubt: {IMBALANCE_MODES}")


@dataclass
class Solver(_Typed):
    """Erzwingt keinen Neubau und wird nie verglichen.

    backend waehlt zwischen RAMA und der CC-Baseline. Das ist keine
    Optimierung, sondern der Vergleichspunkt, der den Solver rechtfertigt:
    CC loescht alle negativen Kanten und nimmt die Zusammenhangskomponenten,
    verschmilzt also zwei Wolken, sobald EINE Kante zwischen ihnen positiv
    ist. Multicut darf von der Kantenpraeferenz abweichen, wenn die Bilanz
    es verlangt. Der Abstand zwischen beiden ist die Zahl, die in die Arbeit
    gehoert -- und fuer Stufe 1 zaehlt sie doppelt, weil Verschmelzen dort
    der einzige unumkehrbare Fehler ist.

    mode wirkt nur bei backend="rama"; ueberkonservativ im Namen zu
    bleiben ist billiger als eine bedingte Regel.

    cost_offset ist der BETRIEBSPUNKT, kein Kalibrierwert. Er wird auf jede
    Kante addiert und verschiebt damit die Schwelle zwischen Schneiden und
    Verbinden, ohne die Rangfolge der Kanten zu aendern.

    VORGABE -0,5, gemessen und nicht gewaehlt: bei 0,0 bleiben 4 von 9
    Partikeln auf run61 unerreichbar, weil ihre Tracklets in mindestens
    einem Fenster unter min_events fallen; bei -0,5 sind es 0 von 9 ueber
    alle geprueften Runs, bei viermal so vielen NEIN-Paaren fuer Stufe 2.
    Die Fragmentierung steigt dabei um 37 %, und genau die nimmt Stufe 2
    wieder zurueck (gemessen 1733 -> 1,13 auf val/run1). 0,0 ist damit
    kein Arbeitspunkt der Kette mehr, sondern eine Ablation -- und muss
    ausgeschrieben werden, wie jede Ablation.

    Denselben Effekt hat das Klassenverhaeltnis im Training -- Untersampeln
    auf P(JA)=0.75 verschiebt jeden gelernten Logit gegenueber P(JA)=0.5 um
    logit(0.75). Der Unterschied ist der Preis: das Verhaeltnis kostet ein
    Neutraining, dieser Wert einen Override.

    Erwartetes Vorzeichen NEGATIV: Verschmelzen ist in Stufe 1 der einzige
    unumkehrbare Fehler, Zerschneiden kostet nur Knoten. Die Schwelle darf
    deshalb zum Schneiden hin liegen. Ein Versatz auf den kalibrierten Wert
    des Einsatzes (gemessen logit(P) = +5,6 bis +7,5 ueber ganze Runs)
    machte praktisch jede Kante positiv und verschmoelze alles.
    """
    mode: str = "P"
    use_knn: bool = False
    k_max: int = 24
    backend: str = "rama"
    cost_offset: float = -0.5

    def short(self) -> str:
        # Kein Suffix im Vorgabefall, damit bestehende Dateinamen bleiben.
        # Sobald ein Feld WIRKT, muss es im Namen stehen -- sonst
        # ueberschriebe ein Sweep seine eigenen Ergebnisse.
        return (self.mode
                + (f"_knn{self.k_max}" if self.use_knn else "")
                + ("" if self.backend == "rama" else f"_{self.backend}")
                + ("" if self.cost_offset == 0.0
                   else f"_o{self.cost_offset:g}"))

    def __post_init__(self) -> None:
        super().__post_init__()          # Typcast aus _Typed zuerst
        if self.backend not in SOLVER_BACKENDS:
            raise ValueError(
                f"solver.backend={self.backend!r} unbekannt. "
                f"Erlaubt: {SOLVER_BACKENDS}")
        if self.mode not in SOLVER_MODES:
                raise ValueError(
                    f"solver.mode={self.mode!r} unbekannt. "
                    f"Erlaubt: {SOLVER_MODES}")


@dataclass
class Features(_Typed):
    """Identitaet der Merkmalstabelle (Artefakt B).

    off_s und off_k sind die Parameter der OFF-Zuordnung und damit das
    Einzige, was zwei Tabellen desselben Tracklet-Artefakts unterscheidet.
    off_s = 0,1 ist gemessen, nicht gewaehlt -- Journal 09.08.2026.
    """
    off_s: float = 0.1
    off_k: int = 8

    def short(self) -> str:
        return f"offs{_num(self.off_s)}_k{self.off_k}"


@dataclass
class Kandidaten(_Typed):
    """Identitaet des Stufe-2-Datensatzes: welche Knoten, welche Kanten.

    Von `Transformer` getrennt nach dem UEBEREINSTIMMUNGS-Kriterium, aus
    demselben Grund wie PairSampling von Training: der Datensatz kostet
    Minuten je Run, das Netz ist ein Sweep. In EINER Gruppe erzwaenge jedes
    neue d_model einen Neubau der Kandidaten.

    alpha/r_max/geom/v_max sind bewusst NICHT `Neighborhood`. Sie messen
    hier den Abstand zweier TRACKLET-ZENTROIDE, nicht zweier Events, und
    `polarity` hat auf Zentroiden keinen Sinn. Eine verschachtelte Gruppe,
    deren Haelfte nicht gilt, waere eine spaetere Falle.

    alpha = 5x Stufe-1-alpha, AUSGEFAHREN und nicht gesetzt: die
    Erreichbarkeit auf run61 steigt 95,6 -> 98,9 -> 100,0 % bei 30/40/50 ms
    und bleibt bei 60 ms stehen. 50 000 µs ist der Saettigungspunkt; 60 000
    kauft nichts mehr fuer 20 % mehr Paare. Physikalisch: ein Partikel darf
    vier aufeinanderfolgende Fenster unter min_events verschwinden und wird
    trotzdem weiterverbunden. (Journal 14.08.2026)

    r_max ist GEMESSEN WIRKUNGSLOS und bleibt bei 15 px. 15, 20 und 25 px
    liefern dieselbe Erreichbarkeit und dieselbe Komponentenzahl, bei nur
    4 % mehr Kandidaten fuer 67 % mehr Radius: die Nachbarschaft ist
    ZEITbegrenzt, nicht raumbegrenzt. Der frueher hier vermutete Engpass
    (Wolkenausdehnung 11 px plus Drift 3 px gegen 15 px) existiert nicht.
    geom bleibt hyperboloid -- Taille = Ausdehnung, Aufweitung = Drift.

    min_events = 15 und n_events = 32 sind gemessen (Journal 14.08.2026):
    unter 15 liegt das Mittel bei 1,8 Events, und 92,0 % der behaltenen
    Tracklets tragen mindestens 32, aber nur 68,3 % mindestens 64.
    """
    split: str = "train"
    runs: list[int] = field(
        default_factory=lambda: [2, 11, 19, 30, 32, 35, 44, 59, 61, 76])
    alpha: int = 50_000
    r_max: float = 15.0
    geom: str = "hyperboloid"
    v_max: float = 1.02e-4
    min_events: int = 15
    n_events: int = 32
    seed: int = 0

    def short(self) -> str:
        # "_r" stand hier ZWEIMAL mit verschiedener Bedeutung -- einmal fuer
        # die Anzahl Runs, einmal fuer den Radius. Jetzt "rn" fuer die
        # Anzahl, "rmax" fuer den Radius.
        return (f"split_{self.split}_rn{len(self.runs)}_a{self.alpha}"
                f"_rmax{_num(self.r_max)}_n{self.n_events}")

    def __post_init__(self) -> None:
        super().__post_init__()          # Typcast aus _Typed zuerst
        if self.geom not in NEIGHBORHOOD_GEOMS:
            raise ValueError(
                f"kandidaten.geom={self.geom!r} unbekannt. "
                f"Erlaubt: {NEIGHBORHOOD_GEOMS}")
        if self.min_events < 2:
            # Unter zwei Events hat ein Tracklet keine Ausdehnung und
            # traegt keine Form -- dann ist der Transformer das falsche
            # Werkzeug.
            raise ValueError(
                f"kandidaten.min_events={self.min_events} -- mindestens 2.")
        if self.n_events < 1:
            raise ValueError(
                f"kandidaten.n_events={self.n_events} -- mindestens 1.")


@dataclass
class Transformer(_Typed):
    """Identitaet des Stufe-2-Netzes.

    Tokens sind EVENTS: 2*n_events Event-Tokens plus ein CLS-Token. Das
    Segment-Token ist ein Etikett, das auf jedes Token seines Tracklets
    ADDIERT wird -- es fasst nichts zusammen.

    KEIN Positional Encoding im Sinne eines Sequenzindex: die Tokens
    tragen ihre Koordinaten selbst, und die Reihenfolge der Stichprobe
    darf das Ergebnis nicht aendern. Attention ist permutationsinvariant,
    und das ist hier die richtige Eigenschaft.

    pooling waehlt, was in den Kopf geht: "cls" liest das CLS-Token,
    "mean" mittelt ueber die ECHTEN Tokens. Als Feld und nicht als
    Konstante, weil der Unterschied bei 65 Tokens vermutlich klein ist --
    NICHT gemessen, und eine billige Ablation.

    label_modus entscheidet, welche Paare Gradienten bekommen -- eine
    TRAININGS-, keine Datensatzentscheidung, deshalb hier und nicht in
    Kandidaten. Das Artefakt traegt gt_track_id, gt_anteil und n_gt je
    Knoten; Ziel und Maske entstehen daraus zur Trainingszeit.
      streng    nur Knoten mit genau einem GT-Partikel (Entwurf, Vorgabe).
                Der interessanteste Fall wird damit nie geuebt.
      mehrheit  auch verschmolzene, hart nach Mehrheit, ab gt_min_anteil.
      weich     auch verschmolzene, Zielwert zwischen 0 und 1. Der Logit
                lernt dessen Log-Odds; bei 0,5 ist er null und RAMA sieht
                eine Kante ohne Praeferenz.

    Kein feature_mode: die Baseline bekommt rohe, je Paar normierte
    (t, x, y). Ob eine Merkmalstabelle noetig ist, soll dieser Lauf
    beantworten.

    lr = 1e-3 statt der 1e-2 von Stufe 1: dort haengen 41 Parameter, hier
    Zehntausende. Nicht durchsucht.
    """
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.0
    pooling: str = "cls"
    label_modus: str = "streng"
    gt_min_anteil: float = 0.9
    imbalance: str = "sample"
    sample_ratio: float = 3.0
    epochs: int = 30
    batch: int = 1024
    lr: float = 1e-3
    val_frac: float = 0.2
    seed: int = 0

    def short(self) -> str:
        return (f"d{self.d_model}_l{self.n_layers}_{self.pooling}"
                f"_{self.imbalance}_{self.label_modus}")

    def __post_init__(self) -> None:
        super().__post_init__()          # Typcast aus _Typed zuerst
        if self.pooling not in POOLINGS:
            raise ValueError(
                f"transformer.pooling={self.pooling!r} unbekannt. "
                f"Erlaubt: {POOLINGS}")
        if self.label_modus not in LABEL_MODI:
            raise ValueError(
                f"transformer.label_modus={self.label_modus!r} unbekannt. "
                f"Erlaubt: {LABEL_MODI}")
        if self.imbalance not in IMBALANCE_MODES:
            raise ValueError(
                f"transformer.imbalance={self.imbalance!r} unbekannt. "
                f"Erlaubt: {IMBALANCE_MODES}")
        if self.d_model % self.n_heads:
            # Die Kopfbreite ist d_model/n_heads. Geht das nicht auf,
            # bricht torch erst beim Bau des Netzes ab -- da steht der Slug
            # schon und ein Verzeichnis heisst danach.
            raise ValueError(
                f"transformer.d_model={self.d_model} ist nicht durch "
                f"n_heads={self.n_heads} teilbar.")


@dataclass
class Solver2(_Typed):
    """Der Solver von Stufe 2. Eigene Gruppe, nicht Solver wiederverwendet.

    `solver` identifiziert den Arbeitspunkt von STUFE 1 und steht im
    Dateinamen der Tracklets und Kandidaten -- cost_offset=-0.5 waehlt dort
    die QUELLE. Dieselbe Gruppe fuer die Rueckseite von Stufe 2 zu nehmen
    vermengte zwei Dinge, die getrennt gewaehlt werden muessen.

    backend="cc" ist ein ERGEBNIS, kein Hilfsmittel: CC kann eine positive
    Kante nie schneiden, zwei Trajektorien verschmelzen also, sobald EINE
    Kante zwischen ihnen positiv ist. Der Abstand zu RAMA ist die Zahl, die
    den Solver rechtfertigt.

    cost_offset wirkt hier wie in Stufe 1 -- er wird auf jede Kante addiert
    und verschiebt den Betriebspunkt Richtung Schneiden. NICHT gemessen auf
    dieser Ebene.
    """
    backend: str = "rama"
    mode: str = "P"
    cost_offset: float = 0.0

    def short(self) -> str:
        return (f"{self.backend}{self.mode if self.backend == 'rama' else ''}"
                + ("" if self.cost_offset == 0.0
                   else f"_o{self.cost_offset:g}"))

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.backend not in SOLVER_BACKENDS:
            raise ValueError(
                f"solver2.backend={self.backend!r} unbekannt. "
                f"Erlaubt: {SOLVER_BACKENDS}")
        if self.mode not in SOLVER_MODES:
            raise ValueError(
                f"solver2.mode={self.mode!r} unbekannt. "
                f"Erlaubt: {SOLVER_MODES}")


@dataclass
class Labeling(_Typed):
    """Stufe [1]. Bewusst OHNE Zeitfenster: Labels gelten IMMER fuer den
    ganzen Run. Ein Teilfenster bricht spaeter in evaluate.align_gt_window
    ("Wurde die Label-Datei ueber den GANZEN Run erzeugt?")."""
    r_assign: float = 8.0
    r_ambig: float = 2.0
    box_w: float = BOX_W


@dataclass
class Selection(_Typed):
    split: str = "train"
    run_nr: int | None = None
    runs: list[int] = field(default_factory=list)


@dataclass
class Window(_Typed):
    """Der AUSSCHNITT: welcher Teil eines Runs betrachtet wird.

    Nicht zu verwechseln mit `Neighborhood.alpha`: das ist die Laenge eines
    TRACKLET-Fensters und zugleich ein harter Schnitt, ueber den keine
    Kante gehen darf. `Window` schneidet dagegen nur aus, was betrachtet
    wird.

    None heisst "ab Anfang" bzw. "ganzer Run" -- load_events kennt das
    ohnehin. Sinnvoll bei animate_run und beim Abschreiten eines Runs; ein
    ganzer Run in EINER Solve-Instanz waere ein VRAM-Selbstmord.
    """
    
    # Vorgabe ist der alte Ankerpunkt und heute vor allem fuer Tests
    # brauchbar: jeder rechnende Aufrufer setzt seinen Ausschnitt selbst,
    # der Treiber auf None/None (ganzer Run).
    t0: int | None = 1_700_000
    dt: int | None = 20_000

    def short(self) -> str:
        return f"t{self.t0}_dt{self.dt}"


@dataclass
class Compute(_Typed):
    """Ressourcen -- NIE Teil einer Artefakt-Identitaet."""
    device: str = "cuda"
    device_index: int = 0
    cost_batch: int = 4_000_000

@dataclass
class IO(_Typed):
    """Ein-/Ausgabe und Notausgaenge.

    NIE Teil einer Artefakt-Identitaet -- deshalb eine eigene Gruppe, die
    identity_meta() nicht anfasst. Felder, die eine Stufe nicht braucht,
    bleiben auf ihrem Default; das ist unschaedlich, weil aus dieser Gruppe
    nichts in Metadaten oder Pfade wandert.

    model_path ist der Notausgang auf dem LESEpfad: ein fremdes Netz laden,
    dessen Parameter nicht aus der Konfiguration folgen. Bewusst ein
    sichtbarer Akt, wie strict=false.

    train_fest nennt Runs, die NICHT in den Holdout gezogen werden duerfen.
    Die Auswahl ist sonst gleichverteilt, und die NEIN-reichen Runs tragen
    den groessten Teil der knappen Klasse -- fallen zwei davon heraus, misst
    die Holdout-TNR etwas anderes als beabsichtigt. Die GROESSE des Holdouts
    bleibt val_frac ueber ALLE Runs; eingeschraenkt ist nur, woraus gezogen
    wird. Leer heisst: keine Einschraenkung.

    Es AENDERT das Ergebnis und steht trotzdem hier, nicht in `Transformer`:
    was der Lauf tatsaechlich getan hat, steht als val_runs in `daten` und im
    Checkpoint, und daraus laesst sich derselbe Split exakt wiederherstellen
    (train_fest = alle Runs ohne val_runs). Das Mittel zusaetzlich in die
    Identitaet zu ziehen braeche jeden vorhandenen Transformer-Hash, ohne
    eine Frage zu beantworten, die val_runs nicht schon beantwortet. Dass
    zwei Laeufe mit verschiedenem train_fest denselben Pfad haben, faengt
    der overwrite-Schutz in train2.
    """
    strict: bool = True
    overwrite: bool = False
    json_out: str | None = None
    model_path: str | None = None
    label_dir: str | None = None
    out_path: str | None = None
    data_path: str | None = None    # Notausgang LESEpfad: fremder Paar-Datensatz
    list_runs: bool = False     # nur auflisten, nichts rechnen (animate_run)
    train_fest: list[int] = field(default_factory=list)


@dataclass
class Overlap(_Typed):
    """Werkzeug: Runs nach Partikel-Ueberlappung klassifizieren.

    box_w ist dieselbe Datensatzkonstante wie in Labeling -- hier dient sie
    als Schwelle fuer "nah". cross_thresh und mean_clean_max sind EMPIRISCH
    an den Ergebnissen kalibriert, nicht hergeleitet -- wer sie aendert,
    aendert die Run-Listen.
    """
    box_w: float = BOX_W
    cross_thresh: float = 1.4
    mean_clean_max: float = 0.05


@dataclass
class AnimRun(_Typed):
    """Darstellung [A]: ganzer Run, Farbe = Polaritaet.

    Konfiguration, die keine Zahl beruehrt: keine Konsistenz, keine
    Identitaet. Framezahl = dt / frame_ms -- ueber ~100 wird das GIF
    unhandlich.
    """
    frame_ms: float = 20.0
    speedup: float = 1.0        # > 1 ueberspringt Frames
    downscale: int = 2
    point: int = 1


@dataclass
class AnimClusters(_Typed):
    """Darstellung [B]: Roh | Cluster | GT nebeneinander.

    color_by entscheidet, WAS man sieht (visualize.py):
      "cluster" eigene Farbe je Cluster  -> FRAGMENTIERUNG sichtbar
      "gt"      Farbe des Mehrheits-GT   -> REINHEITSFEHLER sichtbar,
                                            Fragmentierung dagegen NICHT
    trail=True laesst die Trajektorien aufwachsen (zeigt: ein Cluster ist
    eine Raum-Zeit-Bahn); False zeigt Momentaufnahmen (Verschmelzungen).
    """
    frame_ms: float = 0.25
    trail: bool = True
    downscale: int = 1
    point: int = 2
    min_size: int = 5           # Cluster darunter werden grau
    color_by: str = "cluster"
    show_gt: bool = True
    crop: bool = True
    # Welche Panels gezeichnet werden, Komma-getrennt: roh, tracklets, gt.
    # Einzeln gewaehlt wird das Bild breiter und damit im Report lesbar --
    # drei nebeneinander sind auf einem geteilten Bildschirm zu klein.
    panels: str = "roh,tracklets,gt"


@dataclass
class AnimTraj(_Typed):
    """Darstellung [D]: Tracklets gegen rekonstruierte Trajektorien.

    Das Bild der Stufe: links wechseln alle Farben alle alpha µs, rechts
    stehen sie ueber den ganzen Run. Wechselt rechts eine Farbe mitten in
    einer Bahn oder laufen zwei zusammen, ist das ein Identitaetswechsel --
    genau das, wofuer sonst IDSW noetig ist.

    trail_ms ist GLEITEND, anders als bei AnimClusters: dort wird der Trail
    an jeder Fenstergrenze zurueckgesetzt, damit die Grenze sichtbar
    bleibt. Hier soll sie verschwinden. 0 zeigt Momentaufnahmen.

    min_size wirkt auf BEIDE Panels, deren Cluster sich um Groessen-
    ordnungen unterscheiden: ein Tracklet hat rund 150 Events, eine
    Trajektorie ueber hunderttausend. Ein Wert, der rechts die Bruchstuecke
    ausgraut, graut links fast alles aus. Vorgabe deshalb am
    Kandidatenfilter orientiert.
    """
    frame_ms: float = 20.0
    trail_ms: float = 20.0
    # Anzeigedauer je Frame in ms. 0 = abgeleitet aus frame_ms. Getrennt,
    # weil Abtastung und Abspielgeschwindigkeit nichts miteinander zu tun
    # haben: dieselben 400 Frames koennen in 20 s oder in 2 min laufen.
    hold_ms: float = 0.0
    downscale: int = 1
    point: int = 2
    min_size: int = 15
    crop: bool = True
    # Komma-getrennt: roh, tracklets, trajektorien, gt.
    panels: str = "tracklets,trajektorien,gt"


# ============================================================
# Stufen-Konfigurationen (Zusammensetzung = Abhaengigkeitskette §5)
# ============================================================


@dataclass
class LabelCfg:
    io:        IO        = field(default_factory=IO)
    labeling:  Labeling  = field(default_factory=Labeling)
    selection: Selection = field(default_factory=Selection)


@dataclass
class PairCfg:
    io: IO = field(default_factory=IO)
    neighborhood: Neighborhood = field(default_factory=Neighborhood)
    pairs:        PairSampling = field(default_factory=PairSampling)
    compute:      Compute      = field(default_factory=Compute)


@dataclass
class TrainCfg:
    """neighborhood und pairs werden NICHT gerechnet -- sie leiten den
    EINGABEpfad ab (config.pairs_file). Die Nachbarschaft bleibt trotzdem
    eine Eigenschaft des DATENSATZES: train.py liest sie aus der .npz und
    prueft die ganze Gruppe gegen cfg.neighborhood. Zwei unabhaengige
    Rechnungen desselben Werts -- stimmen sie nicht ueberein, luegt eine."""
    io:           IO           = field(default_factory=IO)
    neighborhood: Neighborhood = field(default_factory=Neighborhood)
    pairs:        PairSampling = field(default_factory=PairSampling)
    training:     Training     = field(default_factory=Training)
    compute:      Compute      = field(default_factory=Compute)


@dataclass
class SolveCfg:
    """pairs und training werden nicht gerechnet, sondern gebraucht, um den
    Modellpfad ABZULEITEN statt ihn zu tippen."""
    io: IO = field(default_factory=IO)
    neighborhood: Neighborhood = field(default_factory=Neighborhood)
    pairs:        PairSampling = field(default_factory=PairSampling)
    training:     Training     = field(default_factory=Training)
    solver:       Solver       = field(default_factory=Solver)
    selection:    Selection    = field(default_factory=Selection)
    window:       Window       = field(default_factory=Window)
    compute:      Compute      = field(default_factory=Compute)


@dataclass
class FeatureCfg:
    """Abgeleiteter Schritt: liest das Tracklet-Artefakt, schreibt (B).

    Kein `window`: die Tabelle deckt den ganzen Run ab, und der Ausschnitt
    steckt bereits in den Fenstern des gelesenen Artefakts.
    """
    io:           IO           = field(default_factory=IO)
    neighborhood: Neighborhood = field(default_factory=Neighborhood)
    pairs:        PairSampling = field(default_factory=PairSampling)
    training:     Training     = field(default_factory=Training)
    solver:       Solver       = field(default_factory=Solver)
    selection:    Selection    = field(default_factory=Selection)
    features:     Features     = field(default_factory=Features)
    compute:      Compute      = field(default_factory=Compute)


@dataclass
class KandidatenCfg:
    """[6a] Kandidatenpaare und Knotentensor, ein Artefakt je Run.

    Traegt die ganze Kette von Stufe 1, weil die Tracklets der EINGABEpfad
    sind -- der Baum haengt entsprechend unter tracklets/. Kein `window`:
    das Artefakt deckt den ganzen Run ab, wie bei FeatureCfg.

    Kein `features`: Stufe 2 liest (A), nicht (B). Die Labels rechnet sie
    ueber dieselbe Mehrheitsmechanik selbst; off_s und off_k beruehren
    keine Groesse, die hier gebraucht wird.
    """
    io:           IO           = field(default_factory=IO)
    neighborhood: Neighborhood = field(default_factory=Neighborhood)
    pairs:        PairSampling = field(default_factory=PairSampling)
    training:     Training     = field(default_factory=Training)
    solver:       Solver       = field(default_factory=Solver)
    selection:    Selection    = field(default_factory=Selection)
    kandidaten:   Kandidaten   = field(default_factory=Kandidaten)
    compute:      Compute      = field(default_factory=Compute)


@dataclass
class Stufe2Cfg:
    """[6b] Training des Transformers.

    Kein `selection`: trainiert wird ueber alle Runs aus kandidaten.runs,
    nicht ueber einen. Dieselbe Rolle wie TrainCfg eine Ebene tiefer --
    `kandidaten` wird nicht gerechnet, sondern leitet den EINGABEpfad ab.
    """
    io:           IO           = field(default_factory=IO)
    neighborhood: Neighborhood = field(default_factory=Neighborhood)
    pairs:        PairSampling = field(default_factory=PairSampling)
    training:     Training     = field(default_factory=Training)
    solver:       Solver       = field(default_factory=Solver)
    kandidaten:   Kandidaten   = field(default_factory=Kandidaten)
    transformer:  Transformer  = field(default_factory=Transformer)
    compute:      Compute      = field(default_factory=Compute)


@dataclass
class Solve2Cfg:
    """[6c] Ein Stufe-2-Solve, ein Run.

    Traegt die ganze Kette: Arbeitspunkt, Netz von Stufe 1, dessen
    Solver-Betriebspunkt, den Kandidatensatz und das Stufe-2-Netz. Erst
    `solver2` gehoert zu diesem Schritt.

    Kein `window`: Stufe 2 loest den GANZEN Run in einem Zug. Das ist der
    Sinn der Stufe -- Trajektorien laufen ueber alle Fenster, und ein
    Ausschnitt zerschnitte genau das, was hier zusammengefuegt werden soll.
    """
    io:           IO           = field(default_factory=IO)
    neighborhood: Neighborhood = field(default_factory=Neighborhood)
    pairs:        PairSampling = field(default_factory=PairSampling)
    training:     Training     = field(default_factory=Training)
    solver:       Solver       = field(default_factory=Solver)
    kandidaten:   Kandidaten   = field(default_factory=Kandidaten)
    transformer:  Transformer  = field(default_factory=Transformer)
    selection:    Selection    = field(default_factory=Selection)
    solver2:      Solver2      = field(default_factory=Solver2)
    compute:      Compute      = field(default_factory=Compute)


@dataclass
class OverlapCfg:
    io:        IO        = field(default_factory=IO)
    selection: Selection = field(default_factory=Selection)
    overlap:   Overlap   = field(default_factory=Overlap)


@dataclass
class AnimRunCfg:
    io:        IO        = field(default_factory=IO)
    selection: Selection = field(default_factory=Selection)
    window:    Window    = field(default_factory=lambda: Window(t0=None, dt=None))
    anim:      AnimRun   = field(default_factory=AnimRun)


@dataclass
class AnimClustersCfg:
    """Wie SolveCfg, plus Darstellung: die Animation LOEST das Fenster selbst."""
    io:           IO           = field(default_factory=IO)
    neighborhood: Neighborhood = field(default_factory=Neighborhood)
    pairs:        PairSampling = field(default_factory=PairSampling)
    training:     Training     = field(default_factory=Training)
    solver:       Solver       = field(default_factory=Solver)
    selection:    Selection    = field(default_factory=Selection)
    window:       Window       = field(default_factory=Window)
    compute:      Compute      = field(default_factory=Compute)
    anim:         AnimClusters = field(default_factory=AnimClusters)


@dataclass
class AnimTrajCfg:
    """Wie Solve2Cfg, plus Darstellung: die Animation LOEST Stufe 2 selbst.

    window schneidet nur die ANZEIGE zu -- geloest wird immer der ganze
    Run. Anders herum waeren die Farben eines Ausschnitts andere als die
    des vollen Laufs, und genau die Stabilitaet der Farben ist die Aussage.
    """
    io:           IO           = field(default_factory=IO)
    neighborhood: Neighborhood = field(default_factory=Neighborhood)
    pairs:        PairSampling = field(default_factory=PairSampling)
    training:     Training     = field(default_factory=Training)
    solver:       Solver       = field(default_factory=Solver)
    kandidaten:   Kandidaten   = field(default_factory=Kandidaten)
    transformer:  Transformer  = field(default_factory=Transformer)
    selection:    Selection    = field(default_factory=Selection)
    window:       Window       = field(
        default_factory=lambda: Window(t0=None, dt=None))
    solver2:      Solver2      = field(default_factory=Solver2)
    compute:      Compute      = field(default_factory=Compute)
    anim:         AnimTraj     = field(default_factory=AnimTraj)


# ============================================================
# Kanonische Form, Identitaet, Vertrag
# ============================================================

def _num(v) -> str:
    """Kompakte Zahldarstellung fuer Verzeichnisnamen: 3.0 -> '3', 7.06 -> '7.06'."""
    return f"{v:g}"


def _norm(o):
    """Rekursiv normieren: Floats auf 12 signifikante Stellen, Tupel -> Liste,
    Dicts sortiert. Damit koennen 3 und 3.0 nicht auseinanderlaufen."""
    if isinstance(o, int):      # schliesst bool ein (Subklasse von int);
        return o                # json.dumps schreibt daraus true/false
    if isinstance(o, float):
        return float(f"{o:.12g}")
    if isinstance(o, (list, tuple)):
        return [_norm(v) for v in o]
    if isinstance(o, dict):
        return {k: _norm(v) for k, v in sorted(o.items())}
    return o


def canonical(group) -> str:
    """Normierte Darstellung EINER Gruppe. Basis von Identitaet UND Vergleich
    -- beides ist damit garantiert dieselbe Rechnung."""
    return json.dumps(_norm(asdict(group)), sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False)


def digest(group, n: int = 6) -> str:
    return hashlib.sha256(canonical(group).encode("utf-8")).hexdigest()[:n]


def slug(group) -> str:
    """Verzeichnisname einer Gruppe: lesbarer Kern + Hash ueber ALLE Felder.

    Der lesbare Teil dient dem Menschen, der Hash der Vollstaendigkeit -- so
    landet auch v_max in der Identitaet, ohne den Namen aufzublaehen. Wer den
    Hash aufloesen will, liest den identity-Block im Artefakt.
    """
    return f"{group.short()}-{digest(group)}"


def same_neighborhood(a: Neighborhood, b: Neighborhood) -> bool:
    """Der Vertrag. Nur diese Gruppe wird verglichen -- Solve-Konfigurationen
    unterscheiden sich per Konstruktion in run_nr, t0, dt, mode."""
    return canonical(a) == canonical(b)

def identity_meta(**gruppen) -> dict:
    """Metadatenblock aus Gruppen -- ersetzt handgepflegte Feldlisten.

    Aufruf: identity_meta(neighborhood=nb, pairs=ps, training=tr)
    Ergebnis: verschachtelt, normiert, in DERSELBEN Form, die auch canonical()
    verwendet -- Metadaten und Vertrag koennen nicht auseinanderlaufen.

    `io` wird bewusst nicht uebergeben: Pfade und Notausgaenge gehoeren nicht
    zur Identitaet eines Artefakts.
    """
    return {name: _norm(asdict(g)) for name, g in gruppen.items()}    