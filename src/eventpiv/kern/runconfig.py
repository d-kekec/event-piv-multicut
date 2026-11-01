#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
runconfig.py -- Laufparameter als Gruppen-Dataclasses.

PARAMETER, nicht Pfade. config.py bleibt die einzige Stelle mit Pfadwissen:
hier steht, WAS gerechnet wird, dort, WOHIN es geschrieben wird.

Die Gruppen der Kette folgen dem UEBEREINSTIMMUNGS-Kriterium: eine Gruppe
ist, was zwischen zwei Stufen gleich sein MUSS (Neighborhood, PairSampling,
Training, Solver, Kandidaten, Transformer, Solver2, Fit). Zwischen den
Stufen geprueft wird nur `Neighborhood` (solve.check_consistency,
train.check_dataset_neighborhood); die uebrigen sichert der Pfad. Weitere
Gruppen nach ihrer Rolle: Selection und Window je Aufruf, Compute und IO
fuer Ressourcen und Ein-/Ausgabe, Overlap und Anim* fuer Werkzeuge, Labeling
fuer [1]. Innerhalb einer Stufe bindet eine Groesse ueber die Gruppengrenze:
`kandidaten.v_max` ist dieselbe wie `neighborhood.v_max`. Geprueft wird
das in KandidatenCfg, wo der Kandidatensatz entsteht.

DIE VORGABEN SIND DER ARBEITSPUNKT VON STELLA und hier begruendet; batch1
und der zweite STELLA-Arbeitspunkt liegen als Gruppen unter configs/
(`datensatz=batch1` usw.). Eine Vorgabe zu aendern verschiebt den
Arbeitspunkt; test_vorgaben_sind_ein_arbeitspunkt vergleicht dafuer alle
Slugs mit festen Werten.

KEIN frozen=True -- OmegaConf macht den Container dann read-only und jeder
Hydra-Override scheitert. Bei einem Wechsel von OmegaConf neu pruefen.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields

BOX_W = 11.0                    # px, Kantenlaenge der quadratischen
                                # GT-Box von batch1.

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
FEATURE_MODES = ("dr", "drjac")
IMBALANCE_MODES = ("sample",) # Klassenverhaeltnis ueber Ziehen, nicht Gewichte
SOLVER_BACKENDS = ("rama", "cc")
SOLVER_MODES = ("P", "PD", "PD+", "D")
POOLINGS = ("cls", "mean")
LABEL_MODI = ("streng", "mehrheit")

@dataclass
class Neighborhood(_Typed):
    """DER VERTRAG zwischen [2] und [4], oberste Ebene von config.wp_dir.

    v_max wirkt nur bei geom="hyperboloid", gehoert aber hierher: er
    entscheidet dort mit, welche Kanten existieren. Ueberkonservativ ist
    zulaessig, eine bedingte Identitaetsregel waere eine spaetere Falle.

    polarity waehlt die KNOTENMENGE: "on" laesst nur positive Events zu,
    "both" alle. Es gehoert aus demselben Grund hierher wie v_max, nur
    zwingender -- der Filter wirkt in Stufe [2] und in Stufe [4], und
    stimmen sie nicht ueberein, sieht das Netz im Einsatz eine andere
    Kantenpopulation als im Training. Ohne dieses Feld wuerde ein mit
    beiden Polaritaeten gebauter Paar-Datensatz still fuer einen
    ON-only-Einsatz weiterverwendet.
    """
    # alpha in µs, r_max in px, v_max in px/µs. alpha ist zugleich die
    # Laenge eines Tracklet-Fensters: kuerzer schnitte Kanten weg, die die
    # Geometrie erlaubt, laenger liesse zwei Partikel oefter in EINEM
    # Tracklet verschmelzen -- in Stufe 1 der einzige unumkehrbare Fehler.
    # alpha ist nicht gemessen, sondern gewaehlt innerhalb der Schranken aus
    # der Tabelle "Kandidaten fuer alpha1" von
    # tools/schaetzer/stufe1_parameter_schaetzer.py: Speicher des Jaccard je
    # Fenster, Drift gegen Wolkenausdehnung. Dasselbe Werkzeug bestimmt v_max
    # (dort in px/ms) und r_max je geom und alpha neu.
    alpha: int = 500
    r_max: float = 5.1
    geom: str = "hyperboloid"
    v_max: float = 1.79e-3
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

    `runs`: die Quell-Runs, aufsteigend (die Reihenfolge geht in den Hash
    ein), ueber die Eventrate geschichtet -- die Rate bestimmt die lokale
    Nachbarschaftsdichte und damit den Merkmalsraum des Netzes. Beide
    Klassen stammen aus DENSELBEN Fenstern: nur so sieht das Netz eine
    Eingangsverteilung und nicht zwei.

    `ja_frac` ist keine Feinabstimmung, sondern die Speicherbedingung: bei
    voller Abdeckung wuechsen die JA-Paare eines Runs weit ueber den
    Arbeitsspeicher. NEIN wird vollstaendig behalten (bis `max_no`), es ist
    die knappe Klasse.

    Das Klassenverhaeltnis setzt `ja_vorrat`, nicht ein JA-Deckel. Eine
    obere Schranke fuer JA folgt daraus von selbst: n_nein <= max_no, also
    max_ja <= ja_vorrat * max_no.

    KEIN Sammelfenster: gesammelt wird in anstossenden Fenstern der Laenge
    `neighborhood.alpha` -- genau die Fenster, die Stufe 1 im Einsatz
    bildet. `fenster_schritt` laesst Fenster aus, nie Teile eines Fensters.
    """
    split: str = "train"
    runs: list[int] = field(
        default_factory=lambda: [2, 6, 8, 9, 13, 18, 23, 28, 30, 37, 38,
                                 41, 42, 43, 46, 48, 49, 54, 55, 64, 65, 66,
                                 72, 80, 82, 85, 92, 93, 96, 98])
    ja_frac: float = 2e-2
    max_no: int = 1_000_000
    # Jedes n-te Fenster abschreiten; ein Fenster bleibt vollstaendig. Am
    # Datensatz kalibriert, damit fast jeder Run den max_no-Deckel erreicht
    # und alle Runs gleich viel beitragen.
    fenster_schritt: int = 20
    # Groesser als training.sample_ratio: [2] legt Vorrat an, [3] setzt den
    # Betriebspunkt.
    ja_vorrat: float = 2.0
    seed: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.ja_vorrat <= 0:
            raise ValueError(f"pairs.ja_vorrat={self.ja_vorrat} muss > 0 sein.")
        if not 0.0 <= self.ja_frac <= 1.0:
            raise ValueError(
                f"pairs.ja_frac={self.ja_frac} muss in [0, 1] liegen.")
        if self.fenster_schritt < 1:
            raise ValueError(
                f"pairs.fenster_schritt={self.fenster_schritt} muss >= 1 sein.")

    def short(self) -> str:
        # "rn" statt "r" fuer die ANZAHL Runs: "r" bezeichnet im Projekt
        # sonst einen Radius (r_max) oder einen einzelnen Run (run<N>).
        # "split_" ausgeschrieben, damit "train" nicht wie ein Name aussieht.
        # Der Hash bleibt unberuehrt -- short() ist eine Methode und steht
        # nicht in asdict(), also nicht in canonical().
        return f"split_{self.split}_rn{len(self.runs)}"

@dataclass
class Training(_Typed):
    """Identitaet des Kantenkosten-Netzes.

    `feature_mode`: Merkmale je Kante. "dr" genuegt, weil die NEIN-Paare
    ueberwiegend nebeneinander statt uebereinander liegen, und spart die
    quadratische Last des Jaccard; batch1 braucht "drjac", weil die Kosten
    mit "dr" allein dort nie negativ werden.

    `hidden`: Breite der versteckten Schicht, am Datensatz kalibriert.
    `lr`: Schrittweite; zu gross laesst sie die Parameter um das Optimum
    zappeln -- eine zackige Kurve, kein schlechteres Ergebnis.

    `sample_ratio`, JA:NEIN im Training, verschiebt nur den Achsenabschnitt
    des Logits um log(s), nicht die Rangfolge. 1:1 legt die Nullstelle in
    die Verteilung, damit TPR, TNR und pred_neg_frac ablesbar bleiben.
    """
    feature_mode: str = "dr"
    imbalance: str = "sample"
    sample_ratio: float = 1.0
    hidden: int = 32
    epochs: int = 100
    batch: int = 65_536
    lr: float = 3e-2
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

    cost_offset: Versatz auf jede Kante, der BETRIEBSPUNKT, kein
    Kalibrierwert -- er verschiebt die Schwelle zwischen Schneiden und
    Verbinden, ohne die Rangfolge der Kanten zu aendern. Er tauscht
    Verschmelzung gegen Fragmentierung, ein Optimum gibt es nicht; bei 0
    entscheidet allein das Netz. Dieselbe Wirkung hat sample_ratio, kostet
    aber ein Neutraining statt eines Overrides.
    """
    mode: str = "P"
    use_knn: bool = False
    k_max: int = 24
    backend: str = "rama"
    cost_offset: float = 0.0

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
class Kandidaten(_Typed):
    """Identitaet des Stufe-2-Datensatzes: welche Knoten, welche Kanten.

    Von `Transformer` getrennt nach dem UEBEREINSTIMMUNGS-Kriterium, aus
    demselben Grund wie PairSampling von Training: der Datensatz ist je Run
    teuer, das Netz wird im Sweep oft neu trainiert. In EINER Gruppe
    erzwaenge jedes neue d_model einen Neubau der Kandidaten.

    alpha/r_max/geom/v_max sind bewusst NICHT `Neighborhood`. Sie messen
    hier den Abstand zweier TRACKLET-ZENTROIDE, nicht zweier Events, und
    `polarity` hat auf Zentroiden keinen Sinn. Eine verschachtelte Gruppe,
    deren Haelfte nicht gilt, waere eine spaetere Falle.

    runs: dieselben wie im Paarbau, aufsteigend (Reihenfolge im Hash);
    Kontrollruns laufen ueber selection.runs unter demselben Slug.
    alpha, µs: Reichweite zwischen den t_min zweier Knoten; mehr Reichweite
    tauscht Brueche gegen Verschmelzung.
    r_max, px: Taille, der Zentroidabstand zweier Knoten derselben Bahn bei
    kleinem dt; die Drift traegt die Aufweitung v_max * dt.
    v_max, px/µs: dieselbe Groesse wie neighborhood.v_max (_pruefe_v_max).
    min_events: kleinste Knotengroesse in Events; jede Schwelle ueber 1
    verwirft mit den Bruchstuecken auch Bruecken ueber Stellen, an denen ein
    Partikel nur wenige Events liefert.
    n_events: Event-Tokens je Knoten; kleinere Knoten werden maskiert
    aufgefuellt.
    """
    split: str = "train"
    runs: list[int] = field(default_factory=lambda: [
        2, 6, 8, 9, 13, 18, 23, 28, 30, 37, 38, 41, 42, 43, 46,
        48, 49, 54, 55, 64, 65, 66, 72, 80, 82, 85, 92, 93, 96, 98])
    alpha: int = 7_500
    r_max: float = 8.0
    geom: str = "hyperboloid"
    v_max: float = 1.79e-3
    min_events: int = 1
    n_events: int = 32
    seed: int = 0

    def short(self) -> str:
        # "rn" fuer die Anzahl Runs, "rmax" fuer den Radius -- ein blosses
        # "r" waere doppeldeutig.
        return (f"split_{self.split}_rn{len(self.runs)}_a{self.alpha}"
                f"_rmax{_num(self.r_max)}_n{self.n_events}")

    def __post_init__(self) -> None:
        super().__post_init__()          # Typcast aus _Typed zuerst
        if self.geom not in NEIGHBORHOOD_GEOMS:
            raise ValueError(
                f"kandidaten.geom={self.geom!r} unbekannt. "
                f"Erlaubt: {NEIGHBORHOOD_GEOMS}")
        if self.min_events < 1:
            # 0 filterte dasselbe wie 1, nur unter anderem Slug. Ein
            # Einzelevent-Knoten ist zulaessig: baue_tokens normiert ueber
            # feste Nenner (alpha, r_max), nicht ueber seine Ausdehnung.
            raise ValueError(
                f"kandidaten.min_events={self.min_events} -- mindestens 1.")
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
    Konstante, weil der Unterschied vermutlich klein ist -- nicht geprueft,
    und eine billige Ablation.

    label_modus entscheidet, welche Paare Gradienten bekommen -- eine
    TRAININGS-, keine Datensatzentscheidung, deshalb hier und nicht in
    Kandidaten. Das Artefakt traegt gt_track_id, gt_anteil und n_gt je
    Knoten; Ziel und Maske entstehen daraus zur Trainingszeit.
      streng    nur Knoten mit genau einem GT-Partikel. Der
                interessanteste Fall wird damit nie geuebt.
      mehrheit  auch verschmolzene, hart nach Mehrheit, ab gt_min_anteil.

    Kein feature_mode: die Baseline bekommt rohe, je Paar normierte
    (t, x, y).

    lr kleiner als in Stufe 1: das Netz hat um Groessenordnungen mehr
    Parameter als das MLP von Stufe 1.
    """
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.0
    pooling: str = "cls"
    label_modus: str = "mehrheit"
    gt_min_anteil: float = 0.9
    imbalance: str = "sample"
    sample_ratio: float = 3.0
    # Obergrenze der Trainingspaare, klassenneutral gezogen. None = kein
    # Deckel. Die Kandidatenzahl waechst mit kandidaten.alpha, die noetige
    # Zahl gesehener Kanten nicht.
    train_max: int | None = 4_000_000
    epochs: int = 100
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

    `solver` identifiziert den Betriebspunkt von STUFE 1 und steht im
    Dateinamen der Tracklets und Kandidaten -- cost_offset waehlt dort
    die QUELLE. Dieselbe Gruppe fuer die Rueckseite von Stufe 2 zu nehmen
    vermengte zwei Dinge, die getrennt gewaehlt werden muessen.

    backend="cc" ist ein ERGEBNIS, kein Hilfsmittel: CC kann eine positive
    Kante nie schneiden, zwei Trajektorien verschmelzen also, sobald EINE
    Kante zwischen ihnen positiv ist. Der Abstand zu RAMA ist die Zahl, die
    den Solver rechtfertigt.

    cost_offset wirkt hier wie in Stufe 1 -- er wird auf jede Kante addiert.
    POSITIV, anders als in Stufe 1: ohne Versatz zerfallen mehr Cluster,
    als es Partikel gibt -- Stufe 2 sitzt auf der fragmentierenden Seite.
    Am Datensatz kalibriert.
    """
    backend: str = "rama"
    mode: str = "P"
    cost_offset: float = 1.0

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
class Fit(_Typed):
    """Stufe [7]: der Kalman-Fit und die Positionskorrektur.

    r_mess, px^2: Varianz eines EINZELEVENTS um die Bahn, die Quervarianz
    der Eventwolke; ein Skalar mittelt die Anisotropie weg.
    q_c, px^2/s^5: Intensitaet des Prozessrauschens, die Q-Matrix des
    Modells mit weissem Ruck, gewichtet mit der verstrichenen ZEIT und
    nicht mit der Zahl der Messungen.
    Auf das Ergebnis wirkt bis auf den Start nur q_c/r_mess: mit demselben
    Faktor skaliert wachsen Q, R und die Startvarianzen von Ort und
    Geschwindigkeit gemeinsam, nur die der Beschleunigung (a_unbekannt)
    nicht. r_mess bleibt trotzdem physikalisch, damit die Kovarianz des
    Filters etwas ueber die Wirklichkeit aussagt.

    c0 (px), c1 (px je px/ms) und beta heben den Vorwaertsversatz der
    Eventwolke auf, Betrag c0 + c1*|u| + beta*sigma_quer, sigma_quer je
    Bahn aus ihren Events (kette.trajektorien.quer_streuung). Haengt der
    Versatz an der Partikelgroesse statt an der Geschwindigkeit, traegt die
    Form aus STELLA (Sachs et al., 2026), Gl. 1: beta*sigma_quer, c1 = 0,
    dazu ein kleiner fester Rest c0. Auf batch1 waechst der Versatz mit der
    Geschwindigkeit bei konstanter Querstreuung, fitsatz=batch1 schaltet
    beta deshalb ab. Alle drei null schalten die Korrektur ab.

    spline_eps, px: Ziel-RMSE eines kubischen Glaettungssplines, der nach
    der Korrektur je Bahn ueber den Ort gelegt wird -- der Hybrid
    (X_KS, u_KS) aus STELLA. Ort aus dem Spline, u als seine Ableitung; so
    passt u zum korrigierten Ort, den die Korrektur allein nur verschiebt
    (kette.trajektorien.spline_bahn). 0 schaltet den Spline ab.

    dt_aus, µs: Ausgabegitter. Fuer den FILTER frei -- zwischen den
    Messungen wird propagiert, nicht interpoliert. Fuer die BEWERTUNG nicht:
    trajektorien_metriken interpoliert linear auf die GT-Zeiten, ein
    groeberes Gitter als das GT-Raster kostet dort Ortsfehler aus der
    Kruemmung; im Arbeitspunkt eines Datensatzes steht deshalb sein Raster.

    init_us, µs: Fenster der Startgerade, kurz gegen die Zeit, in der sich
    die Geschwindigkeit merklich aendert -- die Gerade misst die MITTLERE
    Geschwindigkeit ihres Fensters.
    min_dauer_us, µs: kuerzere Bahnen fallen heraus; der Filter braucht
    init_us allein zum Einschwingen.

    a_unbekannt, px/s^2: Streuung der Beschleunigung beim Start; die
    Startgerade schaetzt sie nicht, sondern setzt sie null. UNTERSCHAETZEN
    IST DER TEURE FEHLER: ein zu sicherer Start wird nur langsam
    korrigiert, ein zu unsicherer kostet nur die ersten Schritte.

    Die Vorgaben sind der Arbeitspunkt von STELLA, am Datensatz kalibriert;
    batch1 setzt fitsatz=batch1.
    """
    r_mess: float = 3.0
    q_c: float = 1e7
    c0: float = 0.06
    c1: float = 0.0
    beta: float = 1.31
    dt_aus: float = 1_000.0
    init_us: float = 5_000.0
    min_dauer_us: float = 10_000.0
    a_unbekannt: float = 1e4
    # RTS-Rueckwaertslauf. Ohne ihn ist der Filter kausal und laeuft
    # jeder Richtungsaenderung hinterher.
    glaetten: bool = True
    spline_eps: float = 0.02

    def short(self) -> str:
        b = f"_b{_num(self.beta)}" if self.beta else ""
        s = f"_s{_num(self.spline_eps)}" if self.spline_eps else ""
        return (f"q{_num(self.q_c)}_r{_num(self.r_mess)}"
                f"{'_rts' if self.glaetten else ''}{b}{s}")

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.r_mess <= 0.0 or self.q_c <= 0.0:
            raise ValueError(
                f"fit.r_mess={self.r_mess} und fit.q_c={self.q_c} muessen "
                "positiv sein -- der Filter teilt durch beide.")
        for name in ("dt_aus", "init_us", "min_dauer_us"):
            if getattr(self, name) <= 0.0:
                raise ValueError(
                    f"fit.{name}={getattr(self, name)} muss positiv sein")
        if self.a_unbekannt < 0.0:
            raise ValueError(
                f"fit.a_unbekannt={self.a_unbekannt} ist eine Streuung und "
                "darf nicht negativ sein")
        if self.beta < 0.0:
            raise ValueError(
                f"fit.beta={self.beta} ist ein Faktor auf eine Streuung "
                "und darf nicht negativ sein")
        if self.spline_eps < 0.0:
            raise ValueError(
                f"fit.spline_eps={self.spline_eps} ist eine Abweichung und "
                "darf nicht negativ sein; 0 schaltet den Spline ab")


@dataclass
class Labeling(_Typed):
    """Stufe [1]. Bewusst OHNE Zeitfenster: Labels gelten IMMER fuer den
    ganzen Run. Ein Teilfenster bricht spaeter in evaluate.align_gt_window
    ("Wurde die Label-Datei ueber den GANZEN Run erzeugt?").
    r_assign, px: Radius um das naechste Zentrum, am Datensatz kalibriert;
    neu bestimmbar mit tools/schaetzer/stufe1_parameter_schaetzer.py.
    r_ambig, px: Mindestvorsprung des naechsten Zentrums vor dem zweiten."""
    r_assign: float = 5.5
    r_ambig: float = 2.0


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
    ohnehin. Sinnvoll bei animate_run und beim Abschreiten eines Runs
    ([5a]). Fuer [4] nicht auf None setzen: scripts/solve_with_nn.py ist
    dafuer nicht ausgelegt, und ohne Ausschnitt wuerde der ganze Run EIN
    Graph, dessen Groesse mit der Eventzahl des Runs waechst.
    """
    
    # t0 und dt in µs. Die Vorgabe ist ein kurzes Fenster fuer Einzelsolves
    # und Tests: jeder rechnende Aufrufer setzt seinen Ausschnitt selbst,
    # der Treiber [5a] auf None/None (ganzer Run).
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

    box_w ist hier die Schwelle fuer "nah", in px, und keine Umrechnung --
    die Boxmitte kommt aus den Spalten. Der Name ist nicht mit einer
    Boxgroesse zu verwechseln.
    cross_thresh und mean_clean_max sind EMPIRISCH am Datensatz kalibriert,
    nicht hergeleitet -- wer sie aendert, aendert die Run-Listen.
    """
    box_w: float = BOX_W
    cross_thresh: float = 1.4
    mean_clean_max: float = 0.05


@dataclass
class AnimRun(_Typed):
    """Darstellung [A]: ganzer Run, Farbe = Polaritaet.

    Konfiguration, die keine Zahl beruehrt: keine Konsistenz, keine
    Identitaet. Framezahl = Dauer in µs / (frame_ms * 1000 * speedup).
    """
    frame_ms: float = 20.0
    speedup: float = 1.0        # > 1 ueberspringt Frames
    downscale: int = 1
    point: int = 1


@dataclass
class AnimClusters(_Typed):
    """Darstellung [B] und [C]: Roh | Cluster | GT nebeneinander.

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
    ordnungen unterscheiden: eine Trajektorie traegt die Events vieler
    Tracklets. Ein Wert, der rechts die Bruchstuecke ausgraut, graut links
    fast alles aus; der Wert ist ein Kompromiss zwischen beiden Panels.
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
    # Komma-getrennt: roh, tracklets, trajektorien, fit, gt.
    # fit liest die Bahnen von [7] und braucht dafuer die Fit-Gruppe.
    panels: str = "tracklets,trajektorien,gt"


# ============================================================
# Stufen-Konfigurationen: ihre Zusammensetzung IST die Abhaengigkeitskette
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
class KandidatenCfg:
    """[6a] Kandidatenpaare und Knotentensor, ein Artefakt je Run.

    Traegt die ganze Kette von Stufe 1, weil die Tracklets der EINGABEpfad
    sind -- der Baum haengt entsprechend unter tracklets/. Kein `window`:
    das Artefakt deckt den ganzen Run ab.
    """
    io:           IO           = field(default_factory=IO)
    neighborhood: Neighborhood = field(default_factory=Neighborhood)
    pairs:        PairSampling = field(default_factory=PairSampling)
    training:     Training     = field(default_factory=Training)
    solver:       Solver       = field(default_factory=Solver)
    selection:    Selection    = field(default_factory=Selection)
    kandidaten:   Kandidaten   = field(default_factory=Kandidaten)
    compute:      Compute      = field(default_factory=Compute)

    def __post_init__(self) -> None:
        _pruefe_v_max(self)


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

    Traegt die ganze Kette: Nachbarschaft, Netz von Stufe 1, dessen
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
class FitCfg:
    """Stufe [7]: liest die Partition von [6c], schreibt die Bahnen.

    Traegt die ganze Kette, weil der Dateiname beide Solver und den Fit
    nennt. Kein `window`: gefittet wird der ganze Run.

    Kein `compute`: der Filter rechnet auf der CPU.
    """
    io:           IO           = field(default_factory=IO)
    neighborhood: Neighborhood = field(default_factory=Neighborhood)
    pairs:        PairSampling = field(default_factory=PairSampling)
    training:     Training     = field(default_factory=Training)
    solver:       Solver       = field(default_factory=Solver)
    kandidaten:   Kandidaten   = field(default_factory=Kandidaten)
    transformer:  Transformer  = field(default_factory=Transformer)
    solver2:      Solver2      = field(default_factory=Solver2)
    fit:          Fit          = field(default_factory=Fit)
    selection:    Selection    = field(default_factory=Selection)


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
    """Wie Solve2Cfg, plus Darstellung und Fit.

    Die Animation LIEST die Partition aus dem Artefakt von [6c], statt sie
    zu loesen -- sie braucht damit weder GPU noch Netz. `fit` wirkt nur,
    wenn das fit-Panel gewaehlt ist.

    window schneidet nur die ANZEIGE zu. Die Farben eines Ausschnitts
    waeren sonst andere als die des vollen Laufs, und genau ihre
    Stabilitaet ist die Aussage.
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
    fit:          Fit          = field(default_factory=Fit)
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

def _pruefe_v_max(cfg) -> None:
    """kandidaten.v_max ist dieselbe Groesse wie neighborhood.v_max: die
    schnellste Partikelgeschwindigkeit, in px/µs.

    Zwei Gruppen, weil Stufe 1 sie auf Events anwendet und Stufe 2 auf
    Zentroide. Weichen die Werte ab, ist in aller Regel eine Gruppe
    vergessen worden -- dann sucht Stufe 2 mit dem v_max eines anderen
    Datensatzes, und nichts faellt auf: die Kandidaten sehen plausibel aus.
    Eine bewusste Abweichung geht nur mit io.strict = false, als sichtbarer
    Akt.

    Aufgerufen nur in KandidatenCfg: [6a] ist die einzige Stufe, die aus
    diesen Werten ein Artefakt schreibt. Alle spaeteren lesen es ueber den
    Slug; mit vergessener Gruppe zeigt der auf ein Verzeichnis ohne
    Dateien, und dort kann nichts Plausibles entstehen.

    Verglichen in der kanonischen Form, damit 1.79e-3 und 0.00179 gleich
    sind.
    """
    if not cfg.io.strict:
        return
    a = _norm(float(cfg.kandidaten.v_max))
    b = _norm(float(cfg.neighborhood.v_max))
    if a != b:
        raise ValueError(
            f"kandidaten.v_max={a} weicht von neighborhood.v_max={b} ab. "
            "Beide sind die schnellste Partikelgeschwindigkeit. Gruppe "
            "vergessen (kandidatensatz=...)? Bewusste Abweichung nur mit "
            "io.strict=false.")


def identity_meta(**gruppen) -> dict:
    """Metadatenblock aus Gruppen -- ersetzt handgepflegte Feldlisten.

    Aufruf: identity_meta(neighborhood=nb, pairs=ps, training=tr)
    Ergebnis: verschachtelt, normiert, in DERSELBEN Form, die auch canonical()
    verwendet -- Metadaten und Vertrag koennen nicht auseinanderlaufen.

    `io` wird bewusst nicht uebergeben: Pfade und Notausgaenge gehoeren nicht
    zur Identitaet eines Artefakts.
    """
    return {name: _norm(asdict(g)) for name, g in gruppen.items()}    