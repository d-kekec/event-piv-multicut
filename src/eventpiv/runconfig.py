#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
runconfig.py -- Laufparameter als Gruppen-Dataclasses.

PARAMETER, nicht Pfade. config.py bleibt die einzige Stelle mit Pfadwissen:
hier steht, WAS gerechnet wird, dort, WOHIN es geschrieben wird.

Der Gruppenschnitt folgt dem UEBEREINSTIMMUNGS-Kriterium, nicht dem
Neubau-Kriterium (BA_Konfiguration.md §1): nur `Neighborhood` muss zwischen
Paarbau [2] und Solve [4] gleich sein, und nur sie wird verglichen.

KEIN frozen=True -- OmegaConf macht den Container dann read-only und jeder
Hydra-Override scheitert (gemessen 27.07.2026). Der Schutz kommt statt dessen
aus der Struktur: scripts/_cli.to_cfg laeuft genau einmal, danach reicht das
fertige Objekt nach unten durch. Module schreiben NIE in cfg.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields

WIN_DT_YES_DEFAULT = 40_000     # µs, JA-Seite (BA_NN_Kantenkosten.md §4.1)

BOX_W = 11.0                    # px, Boxbreite = Hoehe. KONSTANTE DES DATENSATZES,
                                # BoundingBox Länge.

_CAST = {"int": int, "float": float, "str": str, "bool": bool}


class _Typed:
    """Feldwerte auf ihren annotierten Typ bringen.

    `r_max: float` ist in Python nur Dokumentation -- Neighborhood(r_max=3)
    haelt eine 3, Neighborhood(r_max=3.0) eine 3.0, und die kanonische Form
    liefe auseinander: derselbe Arbeitspunkt bekaeme zwei Verzeichnisse, und
    check_consistency schluege ohne Grund an. Ueber Hydra faellt das nie auf
    (OmegaConf castet gegen das Schema) -- bei direkter Konstruktion in
    Tests, Skripten und Auswertungen schon. Gefunden von tests/test_runconfig.py.

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
                # umdrehen, still. Ueber Hydra kann das nicht passieren
                # (OmegaConf castet gegen das Schema), bei direkter
                # Konstruktion in Tests und Skripten schon.
                raise TypeError(
                    f"{type(self).__name__}.{f.name}={v!r}: Wahrheitswert als "
                    "Zeichenkette. bool(str) ist immer True -- echten "
                    "Wahrheitswert uebergeben.")
            setattr(self, f.name, cast(v))

class _Derivable:
    """resolve_derived fuer alle Stufen, die neighborhood UND pairs tragen.

    Wird von scripts/_cli.to_cfg aufgerufen -- genau einmal, an der Grenze
    Hydra -> Modul. Danach ist die Konfiguration vollstaendig, und
    canonical()/slug() sind eindeutig. Ohne diesen Aufruf traegt win_dt noch
    None, und derselbe Datensatz bekaeme zwei verschiedene Identitaeten.

    Kein @dataclass: die Klasse hat keine Felder, sie steuert nur Verhalten
    bei -- genauso wie _Typed.
    """

    def resolve_derived(self) -> None:
        resolve(self.neighborhood, self.pairs)

# ============================================================
# Gruppen
# ============================================================

# Erlaubte Werte der Auswahlparameter. Stehen HIER und nicht in graph.py /
# model.py / train.py: runconfig darf weder scipy noch torch importieren,
# und die Pruefung muss zur Konfigurationszeit greifen -- sonst existiert
# der Slug laengst und ein Verzeichnis ist danach benannt.
NEIGHBORHOOD_GEOMS = ("ellipsoid", "cylinder", "hyperboloid")
FEATURE_MODES = ("dr", "absxy", "raw6")
IMBALANCE_MODES = ("sample", "weight")

@dataclass
class Neighborhood(_Typed):
    """DER VERTRAG -- und zugleich die Identitaet des Arbeitspunkts.

    v_max wirkt nur bei geom="hyperboloid", gehoert aber hierher: er
    entscheidet dort mit, welche Kanten existieren (Befund 27.07.2026).
    """
    alpha: int = 20_000
    r_max: float = 3.0
    geom: str = "cylinder"
    v_max: float = 1.02e-4

    def short(self) -> str:
        return f"a{self.alpha}_r{_num(self.r_max)}_{self.geom}"

    def __post_init__(self) -> None:
        super().__post_init__()          # Typcast aus _Typed zuerst
        # Zur Konfigurationszeit pruefen, nicht erst tief im Lauf: bis
        # graph.neighborhood_pairs es merkt, ist der Slug laengst berechnet
        # und ein Verzeichnis danach benannt.
        if self.geom not in NEIGHBORHOOD_GEOMS:
            raise ValueError(
                f"neighborhood.geom={self.geom!r} unbekannt. "
                f"Erlaubt: {NEIGHBORHOOD_GEOMS}")


@dataclass
class PairSampling(_Typed):
    """Identitaet des Paar-Datensatzes. `split` ist hier die QUELLE der Paare
    -- nicht zu verwechseln mit Selection.split (was geloest wird)."""
    split: str = "train"
    clean_runs: list[int] = field(
        default_factory=lambda: [3, 40, 43, 48, 81])
    overlap_runs: list[int] = field(
        default_factory=lambda: [2, 11, 19, 30, 32, 35, 44, 59, 61, 76])
    win_dt: int | None = None       # None -> aus alpha/r_max abgeleitet
    win_dt_yes: int | None = None   # None -> WIN_DT_YES_DEFAULT
    win_offsets: list[int] = field(
        default_factory=lambda: [1_500_000, 3_500_000, 5_500_000])
    slab_dt: int = 10_000
    max_yes: int = 2_000_000
    max_no: int = 2_000_000
    seed: int = 0

    def short(self) -> str:
        # Engstelle: slug() ruft short() IMMER auf. Ein unaufgeloestes win_dt
        # ergibt still einen anderen Hash fuer denselben Datensatz
        # (gemessen: -b7c58d gegen -6f854e). identity_meta bleibt ungeprueft --
        # dort steht None sichtbar in der JSON, hier verschwindet es im Hash.
        if self.win_dt is None or self.win_dt_yes is None:
            raise RuntimeError(
                "PairSampling ist unaufgeloest (win_dt/win_dt_yes = None). "
                "runconfig.resolve(nb, ps) laeuft normalerweise in "
                "scripts/_cli.to_cfg -- bei direkter Konstruktion selbst rufen.")
        return f"{self.split}_c{len(self.clean_runs)}_o{len(self.overlap_runs)}"


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


@dataclass
class Solver(_Typed):
    """Erzwingt keinen Neubau und wird nie verglichen (…Pipeline_Dateien §5)."""
    mode: str = "P"
    use_knn: bool = False
    k_max: int = 24

    def short(self) -> str:
        return self.mode + (f"_knn{self.k_max}" if self.use_knn else "")


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
    """Solve- bzw. Anzeigefenster. NICHT PairSampling.win_dt (Sammelfenster).

    None heisst "ab Anfang" bzw. "ganzer Run" -- load_events kennt das
    ohnehin. Sinnvoll nur bei animate_run; beim Solve waere ein ganzer Run
    ein VRAM-Selbstmord.
    """
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
    """Schublade 3 und 4 (BA_Konfiguration.md §1): Ein-/Ausgabe und Notausgaenge.

    NIE Teil einer Artefakt-Identitaet -- deshalb eine eigene Gruppe, die
    identity_meta() nicht anfasst. Felder, die eine Stufe nicht braucht,
    bleiben auf ihrem Default; das ist unschaedlich, weil aus dieser Gruppe
    nichts in Metadaten oder Pfade wandert.

    model_path ist der Notausgang auf dem LESEpfad: ein fremdes Netz laden,
    dessen Parameter nicht aus der Konfiguration folgen. Bewusst ein
    sichtbarer Akt, wie strict=false.
    """
    strict: bool = True
    overwrite: bool = False
    json_out: str | None = None
    model_path: str | None = None
    label_dir: str | None = None
    out_path: str | None = None
    data_path: str | None = None    # Notausgang LESEpfad: fremder Paar-Datensatz
    list_runs: bool = False     # nur auflisten, nichts rechnen (animate_run)

@dataclass
class Report(_Typed):
    """Auswertung mehrerer Solve-Laeufe. KEINE Laufauswahl: `crossing` und
    `clean` sind zwei BEWERTUNGSGRUPPEN, die getrennt gemittelt werden --
    ein Mittelwert ueber beide koennte schlechte Trennung durch gute
    Kontrolle kaschieren (BA_NN_Kantenkosten.md §3.8)."""
    split: str = "val"
    crossing: list[int] = field(default_factory=lambda: [3, 4, 6, 20])
    clean: list[int] = field(default_factory=lambda: [7, 9, 12, 13, 21])

@dataclass
class Overlap(_Typed):
    """Werkzeug: Runs nach Partikel-Ueberlappung klassifizieren.

    box_w ist dieselbe Datensatzkonstante wie in Labeling -- hier dient sie
    als Schwelle fuer "nah". cross_thresh und mean_clean_max sind EMPIRISCH
    an den Ergebnissen kalibriert (BA_Datensatz_und_Pipeline.md §2.2), nicht
    hergeleitet.
    """
    box_w: float = BOX_W
    cross_thresh: float = 1.4
    mean_clean_max: float = 0.05


@dataclass
class AnimRun(_Typed):
    """Darstellung [A]: ganzer Run, Farbe = Polaritaet.

    Fuenfte Schublade (BA_Konfiguration.md §1): Konfiguration, aber sie
    beruehrt keine Zahl, braucht keine Konsistenz und gehoert in keine
    Identitaet. Framezahl = dt / frame_ms -- ueber ~100 wird das GIF unhandlich.
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

# ============================================================
# Stufen-Konfigurationen (Zusammensetzung = Abhaengigkeitskette §5)
# ============================================================


@dataclass
class LabelCfg:
    io:        IO        = field(default_factory=IO)
    labeling:  Labeling  = field(default_factory=Labeling)
    selection: Selection = field(default_factory=Selection)


@dataclass
class PairCfg(_Derivable):
    io: IO = field(default_factory=IO)
    neighborhood: Neighborhood = field(default_factory=Neighborhood)
    pairs:        PairSampling = field(default_factory=PairSampling)
    compute:      Compute      = field(default_factory=Compute)


@dataclass
class TrainCfg(_Derivable):
    """neighborhood und pairs werden NICHT gerechnet -- sie leiten den
    EINGABEpfad ab (config.pairs_file). Die Geometrie bleibt trotzdem eine
    Eigenschaft des Datensatzes: train.py liest sie aus der .npz und prueft
    sie gegen cfg.neighborhood.geom. Zwei unabhaengige Rechnungen desselben
    Werts -- stimmen sie nicht ueberein, luegt einer von beiden."""
    io:           IO           = field(default_factory=IO)
    neighborhood: Neighborhood = field(default_factory=Neighborhood)
    pairs:        PairSampling = field(default_factory=PairSampling)
    training:     Training     = field(default_factory=Training)
    compute:      Compute      = field(default_factory=Compute)


@dataclass
class SolveCfg(_Derivable):
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
class ReportCfg(_Derivable):
    """Liest nur -- rechnet nichts. Braucht neighborhood/pairs/training, um
    das solves/-Verzeichnis zu FINDEN, und solver/window, weil beide im
    Dateinamen stehen."""
    io:           IO           = field(default_factory=IO)
    neighborhood: Neighborhood = field(default_factory=Neighborhood)
    pairs:        PairSampling = field(default_factory=PairSampling)
    training:     Training     = field(default_factory=Training)
    solver:       Solver       = field(default_factory=Solver)
    window:       Window       = field(default_factory=Window)
    report:       Report       = field(default_factory=Report)


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
class AnimClustersCfg(_Derivable):
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


# ============================================================
# Abgeleitete Defaults
# ============================================================

def win_dt_for(alpha, r_max, *, ref_alpha=10_000, ref_r_max=7.06,
               ref_win_dt=40_000, cap=300_000) -> int:
    """Sammelfenster so waehlen, dass die NEIN-Ausbeute konstant bleibt.

    NEIN-Paare ~ Nachbarschaftsvolumen ~ r_max^2 * alpha; kleinere
    Nachbarschaft ueber ein breiteres Sammelfenster kompensieren.
    Geeicht am ELLIPSOID -- fuer Zylinder/Hyperboloid konservativ, nicht falsch.
    Nur fuers SAMMELN; das Solve-Fenster (Window.dt) bleibt klein.
    """
    v = int(ref_win_dt * (ref_r_max ** 2 * ref_alpha) / (r_max ** 2 * alpha))
    return min(v, cap)


def resolve(nb: Neighborhood, ps: PairSampling) -> PairSampling:
    """Abgeleitete Defaults einsetzen. MUSS vor canonical()/slug() laufen --
    sonst bekaemen win_dt=None und win_dt=110756 verschiedene Slugs, obwohl
    sie denselben Datensatz beschreiben.

    `is None` statt `or`: win_dt=0 ist ein gueltiger (wenn auch unsinniger)
    Wert und darf nicht still durch den Default ersetzt werden (TODO C2).
    """
    if ps.win_dt is None:
        ps.win_dt = win_dt_for(nb.alpha, nb.r_max)
    if ps.win_dt_yes is None:
        ps.win_dt_yes = WIN_DT_YES_DEFAULT
    return ps


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
    Hash aufloesen will, liest die config.yaml, die danebenliegt.
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