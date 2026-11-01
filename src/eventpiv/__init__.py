"""event-piv-multicut: Event-basierte PIV via Multicut (RAMA).

Rekonstruiert aus dem Eventstrom einer Event-Kamera die Bahnen einzelner
Partikel. Zwei Multicut-Stufen, beide mit RAMA (Abbas und Swoboda, 2022)
und Kantenkosten aus einem kleinen Netz, danach ein Fit je Bahn.

KETTE (welches Modul zu welcher Stufe gehoert: eventpiv.kette)
  Stufe 1, Events -> Tracklets, fensterweise:
    [1]   Event -> track_id und Status aus den GT-Boxen (Trainingswahrheit)
    [2]   Paar-Datensatz fuer das Kantenkosten-Netz
    [3]   Kantenkosten-Netz trainieren
    [4]   ein Zeitfenster loesen
    [5a]  einen ganzen Run in Fenstern der Laenge alpha1 loesen,
          Artefakt (A)
  Stufe 2, Tracklets -> Trajektorien, ein Solve je Run:
    [6a]  Tracklets werden Knoten, dazu die Kandidatenpaare
    [6b]  Paar-Transformer trainieren, Kantenkosten von Stufe 2
    [6c]  Multicut ueber den ganzen Run, Partition je ON-Event
  [7]   Fit je Trajektorie: Kalman-Filter, RTS-Glaetter,
        Positionskorrektur, Spline
  Im Text heissen die Nummern Stufe [N]; Stufe 1 und Stufe 2 ohne
  Klammer sind die beiden Multicuts. [A] bis [D] in eckigen Klammern
  bezeichnen die Darstellungen in eventpiv.zeigen, keine Stufen.

BEGRIFFE
  Tracklet      Cluster von Stufe 1 in einem alpha1-Fenster, im Idealfall
                die Events eines Partikels; IDs je Run global.
  Trajektorie   Cluster von Stufe 2 ueber den ganzen Run, im Idealfall
                alle Tracklets eines Partikels.
  Bahn          die Events einer Trajektorie als Zeitreihe und ihr Fit
                aus [7]; GT-Bahn heissen die Boxmitten eines Partikels.
  Knoten        in Stufe 1 ein ON-Event, in Stufe 2 ein Tracklet. Ein
                kantenloser Knoten bekommt kein Cluster (Label -1).
  alpha1        neighborhood.alpha, µs: Reichweite zwischen Events und
                Laenge eines Stufe-1-Fensters.
  alpha2        kandidaten.alpha, µs: Reichweite zwischen den t_min
                zweier Tracklets. Beide heissen im Slug "a".
  JA / NEIN     Paar-Label: dasselbe Partikel bzw. verschiedene.
  Status        Verlaesslichkeit der GT-Zuordnung eines Events,
                kern.status.
  Slug          Name einer Parametergruppe im Artefaktpfad: lesbarer Kern
                plus Hash ueber alle Felder (kern.runconfig.slug).
  Arbeitspunkt  der ganze Parametersatz; die Vorgaben in kern.runconfig sind
                der von STELLA. wp/ traegt davon die Nachbarschaft.
  Betriebspunkt Lage der Schwelle zwischen Schneiden und Verbinden,
                gesetzt ueber cost_offset und sample_ratio.
  Artefakt (A)  das Tracklet-Artefakt aus [5a]: eine tracklet_id je Event
                des polaritaetsgefilterten Stroms, bei der Vorgabe je
                ON-Event (kette.tracklets.schreibe_tracklets).
  Stufe-1-Basislinie die Partition vor Stufe 2: jedes Tracklet ein eigener
                Cluster auf derselben Knotenmenge. Zeigt, wie weit Stufe 2 die
                Fragmentierung zuruecknimmt; keine Bewertung von Stufe 1.
  Anker         Regressionsanker in tests/test_anker.py: feste Kennzahlen
                eines Arbeitspunkts auf echten Daten.
  batch1        Datensatz mit gleich grossen GT-Boxen und 60-Hz-GT; seine
                Werte stehen in den batch1-Gruppen unter configs/.
  STELLA        synthetischer, oeffentlicher Datensatz der Arbeit STELLA
                (Sachs et al., 2026): polydispers, 1-kHz-GT.
"""

__version__ = "0.1.0"
