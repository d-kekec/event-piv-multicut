"""kette -- die Verarbeitungskette, Stufe fuer Stufe.

  [1]   labels                    Event -> track_id + Qualitaetsstatus
  [2]   pairs                     Paar-Datensatz aus den Labels
  [3]   model, train              Kantenkosten-Netz
  [4]   fenster, loeser, solve    ein Zeitfenster partitionieren
  [5a]  tracklets                 einen Run abschreiten, Artefakt (A)
  [6a]  knoten, kandidaten        Tracklets werden Knoten, Kandidatenkanten
  [6b]  transformer, train2       Paar-Encoder, Kantenkosten von Stufe 2
  [6c]  solve2                    EIN Multicut ueber den ganzen Run

SCHICHTREGEL: kette importiert aus kern und bewerten, nie aus zeigen.

Innerhalb der Kette zeigt jede Abhaengigkeit rueckwaerts auf eine fruehere
Stufe -- [6a] auf [5a], [5a] auf [4]. Bewusst KEINE Unterpakete je Stufe:
die Modulnamen tragen die Zuordnung, ein Unterpaket brauchte dafuer eine
zweite Ebene ohne Gewinn.
"""