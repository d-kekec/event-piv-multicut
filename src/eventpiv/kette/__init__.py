"""kette -- die Verarbeitungskette, Stufe fuer Stufe.

  [1]   labels                Event -> track_id + Qualitaetsstatus
  [2]   pairs                 Paar-Datensatz aus den Labels
  [3]   model, train          Kantenkosten-Netz
  [4]   fenster, loeser, solve   ein Zeitfenster partitionieren
  [5a]  tracklets             einen Run abschreiten, Artefakt (A)
  [5b]  features              Merkmalstabelle, abgeleitet aus (A)

SCHICHTREGEL: kette importiert aus kern und bewerten, nie aus zeigen.

Innerhalb der Kette zeigt jede Abhaengigkeit rueckwaerts auf eine fruehere
Stufe -- [5a] auf [4], [4] auf [3]. Bewusst KEINE Unterpakete je Stufe: vier
der sechs Stufen haetten genau ein Modul, und wo Stufe 2 (Transformer)
ihre Grenzen zieht, steht noch nicht fest.
"""