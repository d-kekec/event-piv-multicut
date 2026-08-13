"""kern -- Grundlagen ohne Fachlogik der Kette.

  runconfig  Parametergruppen, Identitaet, Slugs
  config     Pfade und Metadaten -- einzige Stelle, die weiss WO etwas liegt
  data       Rohdaten laden (Events, Boxen)
  graph      Nachbarschaft in Raum und Zeit
  status     Qualitaetsstatus eines Events

SCHICHTREGEL: kern importiert nichts ausserhalb von kern. Diese Module
werden von mehreren Stufen geteilt -- graph von [2] und [4], status von [1]
und der Bewertung. Genau deshalb koennen sie in keiner Stufe wohnen.

Wer hier `from eventpiv.kette...` oder `from eventpiv.bewerten...` schreibt,
dreht die Abhaengigkeit um. tests/test_schichten.py meldet das.
"""