"""bewerten -- messen, was die Kette produziert hat.

  metrics   Paar-Ebene [3]: Konfusionsmatrix, Raten, AUC
  evaluate  Cluster-Ebene [4]: Reinheit, Fragmentierung, Merge, Kanten;
            Tracking-Masse [6c]: HOTA-Familie, IDF1, IDSW
  evaluate_trajektorien  Bahn-Ebene [7]: MedAE in Ort und Geschwindigkeit
  overlap   Runs nach Begegnungsdichte klassifizieren

SCHICHTREGEL: bewerten importiert nur aus kern. Eine Bewertung, die eine
Stufe der Kette importiert, ist nicht mehr unabhaengig von ihr -- genau
deshalb wohnen die Statuscodes in kern/status.py und nicht in labels.py.

Umgekehrt ist erlaubt und normal: fenster, solve, tracklets und train
importieren von hier. tests/test_schichten.py haelt die Richtung fest.
"""