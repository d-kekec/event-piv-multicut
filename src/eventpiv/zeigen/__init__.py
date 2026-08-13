"""zeigen -- Darstellung: Bilder und Animationen.

  visualize  Farben, Einzelbilder, Animationen ueber Runs und Fenster

SCHICHTREGEL: zeigen darf aus allen Schichten importieren, aber NICHTS darf
aus zeigen importieren. Darstellung ist die aeusserste Schicht -- eine
Rechnung, die weiss, wie ihr Ergebnis aussieht, ist nicht mehr ohne
Matplotlib pruefbar.

Deshalb steht hier nur ein Modul, und es soll dabei bleiben: was mehr als
eine Darstellung braucht, gehoert nach bewerten oder kern.
"""